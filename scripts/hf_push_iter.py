#!/usr/bin/env python3
"""hf_push_iter.py — back up one selfplay-loop iteration to HF.

    python3 scripts/hf_push_iter.py 3 [--root /workspace/go-run-9x9]
                                      [--rename gen0]

Runs ON the training node (selfplay_loop.py calls it right after the
gate), so nothing is pulled back to the Mac. HF_TOKEN comes from the
environment or <repo>/.env.

Model repo (chriszhang01/go-value-net @ 9x9/):
    itNNN-train.pt, itNNN-summary.json, itNNN-gate-sgf.tgz
Dataset repo (chriszhang01/go-selfplay @ 9x9/):
    itNNN/shard-0.spbin
The summary carries the train loss start/end, the head metrics and the
gate result; the SGF pack is one file per match game.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _env(key: str) -> str | None:
    import os
    v = os.environ.get(key)
    if v:
        return v
    f = REPO / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return None


def parse_train_log(path: Path) -> dict:
    """first/last 'step N/M ...' lines of train.log -> metrics dict."""
    steps = []
    for line in path.read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "step" and "/" in p[1]:
            def f(lbl):
                return float(p[p.index(lbl) + 1]) if lbl in p else None
            acc = None
            for tok, nxt in zip(p, p[1:]):
                if tok == "(acc":
                    acc = float(nxt.rstrip(")"))
            steps.append({
                "step": int(p[1].split("/")[0]), "steps_total":
                    int(p[1].split("/")[1]),
                "loss": f("loss"), "policy": f("policy"),
                "policy_acc": acc, "value": f("value"),
                "ownership": f("own"), "score": f("score"),
            })
    return {"start": steps[0], "end": steps[-1]} if steps else None


def pack_sgfs(itd: Path) -> Path | None:
    """gate-sgf/ -> gate-sgf.tgz (skipped when already packed)."""
    d = itd / "gate-sgf"
    tgz = itd / "gate-sgf.tgz"
    if tgz.exists():                  # already packed (pulled from the node)
        return tgz
    if not d.is_dir() or not any(d.iterdir()):
        return None
    if not tgz.exists():
        with tarfile.open(tgz, "w:gz") as t:
            t.add(d, arcname="gate-sgf")
    return tgz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("it", type=int)
    ap.add_argument("--root", default="/workspace/go-run-9x9")
    ap.add_argument("--rename", default=None,
                    help="also push the checkpoint under this name (gen0)")
    args = ap.parse_args()
    it, root = args.it, Path(args.root)
    itd = root / f"it{it:03d}"

    token = _env("HF_TOKEN")
    if not token:
        print("no HF_TOKEN", file=sys.stderr)
        return 1
    model_repo = _env("HF_MODEL_REPO") or "chriszhang01/go-value-net"
    data_repo = _env("HF_DATASET_REPO") or "chriszhang01/go-selfplay"
    ckpt = itd / "train.pt"
    if not ckpt.exists():
        print(f"no checkpoint at {ckpt}", file=sys.stderr)
        return 1

    summary = {"iteration": it, "board": 9, "komi": 7.5}
    tj = itd / "train.json"
    if tj.exists():
        tj = json.loads(tj.read_text())
        fin = tj.get("final", tj)
        summary["train_final"] = {k: fin.get(k) for k in
                                  ("policy_acc", "value_acc", "score",
                                   "elapsed_s")}
        summary["train"] = {k: tj.get(k) for k in
                            ("games", "trainable_plies", "steps",
                             "elapsed_s")}
        summary["train_final"].pop("elapsed_s", None)
    tl = itd / "train.log"
    if tl.exists():
        summary["train_progress"] = parse_train_log(tl)
    ev = itd / "eval.json"
    if ev.exists():
        e = json.loads(ev.read_text())
        summary["gate"] = {k: e.get(k) for k in
                           ("games", "w", "l", "score_a", "waves",
                            "a_sims", "b_sims", "seed")}
    buf = root / "buffer.json"
    if buf.exists():
        summary["buffer"] = [(r["round"], len(r["games"]))
                             for r in json.loads(buf.read_text())["rounds"]]
    sp = itd / "summary.json"
    sp.write_text(json.dumps(summary, indent=2))

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    def up(local: Path, repo_path: str, repo_id: str, kind: str) -> None:
        print(f"  {local.name} -> {repo_id}:{repo_path} "
              f"({local.stat().st_size / 2**20:.0f} MB)", flush=True)
        api.upload_file(path_or_fileobj=str(local), path_in_repo=repo_path,
                        repo_id=repo_id, repo_type=kind)

    print(f"pushing it{it:03d} from {itd}", flush=True)
    up(ckpt, f"9x9/it{it:03d}-train.pt", model_repo, "model")
    up(sp, f"9x9/it{it:03d}-summary.json", model_repo, "model")
    tgz = pack_sgfs(itd)
    if tgz:
        up(tgz, f"9x9/it{it:03d}-gate-sgf.tgz", model_repo, "model")
    if args.rename:
        up(ckpt, f"9x9/{args.rename}.pt", model_repo, "model")
        up(sp, f"9x9/{args.rename}-summary.json", model_repo, "model")
    lj = root / "loop.jsonl"
    if lj.exists():
        up(lj, "9x9/loop.jsonl", model_repo, "model")
    # one unit dir per board size when the round mixes sizes
    shards = ([itd / "selfplay" / "shard-0.spbin"]
              if (itd / "selfplay" / "shard-0.spbin").exists()
              else sorted(itd.glob("sp*/shard-0.spbin")))
    for shard in shards:
        if shard.exists():
            up(shard, f"9x9/it{it:03d}/{shard.parent.name}/shard-0.spbin",
               data_repo, "dataset")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
