#!/usr/bin/env python3
"""hf_push_iter.py — back up one selfplay-loop iteration to HF.

    python scripts/hf_push_iter.py 1 --rename gen0

Pulls from the salad node (scripts/salad.sh get, scp — the ssh-cat route
corrupts binaries): train.pt, train.json, eval.json, train.log, the
selfplay shard. Pushes to the model repo (checkpoint + summary json) and
the dataset repo (games shard). Token stays local (.env)."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
env = {}
for line in (REPO / ".env").read_text().splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

NODE_ROOT = "/workspace/go-run-9x9"
SALAD = REPO / "scripts" / "salad.sh"


def node_get(remote: str, local: Path) -> bool:
    r = subprocess.run([str(SALAD), "get", remote, str(local)],
                       capture_output=True, text=True)
    return r.returncode == 0 and local.exists()


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


def main() -> int:
    it = int(sys.argv[1])
    rename = None
    if "--rename" in sys.argv:
        rename = sys.argv[sys.argv.index("--rename") + 1]
    from huggingface_hub import HfApi
    api = HfApi(token=env["HF_TOKEN"])
    model_repo = env["HF_MODEL_REPO"]
    data_repo = env["HF_DATASET_REPO"]

    itd = f"{NODE_ROOT}/it{it:03d}"
    tmp = Path(tempfile.mkdtemp(prefix=f"hf-it{it:03d}-"))
    pulls = {
        "train.pt": f"{itd}/train.pt",
        "train.json": f"{itd}/train.json",
        "eval.json": f"{itd}/eval.json",
        "train.log": f"{itd}/train.log",
        "shard.spbin": f"{itd}/selfplay/shard-0.spbin",
        # gate SGFs too (small, useful for eyeballing match play)
        "gate-sgf.tgz": f"{itd}/gate-sgf.tgz",
    }
    files = {}
    for name, remote in pulls.items():
        ok = node_get(remote, tmp / name)
        print(f"  {name}: {'ok' if ok else 'MISSING'}", flush=True)
        if ok:
            files[name] = tmp / name
    if "train.pt" not in files:
        print("no checkpoint on node; aborting", file=sys.stderr)
        return 1

    summary = {"iteration": it, "board": 9, "komi": 7.5}
    if "train.json" in files:
        tj = json.loads(files["train.json"].read_text())
        tj = tj.get("final", tj)
        summary["train_final"] = {k: tj.get(k) for k in
                                  ("policy_acc", "value_acc", "score_pdf",
                                   "score_cdf", "elapsed_s")}
    if "train.log" in files:
        summary["train_progress"] = parse_train_log(files["train.log"])
    if "eval.json" in files:
        ev = json.loads(files["eval.json"].read_text())
        summary["gate"] = {k: ev.get(k) for k in
                           ("games", "w", "l", "score_a", "waves",
                            "a_sims", "b_sims", "seed")}
    else:
        summary["gate"] = None
    sp = tmp / "summary.json"
    sp.write_text(json.dumps(summary, indent=2))

    print("pushing model repo...", flush=True)
    api.upload_file(path_or_fileobj=files["train.pt"],
                    path_in_repo=f"9x9/it{it:03d}-train.pt",
                    repo_id=model_repo, repo_type="model")
    api.upload_file(path_or_fileobj=sp,
                    path_in_repo=f"9x9/it{it:03d}-summary.json",
                    repo_id=model_repo, repo_type="model")
    if rename:
        api.upload_file(path_or_fileobj=files["train.pt"],
                        path_in_repo=f"9x9/{rename}.pt",
                        repo_id=model_repo, repo_type="model")
        api.upload_file(path_or_fileobj=sp,
                        path_in_repo=f"9x9/{rename}-summary.json",
                        repo_id=model_repo, repo_type="model")
        print(f"aliased -> 9x9/{rename}.pt")
    if "shard.spbin" in files:
        print("pushing dataset repo (games)...", flush=True)
        api.upload_file(path_or_fileobj=files["shard.spbin"],
                        path_in_repo=f"9x9/it{it:03d}/shard-0.spbin",
                        repo_id=data_repo, repo_type="dataset")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
