#!/usr/bin/env python3
"""elo_ladder.py — standard 19x19 ladder over the HF gonet checkpoints.

For each consecutive pair of checkpoints (it001 vs it002, it002 vs it003,
...; iterations need not be contiguous — the sort order of the present
itNNN names defines the chain), runs a standard match: 25 games with the
older net as Black + 25 with it as White (batch_match's two waves),
200 sims/move, komi 7.5, ply wall 900, the repo's standard resignation
rule. Each pair's result is uploaded to HF as
checkpoints/matches/itXXX-itYYY-match.json, and after the full ladder a
chained Elo (it001 anchored at 1000, 400-pt log-odds steps) is written to
checkpoints/elo.json and uploaded too.

    python3 elo_ladder.py [--dry-run] [--only a:b,...]
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/workspace/gogui")
CKPTS = REPO / "ckpts" / "checkpoints"    # snapshot_download keeps repo layout
MATCHES = REPO / "matches"             # local match.json output
DRIVER = REPO / "scripts" / "batch_match.sh"

REPO_ID = os.environ.get("HF_MODEL_REPO", "chriszhang01/gonet")
GAMES_PER_WAVE = 25                    # x2 waves: 25 white + 25 black
SIMS = 200
KOMI = 7.5
MAX_PLYS = 900                         # 2.5*b^2, the selfplay convention
ELO_ANCHOR = 1000.0


def ckpt_list():
    """Sorted distinct iteration tags that have a checkpoint on disk."""
    tags = sorted({m.group(1) for p in CKPTS.glob("it*-train.pt")
                   if (m := re.match(r"(it\d+)", p.name))})
    if len(tags) < 2:
        sys.exit(f"need >=2 checkpoints under {CKPTS}, found {tags}")
    return tags


def upload(local: Path, remote_name: str):
    from huggingface_hub import HfApi
    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
        path_or_fileobj=str(local), repo_id=REPO_ID,
        path_in_repo=f"checkpoints/matches/{remote_name}")


def run_pair(a: str, b: str, out: Path):
    argv = [sys.executable, str(DRIVER),
            "--ckpt-a", str(CKPTS / f"{a}-train.pt"),
            "--ckpt-b", str(CKPTS / f"{b}-train.pt"),
            "--b", "19", "--komi", str(KOMI),
            "--games", str(GAMES_PER_WAVE),
            "--a-sims", str(SIMS), "--b-sims", str(SIMS),
            "--max-plys", str(MAX_PLYS),
            "--device", "cuda",
            "--seed", str((int(a[2:]) * 1000003 + int(b[2:])) % (1 << 30)),
            "--json", str(out)]
    print(f"RUN {' '.join(argv[1:])}", flush=True)
    return subprocess.run(argv, cwd=str(REPO), text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None,
                    help="restrict to 'it001:it002,it002:it003' pairs")
    args = ap.parse_args()
    os.environ.setdefault("HF_TOKEN", "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")

    tags = ckpt_list()
    pairs = list(zip(tags, tags[1:]))
    if args.only:
        want = {tuple(p.split(":")) for p in args.only.split(",")}
        pairs = [p for p in pairs if p in want]
    print(f"ladder: {len(pairs)} pairs over {tags}", flush=True)

    MATCHES.mkdir(parents=True, exist_ok=True)
    results = {}
    for i, (a, b) in enumerate(pairs):
        out = MATCHES / f"{a}-{b}-match.json"
        if out.exists():
            print(f"SKIP {a} vs {b} — {out.name} exists", flush=True)
        else:
            if args.dry_run:
                print(f"DRY {a} vs {b}", flush=True)
                continue
            t0 = time.time()
            r = run_pair(a, b, out)
            if r.returncode != 0 or not out.exists():
                sys.exit(f"match {a} vs {b} failed (rc={r.returncode})")
            print(f"DONE {a} vs {b} in {time.time()-t0:.0f}s", flush=True)
            upload(out, out.name)
            print(f"UP  {out.name}", flush=True)
        results[f"{a}:{b}"] = __import__("json").loads(out.read_text())

    # chained Elo: it001 = 1000; each pair shifts the next rung by the
    # log-odds of the OLDER net's score (positive = the newer net gained)
    elo = {tags[0]: ELO_ANCHOR}
    for key, res in results.items():
        a, b = key.split(":")
        s = res["score_a"]
        n = res["games"]
        if s in (0.0, 1.0):
            s = min(max(s, 0.5 / n), 1 - 0.5 / n)     # clip perfect scores
        delta = 400.0 * math.log10(s / (1 - s))
        err = 400.0 / math.log(10) * math.sqrt(s * (1 - s) / n)
        elo[b] = elo[a] - delta
        results[f"{a}:{b}"]["elo_a"] = elo[a]
        results[f"{a}:{b}"]["elo_b"] = elo[b]
        results[f"{a}:{b}"]["elo_stderr"] = err

    summary = {
        "anchor": f"{tags[0]}={ELO_ANCHOR}",
        "config": {"games_per_wave": GAMES_PER_WAVE, "sims": SIMS,
                   "komi": KOMI, "max_plys": MAX_PLYS, "b": 19,
                   "driver": "scripts/batch_match.sh"},
        "elo": dict(sorted(elo.items())),
        "pairs": results,
    }
    out = MATCHES / "elo.json"
    out.write_text(__import__("json").dumps(summary, indent=2))
    print(out.read_text()[:800], flush=True)
    upload(out, "elo.json")
    print("UP elo.json — ladder complete", flush=True)


if __name__ == "__main__":
    main()
