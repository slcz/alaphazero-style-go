#!/usr/bin/env python3
"""hf_sync.py — background HF sync for a running loop iteration (Mac-side;
the HF token never touches the node).

    python scripts/hf_sync.py 2 [--poll 120]

Pushes, as they appear/change on the node:
  itNNN/train.ck.pt    -> model  9x9/itNNN/train-partial.pt   (every 500 steps)
  itNNN shard          -> data   9x9/itNNN/shard-partial.spbin (every +1000 games)
  itNNN/train.pt, eval.json, DONE -> final artifacts (same paths as
                                     hf_push_iter; then exits)
Shard pulls mid-generation may catch a torn tail game — the final push
is authoritative."""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
env = dict(l.split("=", 1) for l in (REPO / ".env").read_text().splitlines()
           if "=" in l)
env = {k.strip(): v.strip() for k, v in env.items()}
SALAD = REPO / "scripts" / "salad.sh"
NODE = "/workspace/go-run-9x9"


def ssh(cmd: str) -> str:
    r = subprocess.run([str(SALAD), "ssh", cmd], capture_output=True,
                       text=True)
    return r.stdout


def get(remote: str, local: Path) -> bool:
    r = subprocess.run([str(SALAD), "get", remote, str(local)],
                       capture_output=True, text=True)
    return r.returncode == 0 and local.exists()


def games_done(it: int) -> int:
    out = ssh(f"grep -ac ^GAME {NODE}/it{it:03d}/generate.log")
    try:
        return int(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("iter", type=int)
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()
    it = args.iter
    itd = f"{NODE}/it{it:03d}"

    from huggingface_hub import HfApi
    api = HfApi(token=env["HF_TOKEN"])
    model, data = env["HF_MODEL_REPO"], env["HF_DATASET_REPO"]

    ck_sig, last_games = None, 0
    tmp = Path(tempfile.mkdtemp(prefix=f"hfsync-it{it:03d}-"))
    print(f"[hf_sync] it{it} polling every {args.poll}s", flush=True)
    while True:
        # training checkpoint: push when its (size,mtime) changed
        st = ssh(f"stat -c %s:%Y {itd}/train.ck.pt 2>/dev/null").strip()
        if st and st != ck_sig:
            f = tmp / "train.ck.pt"
            if get(f"{itd}/train.ck.pt", f):
                api.upload_file(path_or_fileobj=str(f),
                                path_in_repo=f"9x9/it{it:03d}/train-partial.pt",
                                repo_id=model, repo_type="model")
                ck_sig = st
                print(f"[hf_sync] pushed train-partial ({st})", flush=True)
        # shard: push every +1000 finished games
        n = games_done(it)
        if n - last_games >= 1000:
            f = tmp / "shard.spbin"
            if get(f"{itd}/selfplay/shard-0.spbin", f):
                api.upload_file(path_or_fileobj=str(f),
                                path_in_repo=f"9x9/it{it:03d}/shard-partial.spbin",
                                repo_id=data, repo_type="dataset")
                last_games = n
                print(f"[hf_sync] pushed shard-partial @ {n} games",
                      flush=True)
        # finals -> same as hf_push_iter, then exit
        if ssh(f"test -f {itd}/eval.json && echo Y").strip().endswith("Y"):
            f = tmp / "train.pt"
            if get(f"{itd}/train.pt", f):
                api.upload_file(path_or_fileobj=str(f),
                                path_in_repo=f"9x9/it{it:03d}-train.pt",
                                repo_id=model, repo_type="model")
                ev = tmp / "eval.json"
                if get(f"{itd}/eval.json", ev):
                    api.upload_file(path_or_fileobj=str(ev),
                                    path_in_repo=f"9x9/it{it:03d}-eval.json",
                                    repo_id=model, repo_type="model")
                print(f"[hf_sync] finals pushed; done", flush=True)
                return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
