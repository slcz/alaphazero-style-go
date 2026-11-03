#!/usr/bin/env python3
"""hf_watch.py — back up finished iterations to HF automatically.

Polls the training node's loop.jsonl (over the salad ssh helper) and, for
every record whose iteration has a checkpoint on the node and no local
`pushed/itNNN.ok` marker, runs hf_push_from_node.py. Needed because the
node itself cannot reach huggingface.co.

    python3 scripts/hf_watch.py [--interval 300]

Stop with: pkill -f hf_watch.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SALAD = REPO / "scripts" / "salad.sh"
MARK = REPO / ".pushed"
ROOT = "/workspace/go-run-9x9"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def remote(cmd: str) -> str:
    r = subprocess.run([str(SALAD), "ssh", cmd], capture_output=True, text=True)
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    args = ap.parse_args()
    MARK.mkdir(exist_ok=True)
    log(f"watching {ROOT} every {args.interval}s")
    while True:
        try:
            done = remote(f"cat {ROOT}/loop.jsonl")
            have = remote(f"ls {ROOT}")
            for line in done.splitlines():
                if not line.strip():
                    continue
                it = json.loads(line)["iteration"]
                tag = f"it{it:03d}"
                if (MARK / f"{tag}.ok").exists():
                    continue
                if f"{tag}\n" not in have:
                    continue
                ck = remote(f"test -f {ROOT}/{tag}/train.pt && echo yes")
                if "yes" not in ck:
                    continue
                log(f"{tag}: pushing")
                rc = subprocess.run([sys.executable,
                                     str(REPO / "scripts" /
                                         "hf_push_from_node.py"), str(it)],
                                    capture_output=True, text=True)
                if rc.returncode == 0:
                    (MARK / f"{tag}.ok").write_text("ok\n")
                    log(f"{tag}: pushed")
                else:
                    log(f"{tag}: FAILED ({rc.returncode}) "
                        f"{rc.stderr.strip().splitlines()[-1][:120]}")
        except Exception as e:
            log(f"watch error: {type(e).__name__}: {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
