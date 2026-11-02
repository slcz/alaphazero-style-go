#!/usr/bin/env python3
"""hf_push_from_node.py — back up one selfplay-loop iteration to HF from
the Mac, for nodes that cannot reach huggingface.co themselves.

    python3 scripts/hf_push_from_node.py 7 [--skip-shard]

Pulls the iteration's artifacts down with scripts/salad.sh get (scp — the
gateway's ssh banners corrupt redirected binaries), lays them out as a
shadow work dir, then calls scripts/hf_push_iter.py on it so the upload
layout is identical to the node-side path.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SALAD = REPO / "scripts" / "salad.sh"
ROOT = "/workspace/go-run-9x9"


def get(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(SALAD), "get", remote, str(local)],
                       capture_output=True, text=True)
    ok = r.returncode == 0 and local.exists() and local.stat().st_size > 0
    print(f"  {'ok  ' if ok else 'MISS'} {Path(remote).name:22} "
          f"{(f'{local.stat().st_size / 2**20:.1f} MB' if ok else '')}",
          flush=True)
    return ok


def run(remote_cmd: str) -> bool:
    """One command, no shell metacharacters."""
    r = subprocess.run([str(SALAD), "ssh", remote_cmd],
                       capture_output=True, text=True)
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("it", type=int)
    ap.add_argument("--skip-shard", action="store_true",
                    help="skip the games shard (323 MB) if bandwidth-bound")
    args = ap.parse_args()
    itd = f"{ROOT}/it{args.it:03d}"

    tmp = Path(tempfile.mkdtemp(prefix=f"go-pull-{args.it:03d}-"))
    shadow = tmp / f"it{args.it:03d}"
    print(f"pulling it{args.it:03d} from {ROOT}", flush=True)

    got = {}
    for name in ("train.pt", "train.json", "eval.json", "train.log"):
        got[name] = get(f"{itd}/{name}", shadow / name)
    if not args.skip_shard:
        got["shard"] = get(f"{itd}/selfplay/shard-0.spbin",
                           shadow / "selfplay" / "shard-0.spbin")
    # gate SGFs: tar on the node first (one round trip), then a single scp
    tgz_remote = f"{itd}/gate-sgf.tgz"
    if run(f"tar -czf {tgz_remote} -C {itd} gate-sgf"):
        get(tgz_remote, shadow / "gate-sgf.tgz")
    for name in ("loop.jsonl", "buffer.json"):
        get(f"{ROOT}/{name}", tmp / name)

    if not (shadow / "train.pt").exists():
        print("no checkpoint pulled; aborting", file=sys.stderr)
        return 1
    rc = subprocess.run([sys.executable,
                         str(REPO / "scripts" / "hf_push_iter.py"),
                         str(args.it), "--root", str(tmp)]).returncode
    shutil.rmtree(tmp, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
