"""find_latest_checkpoint.py — newest `itNNN-train.pt` on the HF repo.

Used by the optional sync-net workflow to discover the checkpoint a
training run published. Works anonymously when the model repo is
public, or with HF_TOKEN (env or the repo's .env) when it is private.

    python scripts/find_latest_checkpoint.py \
        --repo chriszhang01/gonet --dir checkpoints \
        --json /tmp/ckpt.json --download /tmp/it026-train.pt

Exit codes: 0 success, 2 the repo is private/unauthorised.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def _get(url: str, token: str | None):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, timeout=60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="chriszhang01/gonet")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--dir", default="checkpoints",
                    help="directory on the model repo")
    ap.add_argument("--download", default=None,
                    help="download the checkpoint to this path")
    ap.add_argument("--json", default=None,
                    help="also write the metadata JSON here")
    args = ap.parse_args()

    token = _token()
    api = (f"https://huggingface.co/api/models/{args.repo}/tree/"
           f"{args.revision}/{args.dir}?recursive=true&limit=1000")
    try:
        with _get(api, token) as r:
            entries = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            print(f"cannot list {args.repo}/{args.dir}: HTTP {e.code}"
                  + ("" if token else " (private repo: set HF_TOKEN)"),
                  file=sys.stderr)
            return 2
        raise

    best = None
    for ent in entries:
        path = ent.get("path", "")
        m = re.search(r"(?:^|/)it(\d+)-train\.pt$", path)
        if not m:
            continue
        it = int(m.group(1))
        if best is None or it > best["iteration"]:
            best = {
                "iteration": it,
                "path": path,
                "size": ent.get("size") or ent.get("lfs", {}).get("size"),
                "url": (f"https://huggingface.co/{args.repo}/resolve/"
                        f"{args.revision}/{path}"),
                "blob_url": (f"https://huggingface.co/{args.repo}/blob/"
                             f"{args.revision}/{path}"),
            }
    if best is None:
        print(f"no itNNN-train.pt under {args.repo}/{args.dir}",
              file=sys.stderr)
        return 1

    text = json.dumps(best, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")
    if args.download:
        out = Path(args.download)
        out.parent.mkdir(parents=True, exist_ok=True)
        with _get(best["url"], token) as r, out.open("wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        got = out.stat().st_size
        if best["size"] and got != best["size"]:
            print(f"size mismatch: got {got}, expected {best['size']}",
                  file=sys.stderr)
            return 1
        print(f"downloaded {best['path']} -> {out} ({got / 2**20:.1f} MB)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
