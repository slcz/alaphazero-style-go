#!/usr/bin/env python3
"""hf_push_dataset.py — back up a selfplay shard directory to HF.

    python scripts/hf_push_dataset.py data/gen0 gen0

Uploads DONE-gated units (shard + DONE.json) as
<repo>/<name>/unit-*/... on HF_DATASET_REPO. Resumable: unchanged files
are skipped by HF's dedup."""
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[1]
env = {}
for line in (REPO / ".env").read_text().splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

src, name = Path(sys.argv[1]), sys.argv[2]
files = sorted(p for p in src.rglob("*") if p.is_file())
api = HfApi(token=env["HF_TOKEN"])
api.create_repo(env["HF_DATASET_REPO"], repo_type="dataset", exist_ok=True)
for p in files:
    rel = p.relative_to(src)
    api.upload_file(path_or_fileobj=p,
                    path_in_repo=f"{name}/{rel}",
                    repo_id=env["HF_DATASET_REPO"], repo_type="dataset")
    print(f"uploaded {rel}")
print(f"done: {len(files)} files -> {env['HF_DATASET_REPO']}/{name}")
