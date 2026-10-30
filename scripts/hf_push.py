#!/usr/bin/env python3
"""hf_push.py — upload a checkpoint to the HF model repo.

    python scripts/hf_push.py sup/base.pt supervised/base-1ep.pt

HF_TOKEN / HF_MODEL_REPO come from .env. Runs locally so the token
never lands on a training box."""
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[1]
env = {}
for line in (REPO / ".env").read_text().splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

ckpt, remote = sys.argv[1], sys.argv[2]
api = HfApi(token=env["HF_TOKEN"])
api.create_repo(env["HF_MODEL_REPO"], exist_ok=True)
url = api.upload_file(path_or_fileobj=ckpt, path_in_repo=remote,
                      repo_id=env["HF_MODEL_REPO"], repo_type="model")
print(f"uploaded -> {url}")
