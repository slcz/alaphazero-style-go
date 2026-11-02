#!/bin/bash
# node_restore_fresh.sh — fresh-round restore: the newest checkpoint from
# HF becomes best.pt, no shards, no replay buffer. The loop resumes at the
# next iteration and, with --buffer-keep 0, trains on the new round only.
#
#   bash scripts/node_restore_fresh.sh     # on the node, after the repo
#                                          # is shipped + engine built
set -e
cd /workspace/go
set -a; source .env; set +a
: "${HF_TOKEN:?HF_TOKEN missing}"
W=/workspace/go-run-9x9
mkdir -p "$W"

python3 - <<'PY'
import json, os, re, shutil
from huggingface_hub import HfApi, hf_hub_download

tok = os.environ["HF_TOKEN"]
repo = os.environ.get("HF_MODEL_REPO", "chriszhang01/go-value-net")
W = "/workspace/go-run-9x9"
api = HfApi(token=tok)

files = set(api.list_repo_files(repo_id=repo, revision="main"))
its = sorted(int(m.group(1)) for f in files
             for m in [re.match(r"9x9/it(\d{3})-train\.pt$", f)] if m)
assert its, "no 9x9/itNNN-train.pt in the model repo"
n = its[-1]
ck = hf_hub_download(repo_id=repo, filename=f"9x9/it{n:03d}-train.pt", token=tok)
shutil.copy(ck, f"{W}/best.pt")
print(f"best.pt <- 9x9/it{n:03d}-train.pt")

lp = f"{W}/loop.jsonl"
if "9x9/loop.jsonl" in files:
    shutil.copy(hf_hub_download(repo_id=repo, filename="9x9/loop.jsonl",
                                token=tok), lp)
    have = max(json.loads(l)["iteration"] for l in open(lp) if l.strip())
    print(f"loop.jsonl <- HF (records 1..{have})")
else:
    with open(lp, "w") as fh:
        for i in range(1, n + 1):
            fh.write(json.dumps({"iteration": i,
                                 "note": "restored stub: stage state unknown"})
                     + "\n")
    print(f"loop.jsonl <- stubs 1..{n} (no copy on HF)")
    have = n
print(f"loop resumes at it{max(n, have) + 1:03d}")
PY

echo "== work dir =="
ls -la "$W"
sha256sum "$W/best.pt"
echo "launch with: bash scripts/node_run9x9_tmux.sh   (--buffer-keep 0 = fresh round)"
