#!/bin/bash
# node_restore3.sh — kill loop, re-place HF artifacts, relaunch at it2.
set -e
bash /workspace/go/scripts/node_cleanup.sh || true
W=/workspace/go-run-9x9
mkdir -p $W/it001/selfplay
[ -f /workspace/go-restore-train.pt ] && mv /workspace/go-restore-train.pt $W/it001/train.pt
[ -f /workspace/go-restore-shard.spbin ] && mv /workspace/go-restore-shard.spbin $W/it001/selfplay/shard-0.spbin
cp $W/it001/train.pt $W/best.pt
printf '{"n_workers":1,"unit_seed":2,"games_done":30000}\n' > $W/it001/selfplay/DONE.json
[ -f /workspace/go-restore-eval.json ] && mv /workspace/go-restore-eval.json $W/it001/eval.json
sha256sum $W/it001/selfplay/shard-0.spbin
bash /workspace/go/scripts/node_run9x9_tmux.sh 2
