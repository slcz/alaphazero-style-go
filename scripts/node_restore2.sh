#!/bin/bash
# node_restore2.sh — torch 2.11, place restored artifacts, launch it2.
set -e
echo "== torch upgrade =="
python3 -c "import torch; assert torch.__version__.startswith('2.11'), torch.__version__" 2>/dev/null || {
    pip3 install --quiet torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1
}
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "== place artifacts =="
W=/workspace/go-run-9x9
mkdir -p $W/it001/selfplay
[ -f /workspace/go-restore-train.pt ] && mv /workspace/go-restore-train.pt $W/it001/train.pt
[ -f /workspace/go-restore-shard.spbin ] && mv /workspace/go-restore-shard.spbin $W/it001/selfplay/shard-0.spbin
cp $W/it001/train.pt $W/best.pt
printf '{"n_workers":1,"unit_seed":2,"games_done":30000}\n' > $W/it001/selfplay/DONE.json
sha256sum $W/it001/train.pt $W/it001/selfplay/shard-0.spbin
ls -la $W $W/it001/selfplay

echo "== launch it2 =="
bash /workspace/go/scripts/node_run9x9_tmux.sh 2
