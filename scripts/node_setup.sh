#!/bin/bash
# node_setup.sh — one-time setup on the salad node: deps check, engine
# build, small CUDA selfplay smoke run. Run: bash scripts/node_setup.sh
set -e
cd /workspace/go

echo "== torch =="
python3 -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "== disk =="
df -h /workspace | tail -1
echo "== cores =="
nproc

echo "== build engine =="
make -C engine all nn 2>&1 | grep -E "error|warning" || true
test -x engine/build/selfplay_nn || { echo "BUILD FAILED"; exit 1; }
echo "build ok"

echo "== random-init smoke ckpt =="
python3 -c "
import sys; sys.path.insert(0, '/workspace/go')
import torch
from go_ai.net.model import GoNet, NetConfig
from go_ai.train.trainer import Trainer, TrainConfig
torch.manual_seed(1)
m = GoNet(NetConfig())
Trainer(m, TrainConfig(device='cpu', compile=False, seed=1)).save('/workspace/go-smoke.pt')
print('ckpt ok', sum(p.numel() for p in m.parameters()), 'params')"

echo "== smoke: 64 games 9x9 cuda (sims 60/40, pool 64) =="
rm -rf /workspace/go-smoke
PUCT_WALKERS=4 timeout 900 ./engine/build/selfplay_nn /workspace/go-smoke.pt \
    /workspace/go-smoke 64 9 7.5 60 64 256 10 250 4 cuda 7 40 0.75 \
    2>&1 | tail -4
test -f /workspace/go-smoke/shard-0.spbin || { echo "NO SHARD"; exit 1; }
echo "SMOKE OK"
