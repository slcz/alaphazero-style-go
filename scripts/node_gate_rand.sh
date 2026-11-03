#!/bin/bash
# node_gate_rand.sh — it2's net vs the regenerated random init (seed 1).
set -e
cd /workspace/go
pkill -9 -f selfplay_loop.py 2>/dev/null || true
pkill -9 -f engine_mux 2>/dev/null || true
pkill -9 -f gate_match 2>/dev/null || true
sleep 1

python3 - <<'EOF'
import sys
sys.path.insert(0, '/workspace/go')
import torch
from go_ai.net.model import GoNet, NetConfig
from go_ai.train.trainer import Trainer, TrainConfig
torch.manual_seed(1)
m = GoNet(NetConfig())
Trainer(m, TrainConfig(device='cpu', compile=False, seed=1)).save('/workspace/go-run-9x9/rand.pt')
print('rand.pt regenerated (seed 1)')
EOF

export PUCT_WALKERS=4 PUCT_SAMPLE_PLYS=10 NN_B=9 NN_BF16=1 NN_COMPILE=1
mkdir -p /workspace/logs/gate-rand
python3 scripts/gate_match.py \
    --ckpt-a /workspace/go-run-9x9/it002/train.pt \
    --ckpt-b /workspace/go-run-9x9/rand.pt \
    --games 200 --a-sims 200 --b-sims 200 \
    --b 9 --komi 7.5 --max-plys 200 --resign 0 \
    --device cuda --rows 800 --seed 424242 \
    --json /workspace/logs/gate-rand/eval.json \
    --sgf-dir /workspace/logs/gate-rand/sgf 2>&1 | tail -6
