#!/bin/bash
# node_upgrade.sh — stop loop, upgrade torch (cu128), verify + retest
# the batch-4096 bf16 GEMM that failed on 2.8.0.
tmux kill-session -t go9 2>/dev/null
pkill -9 -f selfplay_loop.py 2>/dev/null
pkill -9 -f selfplay_nn 2>/dev/null
pkill -9 -f train_round 2>/dev/null
sleep 1
echo "== pip upgrade =="
pip3 install --upgrade torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "== batch-4096 bf16 train_step retest =="
cd /workspace/go
python3 - <<'EOF'
import sys
sys.path.insert(0, '/workspace/go')
import numpy as np, torch
from go_ai.selfplay.shard import read_shard
from go_ai.train.selfplay_data import trainable_plies, selfplay_sample
from go_ai.net.model import GoNet, NetConfig
from go_ai.train.trainer import Trainer, TrainConfig

games = read_shard('/workspace/go-run-9x9/it001/selfplay/shard-0.spbin')
rows = [(gi, i) for gi, g in enumerate(games) for i in trainable_plies(g)]
rng = np.random.default_rng(1)
B = None
for j in range(4096):
    gi, i = rows[rng.integers(0, len(rows))]
    s = selfplay_sample(games[gi], i)
    if B is None:
        B = {k: np.zeros((4096,) + v.shape, v.dtype) for k, v in s.items()}
    for k, v in s.items():
        B[k][j] = v
batch = {k: torch.from_numpy(v).to("cuda") for k, v in B.items()}
t = Trainer(GoNet(NetConfig()),
            TrainConfig(batch_size=4096, device="cuda", compile=False,
                        optimizer="muon"))
m = t.train_step(batch)
torch.cuda.synchronize()
print("batch 4096 muon OK loss=%.4f" % m["loss"])
EOF
