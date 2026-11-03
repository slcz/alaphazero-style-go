#!/bin/bash
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

def batch_of(bs):
    B = None
    for j in range(bs):
        gi, i = rows[rng.integers(0, len(rows))]
        s = selfplay_sample(games[gi], i)
        if B is None:
            B = {k: np.zeros((bs,) + v.shape, v.dtype) for k, v in s.items()}
        for k, v in s.items():
            B[k][j] = v
    return {k: torch.from_numpy(v).to("cuda") for k, v in B.items()}

for bs in (2048, 3072, 1024):
    try:
        t = Trainer(GoNet(NetConfig()),
                    TrainConfig(batch_size=bs, device="cuda", compile=False))
        m = t.train_step(batch_of(bs))
        torch.cuda.synchronize()
        print("batch", bs, "OK loss=%.4f" % m["loss"], flush=True)
    except Exception as e:
        print("batch", bs, "FAILED:", str(e)[:100], flush=True)
    del t
    torch.cuda.empty_cache()
EOF
