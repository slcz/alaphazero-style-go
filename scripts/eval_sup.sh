#!/bin/bash
# eval_sup.sh — evaluate a supervised checkpoint: val metrics + a
# raw-policy gate match against a random-init net (strength floor).
# Usage: bash eval_sup.sh <ckpt>   (run from /workspace/go)
set -e
CKPT=${1:?ckpt}
VENV=.venv/bin/python

$VENV - <<EOF
import sys, pickle, time
import numpy as np, torch
sys.path.insert(0, ".")
from go_ai.train import supervised_data as SD
from go_ai.net.model import GoNet, NetConfig
from go_ai.train.trainer import Trainer, TrainConfig

# strength floor: untrained net, same architecture
torch.manual_seed(0)
net = GoNet(NetConfig())
Trainer(net, TrainConfig(device="cpu", compile=False)).save("sup/rand.pt")
print("rand.pt saved", flush=True)

# val metrics (same split as training: seed 1, val-frac 0.02)
games = pickle.load(open("data/kgs/games.pkl", "rb"))
h = np.random.default_rng(1).permutation(len(games))
n_val = max(1, int(len(games) * 0.02))
val_games = [games[i] for i in h[:n_val]]
val_rows = [(vi, i) for vi, g in enumerate(val_games) for i in range(len(g.moves))]
tr = Trainer.load("$CKPT", device="cuda")
rng = np.random.default_rng(7)
pick = rng.permutation(len(val_rows))[:8192]
B = 512
t1 = t5 = v = n = 0
with torch.no_grad():
    for s in range(0, len(pick), B):
        rows = [val_rows[i] for i in pick[s:s + B]]
        samples = [SD.supervised_sample(val_games[gi], i, int(rng.integers(0, 8)))
                   for gi, i in rows]
        bt = {k: torch.from_numpy(np.stack([x[k] for x in samples])).cuda()
              for k in samples[0]}
        out = tr.model(bt["tokens"], bt["mask"], bt["aux_mask"])
        pred = out["policy_logp"].topk(5, dim=-1).indices
        tgt = bt["pi_probs"].argmax(-1)
        t1 += (pred[:, 0] == tgt).sum().item()
        t5 += (pred == tgt[:, None]).any(-1).sum().item()
        v += (out["wdl"].argmax(-1) == bt["wdl"].argmax(-1)).sum().item()
        n += len(samples)
print(f"VAL top1 {t1/n:.3f} top5 {t5/n:.3f} value {v/n:.3f} n={n}", flush=True)
EOF

$VENV scripts/gate_match.py --ckpt-a "$CKPT" --ckpt-b sup/rand.pt \
    --a-sims 1 --b-sims 1 --b 9 --komi 7.5 --games 20 --device cuda
