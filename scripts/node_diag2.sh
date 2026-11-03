#!/bin/bash
cd /workspace/go
echo "== full train.log =="
cat /workspace/go-run-9x9/it001/train.log | grep -vE "^\s+File|^\s{4,}|^Traceback" | head -20
echo "== minimal train_step repro =="
python3 - <<'EOF'
import sys, traceback
sys.path.insert(0, '/workspace/go')
import torch
from go_ai.net.model import GoNet, NetConfig
from go_ai.train.trainer import Trainer, TrainConfig

# build the same trainer the loop uses, tiny batch, see if ONE step works
cfg = TrainConfig(batch_size=64, device="cuda", compile=False, bf16=True)
t = Trainer(GoNet(NetConfig()), cfg)
B, T = 64, 81
x = torch.randn(B, T, 4, device="cuda")
m = torch.zeros(B, T + 1, device="cuda"); m[:, :T] = 1
import go_ai.net.features as F
# build a dummy batch from the real sample proto instead: shapes via model
try:
    out = t.model(x, m)
    loss = sum(v.sum() for v in out.values() if v.is_floating_point())
    loss.backward()
    torch.cuda.synchronize()
    print("tiny eager step OK", {k: tuple(v.shape) for k, v in out.items()})
except Exception:
    traceback.print_exc()
    print("tiny eager step FAILED")
EOF
