#!/bin/bash
cd /workspace/go
python3 - <<'EOF'
import sys
sys.path.insert(0, '/workspace/go')
from go_ai.selfplay.shard import read_shard
from go_ai.train.selfplay_data import trainable_plies

games = read_shard('/workspace/go-run-9x9/it001/selfplay/shard-0.spbin')
print("total games:", len(games))
for gi in range(3):
    g = games[gi]
    t = trainable_plies(g)
    print(f"game[{gi}]: plies={len(g.plies)} n_sims={g.n_sims} "
          f"trainable={len(t)} first={t[:3]}")
zero = [gi for gi, g in enumerate(games) if not trainable_plies(g)]
print("games with zero trainable plies:", len(zero), "first few:", zero[:5])
g = games[zero[0]] if zero else None
if g is not None:
    print("example:", len(g.plies), "plies, sims list head:",
          [p.sims for p in g.plies][:8])
EOF
