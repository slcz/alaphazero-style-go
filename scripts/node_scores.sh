#!/bin/bash
# node_scores.sh — per-game (plies, TT score) census -> compact JSON.
cd /workspace/go
python3 - <<'EOF'
import sys, json
sys.path.insert(0, '/workspace/go')
from go_ai.selfplay.shard import read_shard
from go_ai.engine import Board

games = read_shard('/workspace/go-run-9x9/it001/selfplay/shard-0.spbin')
out = []
for g in games:
    bd = Board(g.b, g.komi, g.n_plies)
    for p in g.plies:
        bd.push_fast(p.move)
    code, score, _ = bd.termination()
    out.append({"plies": len(g.plies), "score": round(float(score), 1)})
json.dump(out, open('/workspace/logs/it001_census.json', 'w'))
print("wrote", len(out), "games")
EOF
ls -la /workspace/logs/it001_census.json
