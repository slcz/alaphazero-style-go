#!/bin/bash
# node_variety.sh — variety census of the it001 selfplay shard.
cd /workspace/go
python3 - <<'EOF'
import sys, collections
sys.path.insert(0, '/workspace/go')
from go_ai.selfplay.shard import read_shard

games = read_shard('/workspace/go-run-9x9/it001/selfplay/shard-0.spbin')
n = len(games)
plies = [len(g.plies) for g in games]
sims = [sum(p.sims for p in g.plies) for g in games]

# outcome variety: TT margin per game, black view = sum of per-ply mover-view?0
# shard stores outlook (mover-view root value) — use move+count instead:
# unique move sequences and opening prefixes
open1 = collections.Counter()
open4 = collections.Counter()
seqs = set()
for g in games:
    ms = [p.move for p in g.plies]
    open1[ms[0]] += 1
    open4[tuple(ms[:4])] += 1
    seqs.add(tuple(ms))

import statistics as st
print(f"games={n}")
print(f"plies: mean={st.mean(plies):.1f} median={st.median(plies)} "
      f"min={min(plies)} max={max(plies)}")
print(f"total_sims: mean={st.mean(sims):.0f}")
print(f"unique first moves: {len(open1)} (top5: "
      f"{[(m, c) for m, c in open1.most_common(5)]})")
print(f"unique 4-move openings: {len(open4)}; top5 share = "
      f"{sum(c for _, c in open4.most_common(5)) / n:.2%}")
print(f"unique full move sequences: {len(seqs)} / {n}")
# length histogram buckets
buckets = collections.Counter(min(p // 50, 5) * 50 for p in plies)
print("ply buckets:", dict(sorted(buckets.items())))
EOF
