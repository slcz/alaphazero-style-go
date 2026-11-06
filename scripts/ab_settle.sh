#!/bin/bash
# ab_settle.sh — settle-rule A/B on one fixed net, run on the node:
#   bash scripts/ab_settle.sh <ckpt> <mode> [games] [board]
# modes: 0 off | 1 sign-aware | 2 unconditional | 3 separate dame term.
# Writes /tmp/abm<mode>_b<board> and prints a one-line summary per mode.
set -u
cd /workspace/go
CK=${1:?ckpt}; GAMES=${3:-2000}; B=${4:-7}
for M in 0 "$2"; do
    d=/tmp/abm${M}_b${B}; rm -rf $d
    if [ "$B" = 7 ]; then KO=9.5; MP=122; PO=2048; RW=4096
    else KO=7.5; MP=202; PO=2048; RW=4096; fi
    GO_SETTLE_MODE=$M GO_SETTLE_W=$([ "$M" = 0 ] && echo 0 || echo 1) \
      GO_UTIL_LAMBDA=0.8 GO_SWITCH_V=0 PUCT_WALKERS=4 \
      ./engine/build/selfplay_nn "$CK" $d $GAMES $B $KO 200 $PO $RW 10 $MP 4 \
      cuda 11 40 0.75 > $d.log 2>&1
    python3 - "$d" "$M" "$B" <<'PY'
import sys, statistics as st
sys.path.insert(0, "/workspace/go")
from go_ai.selfplay.shard import read_shard
d, m, b = sys.argv[1], sys.argv[2], sys.argv[3]
g = list(read_shard(f"{d}/shard-0.spbin"))
dame = [sum(1 for o in x.ownership if abs(o - 0.5) < 1e-6) for x in g]
cap = sum(1 for x in g if x.end_kind == 3)
print(f"mode={m} b={b}: dame {st.mean(dame):.2f} | w/dame "
      f"{100*sum(1 for x in dame if d)/len(g):.0f}% | plies "
      f"{st.median([x.n_plies for x in g]):.0f} | cap {100*cap/len(g):.1f}% "
      f"| n={len(g)}")
PY
done
