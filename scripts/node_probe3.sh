#!/bin/bash
# node_probe3.sh — wave-fill probe: real config (2048 live games),
# hard-killed at 150s; WAVES stats are cumulative.
cd /workspace/go
export PUCT_WALKERS=4 PUCT_TASKS=8 NN_B=9 NN_BF16=1 NN_COMPILE=1 NN_WAVE_STATS=1
rm -rf /workspace/logs/probe9
timeout 150 ./engine/build/selfplay_nn /workspace/go/bench.pt /workspace/logs/probe9 \
    30000 9 7.5 200 2048 4096 10 250 4 cuda 1 40 0.75 > /workspace/logs/probe.raw 2>&1
echo "=== last WAVES ==="
grep WAVES /workspace/logs/probe.raw | tail -5
echo "=== wave count / games done ==="
grep -c WAVES /workspace/logs/probe.raw
grep -c "^GAME" /workspace/logs/probe.raw
