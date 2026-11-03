#!/bin/bash
# node_probe2.sh — detached 64-game wave-fill probe; log /workspace/logs.
mkdir -p /workspace/logs
cd /workspace/go
export PUCT_WALKERS=4 PUCT_TASKS=8 NN_B=9 NN_BF16=1 NN_COMPILE=1 NN_WAVE_STATS=1
rm -rf /workspace/logs/probe9 /workspace/logs/probe.log
nohup ./engine/build/selfplay_nn /workspace/go/bench.pt /workspace/logs/probe9 \
    64 9 7.5 200 2048 4096 10 250 4 cuda 1 40 0.75 > /workspace/logs/probe.log 2>&1 &
echo "probe pid $! -> /workspace/logs/probe.log"
