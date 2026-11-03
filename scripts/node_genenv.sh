#!/bin/bash
echo "== env of live selfplay_nn =="
P=$(pgrep -f selfplay_nn | head -1)
tr '\0' '\n' < /proc/$P/environ | grep -E "NN_B|NN_COMPILE|NN_BF16|PUCT_TASKS|PUCT_WALKERS|NN_WAVE" | sort
echo "== quick wave probe: same config, 100s, stats on =="
cd /workspace/go
export PUCT_WALKERS=4 PUCT_TASKS=8 NN_B=9 NN_BF16=1 NN_COMPILE=1 NN_WAVE_STATS=1
rm -rf /workspace/logs/probe9
timeout 100 ./engine/build/selfplay_nn /workspace/go-run-9x9/it001/train.pt \
    /workspace/logs/probe9 3000 9 7.5 200 2048 4096 10 250 4 cuda 7 40 0.75 \
    2>&1 | grep -aE "WAVES|SUMMARY" | tail -5
