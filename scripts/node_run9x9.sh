#!/bin/bash
# node_run9x9.sh — launch the 9x9 gen-from-scratch loop on the salad
# node (nohup; logs under the work dir). Run: bash scripts/node_run9x9.sh
set -e
cd /workspace/go
WORK=/workspace/go-run-9x9
mkdir -p "$WORK"

export PUCT_WALKERS=4
export PUCT_TASKS=8      # mux walk-task pool (tree walking is CPU-bound)
export NN_B=9            # FP8 static-wave inference path (nn_worker setup)
export NN_BF16=1
export NN_COMPILE=1

nohup python3 scripts/selfplay_loop.py \
    --work-dir "$WORK" \
    --b 9 --komi 7.5 \
    --iterations 1 \
    --games 30000 \
    --sims 200 --fast-sims 40 --fast-p 0.75 \
    --pool 2048 --rows 4096 \
    --sample-plys 10 --max-plys 250 \
    --epochs 1 --batch 4096 --lr 2e-4 \
    --gate-games 400 --gate-sims 200 \
    --device cuda --seed 1 \
    > "$WORK/loop.out" 2>&1 &
echo $! > "$WORK/loop.pid"
echo "launched pid $(cat "$WORK/loop.pid"); log: $WORK/loop.out"
