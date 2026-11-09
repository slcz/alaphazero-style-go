#!/bin/bash
# node_loopbig.sh — zero-knowledge 7x7 run: random init, 30k games a round.
#
# --gate-min 0: promote EVERY round. On a cold start the gate is noise for
# the first few rounds (the base is a random net), and rejecting a 0.50
# round would keep the random net as the generator, so the data never
# improves and the run stalls. Raise this back to ~0.52 once the gate is
# clearly climbing.
# One iteration on large boards only. Expect ~4 h of generation and ~5 GB
# of shards; 19x19 dominates (each game ~360 plies, wall 902 < MAX_PLY).
# Utility: fixed blend lambda 0.8 (no phase switch), plain
# lambda*wdl + (1-lambda)*score. Seed best.pt from the trunk first:
#   cp /workspace/it023.pt /workspace/go-run-big/best.pt
export HF_PUSH=0
cd /workspace/go || exit 1
export PUCT_WALKERS=4 PUCT_TASKS=6 NN_B=15 NN_BF16=1 NN_COMPILE=1
export GO_UTIL_LAMBDA=0.8
export GO_SWITCH_V=0            # fixed blend, no score-phase switch
export GO_SCORE_FAST_SIMS=0
export GO_ROOT_NOISE_EPS=0
export NN_WAVE_STATS=1
WORK=/workspace/go-run-big
mkdir -p "$WORK"
while true; do
    pkill -9 -f selfplay_nn 2>/dev/null
    pkill -9 -f engine_mux 2>/dev/null
    pkill -9 -f train_round.py 2>/dev/null
    sleep 2
    python3 scripts/selfplay_loop.py \
        --work-dir "$WORK" \
        --b 19 --komi 7.5 \
        --forever \
        --games 30000 \
        --sizes 15:5000,17:5000,19:5000 \
        --sims 200 --fast-sims 40 --fast-p 0.75 \
        --pool 2048 --rows 4096 \
        --sample-plys 10 --max-plys 250 \
        --epochs 1 --batch 3072 --lr 0.01 \
        --buffer-keep 0 \
        --gate-games 150 --gate-sims 200 \
        --gate-sizes 15:7.5,17:7.5,19:7.5 \
        --gate-min 0.0 \
        --device cuda --seed 1 \
        >> "$WORK/loop.out" 2>&1
    echo "[$(date +%H:%M:%S)] loop exited rc=$? — restarting in 30s" \
        >> "$WORK/loop.out"
    sleep 30
done
