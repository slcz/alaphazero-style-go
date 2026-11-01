#!/bin/bash
# node_run9x9_tmux.sh — launch the 9x9 loop inside a tmux session so it
# survives ssh disconnects (the gateway reaps nohup'd children).
set -e
cd /workspace/go
WORK=/workspace/go-run-9x9
mkdir -p "$WORK"
ITERS="${1:-1}"

# kill the previous run ENTIRELY: the loop dies with its tmux session,
# but engine_mux / train_round children survive as orphans and hog the GPU
tmux kill-session -t go9 2>/dev/null || true
pkill -9 -f selfplay_loop.py 2>/dev/null || true
pkill -9 -f train_round.py 2>/dev/null || true
pkill -9 -f engine_mux 2>/dev/null || true
sleep 2
tmux new-session -d -s go9 "
cd /workspace/go
export PUCT_WALKERS=4 PUCT_TASKS=8 NN_B=9 NN_BF16=1 NN_COMPILE=1
export NN_WAVE_STATS=1   # wave fill telemetry -> generate.log (stderr)
python3 scripts/selfplay_loop.py \
    --work-dir $WORK \
    --b 9 --komi 7.5 \
    --iterations $ITERS \
    --games 30000 \
    --sims 200 --fast-sims 40 --fast-p 0.75 \
    --pool 2048 --rows 4096 \
    --sample-plys 10 --max-plys 250 \
    --epochs 1 --batch 3072 --lr 0.01 \
    --gate-games 400 --gate-sims 200 \
    --device cuda --seed 1 \
    > $WORK/loop.out 2>&1"
echo "tmux session go9 started; log: $WORK/loop.out"
