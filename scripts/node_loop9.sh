#!/bin/bash
# node_loop9.sh — the supervisor: the 9x9 selfplay loop, forever.
#
# Every stage is resumable and each finished iteration is recorded in
# loop.jsonl, so a crash only costs the stage that was in flight: this
# restarts the loop, which resumes at the next unrecorded iteration and
# skips any iteration already backed up to HF (push.ok).
# HF_PUSH=0: this box cannot reach huggingface.co (resolves, times out),
# so the loop skips the upload and the Mac pulls the artifacts instead
# (scripts/hf_push_from_node.py). Set to 1 on a box with HF access.
export HF_PUSH=0
# Two-phase search: play purely for win/loss, then finish for points.
#   GO_SCORE_UTILITY=0    -> the early game backs up win/loss only
#   GO_SCORE_ONLY_V=0.95  -> once |value| >= 0.95 the result is settled, so
#                            the driver drops win/loss, clears the tree and
#                            finishes the game on the margin alone (clean
#                            endgames -> clean ownership/score labels)
export GO_SCORE_UTILITY=0.0
export GO_SCORE_ONLY_V=0.95
# --buffer-keep 0: train on the new round only (no history replay).
# Set it to 1/3 to bring back the decaying buffer.
cd /workspace/go || exit 1
export PUCT_WALKERS=4 PUCT_TASKS=8 NN_B=9 NN_BF16=1 NN_COMPILE=1
export NN_WAVE_STATS=1          # wave fill telemetry -> loop.out
WORK=/workspace/go-run-9x9
mkdir -p "$WORK"
while true; do
    # A killed loop leaves its stage child orphaned (pump() does not use a
    # process group), and a stray selfplay_nn would write the same shard as
    # the restarted one. Anything still alive here is an orphan.
    pkill -9 -f selfplay_nn 2>/dev/null
    pkill -9 -f engine_mux 2>/dev/null
    pkill -9 -f train_round.py 2>/dev/null
    sleep 2
    python3 scripts/selfplay_loop.py \
        --work-dir "$WORK" \
        --b 9 --komi 7.5 \
        --forever \
        --games 30000 \
        --sims 200 --fast-sims 40 --fast-p 0.75 \
        --pool 2048 --rows 4096 \
        --sample-plys 10 --max-plys 250 \
        --epochs 1 --batch 3072 --lr 0.01 \
        --buffer-keep 0 \
        --score-utility 0.0 \
        --gate-games 400 --gate-sims 200 \
        --device cuda --seed 1 \
        >> "$WORK/loop.out" 2>&1
    echo "[$(date +%H:%M:%S)] loop exited rc=$? — restarting in 30s" \
        >> "$WORK/loop.out"
    sleep 30
done
