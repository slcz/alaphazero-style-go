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
# --buffer-keep 0: train on the new round only (no history replay).
# Set it to 1/3 to bring back the decaying buffer.
cd /workspace/go || exit 1
export PUCT_WALKERS=4 PUCT_TASKS=6 NN_B=9 NN_BF16=1 NN_COMPILE=1
# Two-phase utility. A game is played PURELY for the result (lambda 1.0)
# until the value head is sure, then PURELY for the margin (lambda 0.0).
# Early: no margin-chasing in the opening. Late: the search still
# discriminates in decided positions, where win/loss saturates and every
# move would otherwise look equal (the eye-fill blunder). The switch clears
# the tree, because stored Q values are in the old units.
export GO_UTIL_LAMBDA=1.0
# Root Dirichlet noise (selfplay only): OFF. it016 ran with eps 0.25 and
# gated 0.5075, i.e. exploration was not what was limiting progress.
# 0.95, not 0.98: with a cheap tail the threshold is not about being sure
# before paying for search, it is about how many games get a score-driven
# endgame at all. 89 % switch at 0.95 vs 85 % at 0.98, and the 0.90-0.98
# band would otherwise play its endgame with a saturated value head.
export GO_SWITCH_V=0.95
# Tail at the normal budget (0 = off): the decided part is played out with
# full search and its slow-cap plies are trained, so the endgame policy keeps
# learning. Set 4 to make the tail cheap again.
export GO_SCORE_FAST_SIMS=0
export GO_ROOT_NOISE_EPS=0
# TOTAL concentration over the move list, KataGo's 10.83 — per move that is
# ~0.13 on 9x9 (~0.03 on 19x19), so the same number works at every size.
export GO_ROOT_NOISE_TOTAL=10.83
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
    # --max-plys above is only a fallback: the wall is 2.5*b*b per size,
    # computed in size_plans (202 for 9x9, 562 for 15x15).
    python3 scripts/selfplay_loop.py \
        --work-dir "$WORK" \
        --b 9 --komi 7.5 \
        --forever \
        --games 30000 \
        --sizes 9:15000,11:8000,13:5000,15:2000 \
        --sims 200 --fast-sims 40 --fast-p 0.75 \
        --pool 2048 --rows 4096 \
        --sample-plys 10 --max-plys 250 \
        --epochs 1 --batch 3072 --lr 0.01 \
        --buffer-keep 0 \
        --gate-games 400 --gate-sims 200 \
        --device cuda --seed 1 \
        >> "$WORK/loop.out" 2>&1
    echo "[$(date +%H:%M:%S)] loop exited rc=$? — restarting in 30s" \
        >> "$WORK/loop.out"
    sleep 30
done
