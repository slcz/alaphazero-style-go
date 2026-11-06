#!/bin/bash
# node_loop7.sh — zero-knowledge 7x7 run: random init, 30k games a round.
#
# --gate-min 0: promote EVERY round. On a cold start the gate is noise for
# the first few rounds (the base is a random net), and rejecting a 0.50
# round would keep the random net as the generator, so the data never
# improves and the run stalls. Raise this back to ~0.52 once the gate is
# clearly climbing.
# Board mix climbs with the run: equal 10k shares of 7x7, 9x9 and 11x11,
# same weights in one round. 7x7 is near done (cap 1.2%, dame 2.0) so its
# share stops growing; 11x11 was the weakest last round (dame 7.1, cap 10%).
# A fresh work dir means no best.pt, and selfplay_loop then bootstraps from
# a random-init checkpoint, so nothing from the 9x9/11-15 lineage leaks in.
# Utility is the FIXED blend lambda 0.8 (no two-phase switch: at 7x7 the
# endgame is a handful of moves and the switch bought nothing measurable),
# plus the settled-board discount on terminals.
export HF_PUSH=0
cd /workspace/go || exit 1
export PUCT_WALKERS=4 PUCT_TASKS=6 NN_B=7 NN_BF16=1 NN_COMPILE=1
export GO_UTIL_LAMBDA=0.8
export GO_SWITCH_V=0            # fixed blend, no score-phase switch
export GO_SCORE_FAST_SIMS=0
# Terminal dame charge: OFF, and settled. Four rules x both signs, all
# A/B'd on one fixed net (it009): dame 7x7/9x9 of
#   off 5.23/4.43 | k=1 sign-corrected 9.44/6.34 | k=1 sign-flipped 7.46/7.02
#   | sign-aware scale 5.27/5.77 | unconditional scale 6.16/-
# Every version is worse than off. The dame a game leaves behind is set by
# the play that leads there (score resolution), not by a terminal charge.
export GO_SETTLE_K=0
export GO_ROOT_NOISE_EPS=0
export NN_WAVE_STATS=1
WORK=/workspace/go-run-7x7
mkdir -p "$WORK"
while true; do
    pkill -9 -f selfplay_nn 2>/dev/null
    pkill -9 -f engine_mux 2>/dev/null
    pkill -9 -f train_round.py 2>/dev/null
    sleep 2
    python3 scripts/selfplay_loop.py \
        --work-dir "$WORK" \
        --b 7 --komi 9.5 \
        --forever \
        --games 30000 \
        --sizes 7:10000,9:10000,11:10000 \
        --sims 200 --fast-sims 40 --fast-p 0.75 \
        --pool 2048 --rows 4096 \
        --sample-plys 10 --max-plys 250 \
        --epochs 1 --batch 3072 --lr 0.01 \
        --buffer-keep 0 \
        --gate-games 300 --gate-sims 200 \
        --gate-sizes 7:9.5,9:7.5,11:7.5 \
        --gate-min 0.0 \
        --device cuda --seed 1 \
        >> "$WORK/loop.out" 2>&1
    echo "[$(date +%H:%M:%S)] loop exited rc=$? — restarting in 30s" \
        >> "$WORK/loop.out"
    sleep 30
done
