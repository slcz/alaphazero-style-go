#!/bin/bash
# node_run9x9_tmux.sh — launch the 9x9 loop inside a tmux session so it
# survives ssh disconnects (the gateway reaps nohup'd children).
#
# It runs FOREVER (scripts/node_loop9.sh supervises and restarts it) and
# every iteration backs itself up to HF: checkpoint + summary + gate SGFs
# to the model repo, the games shard to the dataset repo.
set -e
cd /workspace/go
WORK=/workspace/go-run-9x9
mkdir -p "$WORK"

# kill the previous run ENTIRELY: the loop dies with its tmux session,
# but engine_mux / train_round children survive as orphans and hog the GPU
tmux kill-session -t go9 2>/dev/null || true
pkill -9 -f selfplay_loop.py 2>/dev/null || true
pkill -9 -f node_loop9.sh 2>/dev/null || true
pkill -9 -f train_round.py 2>/dev/null || true
pkill -9 -f engine_mux 2>/dev/null || true
pkill -9 -f selfplay_nn 2>/dev/null || true
sleep 2
tmux new-session -d -s go9 "bash /workspace/go/scripts/node_loop9.sh"
echo "tmux session go9 started (forever); log: $WORK/loop.out"
tail -2 "$WORK/loop.out" 2>/dev/null || true
