#!/bin/bash
# node_cleanup.sh — stop any running go loop on the salad node.
# WORK DIR IS KEPT (restore-from-HF is expensive). FORCE_RM=1 to wipe.
tmux kill-session -t go9 2>/dev/null
pkill -9 -f selfplay_loop.py
pkill -9 -f selfplay_nn
pkill -9 -f engine_mux
pkill -9 -f train_round.py
sleep 1
pgrep -f selfplay && echo "STILL RUNNING" || echo "clean"
[ "${FORCE_RM:-0}" = "1" ] && rm -rf /workspace/go-run-9x9 && echo "work dir removed"
