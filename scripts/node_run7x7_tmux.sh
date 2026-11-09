#!/bin/bash
# start the 7x7 loop in its own tmux session (go7)
pkill -9 -f node_loop7.sh 2>/dev/null
pkill -9 -f "selfplay_loop.py.*go-run-7x7" 2>/dev/null
tmux kill-session -t go7 2>/dev/null
sleep 1
tmux new-session -d -s go7 "bash /workspace/go/scripts/node_loop7.sh"
echo "tmux go7 started; log /workspace/go-run-7x7/loop.out"
