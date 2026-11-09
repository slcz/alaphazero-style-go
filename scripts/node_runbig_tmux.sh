#!/bin/bash
# start the large-board iteration in its own tmux session (gobig)
pkill -9 -f node_loopbig.sh 2>/dev/null
tmux kill-session -t gobig 2>/dev/null
sleep 1
tmux new-session -d -s gobig "bash /workspace/go/scripts/node_loopbig.sh"
echo "tmux gobig started; log /workspace/go-run-big/loop.out"
