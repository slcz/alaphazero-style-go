#!/bin/bash
# node_stop_after.sh — stop the forever loop once iteration N is recorded in
# loop.jsonl (i.e. generated, trained, gated and pushed), then optionally
# relaunch it so the supervisor picks up a changed launcher/environment.
#
#   bash scripts/node_stop_after.sh 7            # just stop
#   bash scripts/node_stop_after.sh 7 --restart  # stop, then it(N+1) starts
#
# Run it in its own tmux session (it must outlive the ssh connection).
set -u
W=/workspace/go-run-9x9
N=${1:?usage: node_stop_after.sh <iteration> [--restart]}
RESTART=${2:-}

while ! grep -q "\"iteration\": $N," "$W/loop.jsonl" 2>/dev/null; do
    sleep 20
done
echo "[$(date +%H:%M:%S)] it$N recorded — stopping"
sleep 5

pkill -9 -f node_loop9.sh          # the supervisor first, or it restarts
tmux kill-session -t go9 2>/dev/null
sleep 1
pkill -9 -f selfplay_loop.py
pkill -9 -f selfplay_nn
pkill -9 -f engine_mux
pkill -9 -f train_round.py

# the next round may already have started; it has no loop.jsonl record yet
NEXT=$(printf '%s/it%03d' "$W" $((N + 1)))
case "$NEXT" in
    "$W"/it*) [ -d "$NEXT" ] && rm -rf "$NEXT" && echo "removed $NEXT" ;;
esac
echo "[$(date +%H:%M:%S)] stopped after it$N" >> "$W/loop.out"

if [ "$RESTART" = "--restart" ]; then
    echo "[$(date +%H:%M:%S)] relaunching (it$((N + 1)))" >> "$W/loop.out"
    bash /workspace/go/scripts/node_run9x9_tmux.sh
fi
