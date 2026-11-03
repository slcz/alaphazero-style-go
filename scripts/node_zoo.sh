#!/bin/bash
echo "== all go-related processes (pid ppid stat etime cmd) =="
ps -eo pid,ppid,stat,etime,comm | grep -E "engine_mux|python3|tmux|selfplay|train" | grep -v grep
echo
echo "== python cmdlines =="
for p in $(pgrep python3); do
    echo "$p: $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-100)"
done
