#!/bin/bash
echo "selfplay procs:"
for p in $(pgrep -f selfplay_nn); do
    S=$(awk '/^State/{print $2}' /proc/$p/status 2>/dev/null)
    echo "  $p state=$S"
done
echo "logs:"
ls -la /workspace/logs/
echo "nohup.out:"
tail -3 /workspace/go/nohup.out 2>/dev/null
