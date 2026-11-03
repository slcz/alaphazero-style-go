#!/bin/bash
L=/workspace/go-run-9x9/it002/generate.log
echo "== it002 generate.log =="
ls -la /workspace/go-run-9x9/it002/ 2>/dev/null
tail -3 $L 2>/dev/null
A=$(grep -ac '^GAME' $L 2>/dev/null || echo 0)
sleep 45
B=$(grep -ac '^GAME' $L 2>/dev/null || echo 0)
echo "games: $A -> $B in 45s"
echo "== live procs =="
for p in $(pgrep -f selfplay_nn); do
  S=$(awk '/^State/{print $2}' /proc/$p/status 2>/dev/null)
  [ "$S" = "Z" ] && continue
  ps -o pid,etime,%cpu -p $p | tail -1
done
echo "== gpu =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
echo "== cpu load =="
uptime
