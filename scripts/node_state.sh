#!/bin/bash
echo "== loop.out tail =="
tail -4 /workspace/go-run-9x9/loop.out 2>/dev/null
echo "== it001 dir =="
ls -la /workspace/go-run-9x9/it001/ 2>/dev/null
echo "== generate.log tail =="
tail -4 /workspace/go-run-9x9/it001/generate.log 2>/dev/null || echo "(no generate.log)"
echo "== procs =="
pgrep -f selfplay_loop | while read p; do ps -o pid,stat,etime -p $p | tail -1; done
pgrep -f selfplay_nn | while read p; do
  S=$(awk '/^State/{print $2}' /proc/$p/status 2>/dev/null)
  [ "$S" != "Z" ] && ps -o pid,stat,etime,cmd -p $p | tail -1 | cut -c1-90
done
echo "== gpu =="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
