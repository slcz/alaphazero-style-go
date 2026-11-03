#!/bin/bash
echo "== train.log tail =="
tail -8 /workspace/go-run-9x9/it001/train.log 2>/dev/null
echo "== gpu procs =="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "== gpu =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "== live go procs =="
pgrep -f selfplay_loop | while read p; do ps -o pid,stat,etime -p $p | tail -1; done
pgrep -f selfplay_nn | while read p; do
  S=$(awk '/^State/{print $2}' /proc/$p/status 2>/dev/null)
  [ "$S" != "Z" ] && ps -o pid,stat,etime -p $p | tail -1
done
pgrep -f train_round | while read p; do ps -o pid,stat,etime -p $p | tail -1; done
echo "== loop.out tail =="
tail -3 /workspace/go-run-9x9/loop.out
