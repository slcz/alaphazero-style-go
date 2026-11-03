#!/bin/bash
echo "== eval.log tail =="
tail -2 /workspace/go-run-9x9/it002/eval.log 2>/dev/null || tail -2 /workspace/go-run-9x9/it001/eval.log 2>/dev/null
echo "== gates running (mux procs) =="
for p in $(pgrep -f engine_mux); do
  S=$(awk '/^State/{print $2}' /proc/$p/status 2>/dev/null)
  [ "$S" = "Z" ] && continue
  echo "pid $p state=$S cmd=$(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-110)"
done
echo "== gpu =="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "== NN_B env of live mux =="
P=$(pgrep -f engine_mux | head -1)
tr '\0' '\n' < /proc/$P/environ 2>/dev/null | grep -E "NN_B|NN_WAVE|PUCT_SAMPLE|NN_COMPILE" | sort
