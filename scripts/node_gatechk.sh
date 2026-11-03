#!/bin/bash
echo "== eval.log: games done =="
grep -ac "^GAME" /workspace/go-run-9x9/it001/eval.log 2>/dev/null
tail -3 /workspace/go-run-9x9/it001/eval.log
echo "== procs (engine_mux count) =="
pgrep -f engine_mux | while read p; do ps -o pid,etime,%cpu -p $p | tail -1; done
echo "== gpu =="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "== gate env (PID env of first mux) =="
P=$(pgrep -f engine_mux | head -1)
tr '\0' '\n' < /proc/$P/environ 2>/dev/null | grep -E "NN_B|NN_COMPILE|NN_BF16|PUCT_SAMPLE" | sort
