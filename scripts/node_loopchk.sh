#!/bin/bash
echo "loop.out:"; cat /workspace/go-run-9x9/loop.out 2>/dev/null | tail -8
echo "pid 5853:"; ps -o pid,stat,etime,cmd -p 5853 2>/dev/null | tail -1
echo "python procs:"; pgrep -f selfplay_loop | while read p; do ps -o pid,stat,cmd -p $p | tail -1; done
echo "gpu:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
