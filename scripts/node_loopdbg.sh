#!/bin/bash
cd /workspace/go
echo "=== py compile check ==="
python3 -m py_compile go_ai/net/nn_worker.py scripts/selfplay_loop.py && echo compile-ok
echo "=== 12s foreground loop run ==="
export PUCT_WALKERS=4 PUCT_TASKS=8 NN_B=9 NN_BF16=1 NN_COMPILE=1
timeout 12 python3 scripts/selfplay_loop.py --work-dir /workspace/go-run-9x9 \
    --b 9 --komi 7.5 --iterations 1 --games 30000 \
    --sims 200 --fast-sims 40 --fast-p 0.75 --pool 2048 --rows 4096 \
    --sample-plys 10 --max-plys 250 --epochs 1 --batch 4096 --lr 2e-4 \
    --gate-games 400 --gate-sims 200 --device cuda --seed 1 2>&1 | head -12
echo "rc=$?"
