#!/bin/bash
# node_sgf.sh — sample SGFs from the it001 selfplay shard.
cd /workspace/go
mkdir -p /workspace/logs/sgf && rm -f /workspace/logs/sgf/*.sgf
python3 scripts/shard_to_sgf.py --unit /workspace/go-run-9x9/it001/selfplay \
    --n 12 --random --seed 7 --out /workspace/logs/sgf
ls -la /workspace/logs/sgf | tail -4
tar czf /workspace/logs/it001-sgf.tgz -C /workspace/logs/sgf .
echo done
