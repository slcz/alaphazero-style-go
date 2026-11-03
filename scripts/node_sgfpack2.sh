#!/bin/bash
set -e
N=$(ls /workspace/logs/gate-rand/sgf | wc -l)
echo "sgf count: $N"
tar czf /workspace/logs/gate-rand-sgf.tgz -C /workspace/logs/gate-rand sgf
ls -la /workspace/logs/gate-rand-sgf.tgz
