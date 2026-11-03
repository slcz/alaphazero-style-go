#!/bin/bash
# node_torch211.sh — force torch 2.11 on the right interpreter, relaunch it2.
set -e
tmux kill-session -t go9 2>/dev/null || true
pkill -9 -f selfplay_loop.py 2>/dev/null || true
pkill -9 -f train_round.py 2>/dev/null || true
pkill -9 -f selfplay_nn 2>/dev/null || true
pkill -9 -f engine_mux 2>/dev/null || true
sleep 2
python3 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128 --progress-bar off 2>&1 | tail -2
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
bash /workspace/go/scripts/node_run9x9_tmux.sh 2
