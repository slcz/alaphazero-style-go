#!/bin/bash
# node_restore1.sh — fresh instance: build engine, install torch.
set -e
cd /workspace/go
echo "== build =="
make -C engine all nn 2>&1 | grep -cE "\berror\b" && exit 1
test -x engine/build/selfplay_nn && echo "build ok"
echo "== torch =="
python3 -c "import torch; print(torch.__version__)" 2>/dev/null || {
    pip3 install --quiet torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -1
    python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
