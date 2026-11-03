#!/bin/bash
# node_pip.sh — finish the torch upgrade, bounded.
pip3 install --upgrade torch --index-url https://download.pytorch.org/whl/cu128 \
    --progress-bar off 2>&1 | tail -2
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
