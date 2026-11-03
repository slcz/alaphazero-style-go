#!/bin/bash
# node_pip.sh — run ON the node: install the runtime deps, then leave a
# marker. Started with nohup so a dropped ssh session cannot kill it.
cd /workspace/go || exit 1
set -x
python3 -m pip install --break-system-packages --quiet --timeout 60 \
    --retries 5 --index-url https://download.pytorch.org/whl/cu128 torch
python3 -m pip install --break-system-packages --quiet --timeout 60 \
    --retries 5 huggingface_hub pytest sgfmill
python3 -c "import torch; print('torch', torch.__version__,
  'cuda', torch.cuda.is_available(),
  'muon', hasattr(torch.optim, 'Muon'))" > /tmp/pip.result 2>&1
echo done >> /tmp/pip.result
touch /tmp/pip.done
