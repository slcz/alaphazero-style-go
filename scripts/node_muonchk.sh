#!/bin/bash
python3 -c "import torch; print('torch', torch.__version__); print('Muon:', hasattr(torch.optim, 'Muon'))"
echo "== train.log tail =="
tail -4 /workspace/go-run-9x9/it001/train.log 2>/dev/null
echo "== loop.out tail =="
tail -4 /workspace/go-run-9x9/loop.out 2>/dev/null
echo "== gpu =="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
