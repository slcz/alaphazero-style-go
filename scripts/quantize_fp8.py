#!/usr/bin/env python3
"""quantize_fp8.py — convert a checkpoint to FP8 (E4M3 weights, per-
tensor scale) for inference.

    python scripts/quantize_fp8.py sup/base.pt sup/base_fp8.pt

Every Linear and 1x1-conv weight is quantized (activations stay dynamic
per call); the fp8 flag lands in net_config so the loader rebuilds the
quantized graph. Accuracy is checked by eval_valloss against the source
checkpoint."""
import sys
from dataclasses import asdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from go_ai.net.model import to_fp8_
from go_ai.train.trainer import Trainer

src, dst = sys.argv[1], sys.argv[2]
tr = Trainer.load(src, device="cpu")
to_fp8_(tr._raw_model)
net_cfg = asdict(tr._raw_model.cfg)
net_cfg["fp8"] = True
torch.save({"model": tr._raw_model.state_dict(),
            "optimizer": None,
            "steps": tr.steps,
            "config": asdict(tr.cfg),
            "net_config": net_cfg}, dst)
print(f"fp8: {src} -> {dst} "
      f"({Path(dst).stat().st_size / 1e6:.1f} MB, "
      f"source {Path(src).stat().st_size / 1e6:.1f} MB)")
