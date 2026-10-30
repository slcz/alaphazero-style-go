"""test_train_round.py — the round trainer end to end on the dummy
selfplay fixture: tiny net, CPU, 8× D4 augmentation in the batch."""
import subprocess
import sys
from pathlib import Path

import torch

from go_ai.net.model import GoNet, NetConfig
from go_ai.train.trainer import Trainer, TrainConfig

REPO = Path(__file__).resolve().parents[1]


def test_train_round(tmp_path, sp_dir):
    ck = tmp_path / "base.pt"
    torch.manual_seed(0)
    net = GoNet(NetConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64,
                          rope_pairs=2))
    Trainer(net, TrainConfig(device="cpu", compile=False)).save(ck)
    out = tmp_path / "train.pt"
    r = subprocess.run(
        [sys.executable, "scripts/train_round.py",
         "--data-dir", str(sp_dir), "--base-ckpt", str(ck),
         "--out", str(out), "--epochs", "1", "--batch", "16",
         "--device", "cpu", "--seed", "3", "--log-every", "5",
         "--json", str(tmp_path / "train.json")],
        capture_output=True, text=True, cwd=str(REPO), timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    meta = __import__("json").loads((tmp_path / "train.json").read_text())
    assert meta["games"] == 4
    assert meta["steps"] >= 8                     # 8× aug, ≥1 step/ply-8
    assert out.exists()
    tr = Trainer.load(out, device="cpu")          # loads + resumes
    assert tr.steps == meta["steps"]
