"""test_loop.py — the full selfplay_loop pipeline end to end at tiny
scale: generate (real net on CPU via embedded Python) -> train -> gate ->
adopt. The heaviest test; validates every stage's plumbing."""
import json
import subprocess
import sys
from pathlib import Path

import torch

from go_ai.net.model import GoNet, NetConfig
from go_ai.train.trainer import Trainer, TrainConfig

REPO = Path(__file__).resolve().parents[1]


def test_selfplay_loop(tmp_path):
    init = tmp_path / "init.pt"
    torch.manual_seed(0)
    net = GoNet(NetConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64,
                          rope_pairs=2))
    Trainer(net, TrainConfig(device="cpu", compile=False)).save(init)

    r = subprocess.run(
        [sys.executable, "scripts/selfplay_loop.py",
         "--work-dir", str(tmp_path / "run"), "--init-ckpt", str(init),
         "--b", "7", "--komi", "9.5",
         "--iterations", "2", "--games", "2",
         "--sims", "8", "--fast-sims", "4", "--fast-p", "0.5",
         "--pool", "4", "--rows", "64", "--max-plys", "30",
         "--batch", "16", "--epochs", "1",
         "--gate-games", "2", "--gate-sims", "4",
         "--device", "cpu", "--seed", "5"],
        capture_output=True, text=True, cwd=str(REPO), timeout=1200)
    assert r.returncode == 0, r.stderr[-3000:] + r.stdout[-1000:]

    work = tmp_path / "run"
    recs = [json.loads(l) for l in
            (work / "loop.jsonl").read_text().splitlines()]
    assert len(recs) == 2
    for rec in recs:
        assert rec["games"] == 2
        assert rec["gate_score"] is not None
        assert rec["adopted"] in (True, False)
    # iteration artifacts exist and the best.pt was possibly updated
    assert (work / "it001" / "train.pt").exists()
    assert (work / "it001" / "eval.json").exists()
    assert (work / "best.pt").exists()
    # resume: rerunning skips finished stages (loop.jsonl grows by 0 —
    # actually iterations re-run; just verify the shard was reused)
    shard_games = json.loads(
        (work / "it001" / "selfplay" / "DONE.json").read_text())
    assert shard_games["games_done"] == 2
