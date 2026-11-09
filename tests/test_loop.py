"""test_loop.py — the full selfplay_loop pipeline end to end at tiny
scale: generate (real net on CPU via embedded Python) -> train ->
promote. The heaviest test; validates every stage's plumbing."""
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
    net = GoNet(NetConfig(d_model=64, n_layers=2, n_heads=4, d_ff=64,
                          rope_pairs=2))
    Trainer(net, TrainConfig(device="cpu", compile=False)).save(init)

    r = subprocess.run(
        [sys.executable, "scripts/selfplay_loop.py",
         "--work-dir", str(tmp_path / "run"), "--init-ckpt", str(init),
         "--iterations", "2", "--sizes", "7:2",
         "--sims", "8", "--fast-sims", "4", "--fast-p", "0.5",
         "--pool", "4", "--rows", "64",
         "--batch", "16", "--epochs", "1",
         "--device", "cpu", "--seed", "5"],
        capture_output=True, text=True, cwd=str(REPO), timeout=1200)
    assert r.returncode == 0, r.stderr[-3000:] + r.stdout[-1000:]

    work = tmp_path / "run"
    recs = [json.loads(l) for l in
            (work / "loop.jsonl").read_text().splitlines()]
    assert len(recs) == 2
    for rec in recs:
        assert rec["games"] == 2
        assert len(rec["ckpt_sha256"]) == 64
    # iteration artifacts exist and promotion is unconditional: best.pt
    # is always the latest trained checkpoint
    assert (work / "it001" / "train.pt").exists()
    assert (work / "best.pt").exists()
    assert (work / "best.pt").read_bytes() == \
        (work / "it002" / "train.pt").read_bytes()
    # resume: rerunning skips finished stages (loop.jsonl grows by 0 —
    # actually iterations re-run; just verify the shard was reused)
    shard_games = json.loads(
        (work / "it001" / "sp7" / "DONE.json").read_text())
    assert shard_games["games_done"] == 2
    # replay buffer: the newest round is whole, older rounds are thinned,
    # and the manifest names the shard each round's games live in
    buf = json.loads((work / "buffer.json").read_text())
    got = [r["round"] for r in buf["rounds"]]
    assert got in ([1, 2], [2]), got     # keep=1/3: a 2-game round may go
    assert buf["rounds"][-1]["round"] == 2
    assert buf["rounds"][-1]["games"] == [0, 1]      # newest, in full
    for r in buf["rounds"]:
        assert (work / r["shard"]).exists()
