"""The replay buffer manifest: only the preserved game ids are parsed and
trained on, and the trim is a pure function of (iteration, round, seed).

    GO_TEST_SHARD=... python3 -m pytest tests/test_buffer.py
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from go_ai.net.model import GoNet, NetConfig  # noqa: E402
from go_ai.selfplay.shard import read_games, read_shard  # noqa: E402
from go_ai.train.trainer import Trainer, TrainConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TINY = NetConfig(d_model=64, n_layers=2, n_heads=4, d_ff=64, rope_pairs=2)


def _module(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _fake_round(work: Path, r: int, n: int) -> None:
    d = work / f"it{r:03d}" / "selfplay"
    d.mkdir(parents=True, exist_ok=True)
    (d / "shard-0.spbin").write_bytes(b"")
    (d / "DONE.json").write_text(
        json.dumps({"n_workers": 1, "unit_seed": r, "games_done": n}))


def test_buffer_decay(tmp_path, monkeypatch):
    loop = _module("selfplay_loop")
    monkeypatch.setattr(loop, "push_buffer", lambda *a, **k: None)
    for r in (1, 2, 3):
        _fake_round(tmp_path, r, 1000)
    rec = loop.update_buffer(tmp_path, 3, 0.5, 1)
    sizes = [len(x["games"]) for x in rec["rounds"]]
    for got, want in zip(sizes, (250, 500, 1000)):
        assert abs(got - want) < 60, (sizes, "keep^age of 1000 games")
    assert rec["rounds"][-1]["games"] == list(range(1000))   # newest in full
    # deterministic: a resume rebuilds the identical manifest
    assert loop.update_buffer(tmp_path, 3, 0.5, 1) == rec
    # the trim only ever takes games away (same per-round RNG stream)
    nxt = loop.update_buffer(tmp_path, 4, 0.5, 1)
    assert set(nxt["rounds"][0]["games"]) <= set(rec["rounds"][0]["games"])


def test_buffer_drops_expired_rounds(tmp_path, monkeypatch):
    loop = _module("selfplay_loop")
    monkeypatch.setattr(loop, "push_buffer", lambda *a, **k: None)
    for r in (1, 2):
        _fake_round(tmp_path, r, 100)
    # keep 0.5 eleven rounds on: ~0.05 of 100 games survives, i.e. none
    assert loop.update_buffer(tmp_path, 12, 0.5, 1)["rounds"] == []


def test_read_games_subset(sp_dir):
    shard = sp_dir / "shard-0.spbin"
    all_games = {g.game_index: g for g in read_shard(shard)}
    keep = sorted(all_games)[:2]
    got = read_games(shard, keep)
    assert [g.game_index for g in got] == keep
    for g in got:
        ref = all_games[g.game_index]
        assert g.score == ref.score and g.n_plies == ref.n_plies
        assert [p.move for p in g.plies] == [p.move for p in ref.plies]
        assert [p.sims for p in g.plies] == [p.sims for p in ref.plies]


def test_train_round_buffer(tmp_path, sp_dir):
    shard = sp_dir / "shard-0.spbin"
    all_games = read_shard(shard)
    keep = [all_games[0].game_index, all_games[-1].game_index]
    man = tmp_path / "buffer.json"
    man.write_text(json.dumps({"version": 1, "rounds": [
        {"round": 1, "shard": str(shard), "count": len(all_games),
         "games": keep}]}))

    ck = tmp_path / "base.pt"
    Trainer(GoNet(TINY), TrainConfig(device="cpu", compile=False)).save(ck)
    out = tmp_path / "train.pt"
    r = subprocess.run(
        [sys.executable, "scripts/train_round.py", "--buffer", str(man),
         "--base-ckpt", str(ck), "--out", str(out), "--epochs", "1",
         "--batch", "16", "--device", "cpu", "--seed", "3",
         "--log-every", "5", "--json", str(tmp_path / "train.json")],
        capture_output=True, text=True, cwd=str(REPO), timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    meta = json.loads((tmp_path / "train.json").read_text())
    assert meta["games"] == len(keep)
    assert meta["steps"] >= 1
    assert out.exists()


def test_buffer_survives_a_missing_shard(tmp_path, sp_dir):
    """A restored node may lack an old round: the manifest drops it
    instead of failing the run."""
    man = tmp_path / "buffer.json"
    man.write_text(json.dumps({"version": 1, "rounds": [
        {"round": 1, "shard": "gone/shard-0.spbin", "games": [0, 1]},
        {"round": 2, "shard": str(sp_dir / "shard-0.spbin"), "games": [0]}]}))
    out = _module("train_round").load_buffer(man)
    assert [r[0] for r in out] == [2]
