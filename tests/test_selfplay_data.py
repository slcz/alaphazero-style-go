"""test_selfplay_data.py — samples from a real shard: PCR filter, masks,
π̄ renorm, WDL/ownership/score labels, aux reply."""
import numpy as np
import pytest

from go_ai.engine import BLACK
from go_ai.net import features as F
from go_ai.net.model import score_to_bin
from go_ai.selfplay.shard import read_unit
from go_ai.train.selfplay_data import selfplay_samples, stack_selfplay


def _all_samples(sp_dir):
    out = []
    for g in read_unit(sp_dir):
        out.extend(selfplay_samples(g))
    return out


def test_samples(sp_dir):
    games = read_unit(sp_dir)
    samples = _all_samples(sp_dir)
    assert len(samples) > 20                  # 4 games × slow-cap plies

    total_plies = sum(g.n_plies for g in games)
    slow_plies = sum(1 for g in games for p in g.plies if p.sims == g.n_sims)
    assert len(samples) == slow_plies
    assert slow_plies < total_plies            # PCR actually filtered

    b = 7
    for s in samples:
        assert s["tokens"].shape == (49, 18)
        assert not s["mask"][b * b + 1:].any()          # tail inactive
        legal_slots = np.unique(s["pi_slots"][s["pi_probs"] > 0])
        assert s["mask"].sum() == len(legal_slots)
        assert s["pi_probs"].sum() == pytest.approx(1.0, abs=1e-5)
        assert (s["pi_probs"] >= 0).all()
        assert s["wdl"].sum() == 1.0 and s["wdl"].max() == 1.0
        assert set(np.unique(s["own"])) <= {0.0, 0.5, 1.0}
        assert 0 <= s["score_bin"] < 761
        assert s["aux_w"] in (0.0, 1.0)

    # labels consistent with the recorded game score per mover
    g = games[0]
    ss = selfplay_samples(g)
    mover0 = g.plies[0].mover
    win0 = (mover0 == BLACK) == (g.score > 0)
    assert ss[0]["wdl"][0 if win0 else 1] == 1.0
    exp_bin = score_to_bin(g.score if mover0 == BLACK else -g.score)
    assert ss[0]["score_bin"] == exp_bin
    # aux: the reply actually played (every trainable ply but the last)
    trainable = [i for i, p in enumerate(g.plies) if p.sims == g.n_sims]
    for j, i in enumerate(trainable[:-1]):
        assert ss[j]["aux_idx"] == F.move_to_slot(g.plies[i + 1].move, g.b)


def test_stack(sp_dir):
    ss = _all_samples(sp_dir)
    stacked = stack_selfplay(ss[:8])
    assert stacked["tokens"].shape == (8, 49, 18)
    assert stacked["mask"].shape == (8, F.POLICY_MAX)
    assert stacked["pi_slots"].shape == (8, F.PI_SUPPORT_MAX)
    assert stacked["own"].shape == (8, 49)
    assert stacked["wdl"].shape == (8, 2)
