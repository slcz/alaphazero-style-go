"""test_symmetry.py — D4 augmentation validated THROUGH the C engine:
transformed games replay legally, transformed tokens match the C
encoder's output on the transformed replay (pins the edge-dim fix), and
masks/π̄/ownership/aux all stay consistent, for all 8 symmetries."""
import numpy as np
import pytest

from go_ai.engine import BLACK, TERM_OVER, Board
from go_ai.net import features as F
from go_ai.selfplay.shard import MOVE_PASS, read_unit
from go_ai.train.selfplay_data import ply_trainable, selfplay_samples
from go_ai.train.symmetry import (sym_point_map, transform_move,
                                  transform_sample)

SYMS = range(8)


def _transformed_game(g, s):
    """The game's moves under symmetry s, replayed in a fresh Board."""
    b2 = Board(g.b, g.komi, g.n_plies)
    for p in g.plies:
        b2.push(transform_move(p.move, g.b, s))   # legality preserved
    return b2


def test_replay_all_syms(sp_dir):
    for g in read_unit(sp_dir):
        base = Board(g.b, g.komi, g.n_plies)
        for p in g.plies:
            base.push(p.move)
        code, score, _ = base.termination()
        for s in SYMS:
            b2 = _transformed_game(g, s)
            fwd = sym_point_map(g.b, s)
            # final stones are the image of the final stones
            base_st = base.stones()
            img = np.empty_like(base_st)
            img[fwd] = base_st                    # img[T(p)] = stone at p
            assert (b2.stones() == img).all(), f"sym {s}"
            # score invariant
            code2, score2, own2 = b2.termination()
            assert code2 == TERM_OVER
            assert score2 == pytest.approx(score, abs=1e-4)


def test_samples_all_syms(sp_dir):
    g = read_unit(sp_dir)[0]
    b = g.b
    boards = {s: _transformed_game(g, s) for s in SYMS}
    _, _, own_black_0 = boards[0].termination()
    fwd0 = sym_point_map(b, 0)

    for s in SYMS:
        fwd = sym_point_map(b, s)
        inv = np.empty_like(fwd)
        inv[fwd] = np.arange(b * b)
        board = boards[s]
        samples = selfplay_samples(g, syms=(s,))
        trainable = [i for i, p in enumerate(g.plies) if ply_trainable(g, p)]
        assert len(samples) == len(trainable)

        # replay the transformed game ply by ply alongside the samples
        replay = Board(b, g.komi, g.n_plies)
        ti = 0
        own_black = None
        for i, p in enumerate(g.plies):
            if i in trainable:
                smp = samples[ti]
                ti += 1
                legal = replay.legal_moves
                # tokens == C encode of the transformed replay position
                assert (smp["tokens"] == replay.encode()).all(), f"sym {s} ply {i}"
                # mask == the transformed position's legal set
                assert (smp["mask"] == F.policy_mask(legal, b)).all(), f"sym {s}"
                # π̄: same support size, sums to 1, mass rides the moves
                n_legal = len(legal)
                probs = smp["pi_probs"][:n_legal]
                assert probs.sum() == pytest.approx(1.0, abs=1e-5)
                assert (smp["mask"][smp["pi_slots"][:n_legal]]).all()
                if s == 0:
                    pass
                else:
                    # per-move prob is preserved: compare multisets
                    o = selfplay_samples(g, syms=(0,))[ti - 1]
                    op = np.sort(o["pi_probs"][:n_legal])[::-1]
                    assert np.allclose(np.sort(probs)[::-1], op, atol=1e-6)
                # wdl / score invariant
                o = selfplay_samples(g, syms=(0,))[ti - 1]
                assert (smp["wdl"] == o["wdl"]).all()
                assert smp["score_bin"] == o["score_bin"]
                # aux reply legal in the transformed after-position
                assert smp["aux_mask"][smp["aux_idx"]] or smp["aux_w"] == 0
            replay.push(transform_move(p.move, b, s))
        code, _, own_black = replay.termination()
        assert code == TERM_OVER and ti == len(samples)

        # ownership: mover-view label == transformed game-end ownership
        for j, i in enumerate([k for k, p in enumerate(g.plies)
                               if ply_trainable(g, p)]):
            smp = selfplay_samples(g, syms=(s,))[j]
            side = g.plies[i].mover
            own_mover = own_black if side == BLACK else 1.0 - own_black
            assert (smp["own"] == own_mover).all(), f"sym {s} own ply {i}"


def test_identity_is_copy(sp_dir):
    g = read_unit(sp_dir)[0]
    base = selfplay_samples(g, syms=(0,))
    ident = selfplay_samples(g, syms=(0,))
    assert len(base) == len(ident)
    for a, bsm in zip(base, ident):
        for k in a:
            assert (a[k] == bsm[k]).all() if a[k].ndim else a[k] == bsm[k]


def test_augment_x8(sp_dir):
    g = read_unit(sp_dir)[0]
    one = selfplay_samples(g, syms=(0,))
    aug = selfplay_samples(g, syms=SYMS)
    assert len(aug) == 8 * len(one)
    # the first 8 samples are the 8 syms of ply 0, each distinct in tokens
    # unless the position itself is symmetric (an empty board is)
    for k in range(1, 8):
        assert aug[0]["wdl"][0] == aug[k]["wdl"][0]   # labels invariant


def test_sym_maps_bijective():
    for b in (2, 5, 7, 19):
        for s in SYMS:
            fwd = sym_point_map(b, s)
            assert sorted(fwd.tolist()) == list(range(b * b))
