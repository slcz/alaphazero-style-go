"""test_shard_format.py — the GOSHARD1 reader against a real shard from
the C driver (dummy evaluator): field sanity, replayed π̄ order contract,
extent boundaries, DONE gating."""
import json

import pytest

from go_ai.engine import Board
from go_ai.selfplay.shard import (MOVE_PASS, SP_END_ADJUDICATED,
                                  SP_END_NATURAL, game_extents, read_shard,
                                  read_unit, scan_shard)


def test_read_shard(sp_dir):
    games = read_shard(sp_dir / "shard-0.spbin")
    assert len(games) == 4
    for g in games:
        assert g.b == 7 and g.komi == 9.5 and g.n_sims == 8
        assert len(g.ownership) == 49
        assert g.n_plies == len(g.plies) and g.n_plies > 0
        assert g.score != 0.0                     # fractional komi: no jigo
        for o in g.ownership:
            assert o in (0.0, 0.5, 1.0)
        # replay: the order contract + bookkeeping (maxply = n_plies so
        # the cap game's wall sits exactly at its last ply)
        b = Board(g.b, g.komi, g.n_plies)
        for p in g.plies:
            legal = b.legal_moves
            assert len(p.pi) == len(legal)
            assert p.mover == b.side_to_move
            assert p.move in legal
            assert p.sims in (8, 4)               # slow or fast budget
            b.push(p.move)
        code, score, own = b.termination()
        assert code == 1
        assert score == pytest.approx(g.score, abs=1e-4)


def test_end_kinds_and_extents(sp_dir):
    path = sp_dir / "shard-0.spbin"
    games = read_shard(path)
    for g in games:
        last2 = [p.move for p in g.plies[-2:]]
        if g.end_kind == SP_END_NATURAL:
            assert last2 == [MOVE_PASS, MOVE_PASS]
        else:
            assert g.end_kind == SP_END_ADJUDICATED
            assert g.n_plies == 40
    extents = list(game_extents(path))
    assert len(extents) == len(games)
    parsed = [(s, e) for _g, s, e in scan_shard(path)]
    assert extents == parsed


def test_read_unit(sp_dir):
    games = read_unit(sp_dir)
    assert len(games) == 4
    done = json.loads((sp_dir / "DONE.json").read_text())
    assert done["games_done"] == 4 and done["n_workers"] == 1
