"""test_sgf.py — SGF parsing filters, coordinate mapping, and the
supervised sample builder (parity through the C engine)."""
import numpy as np
import pytest

from go_ai.data.sgf import parse_sgf_bytes
from go_ai.engine import PASS, BLACK, Board
from go_ai.net import features as F
from go_ai.train.supervised_data import supervised_sample

HDR = "(;GM[1]FF[4]SZ[19]KM[6.5]RE[{res}]PB[w]PW[b]{extra}{moves})\n"

# 12 distant legal points, alternating play (no captures)
BASE = ["pd", "dd", "pp", "dp", "qe", "dc", "cc", "ee", "gg", "jj",
        "kj", "jk"]


def sgf(res="B+2.5", extra="", moves=""):
    return HDR.format(res=res, extra=extra, moves=moves).encode()


def seq_sgf(res="B+2.5", extra="", seq=BASE):
    moves = "".join(f";{'B' if i % 2 == 0 else 'W'}[{m}]"
                    for i, m in enumerate(seq))
    return sgf(res, extra, moves)


def test_coordinate_map_and_replay():
    # B[pd]: col p = 15, SGF row d = 3 from the top -> row from the
    # bottom 15 (sgfmill gives row 0 at the bottom, same as ours)
    g = parse_sgf_bytes(seq_sgf())
    assert g is not None
    assert g.moves[0] == 15 * 19 + 15
    assert g.moves[1] == 15 * 19 + 3
    bd = Board(19, komi=6.5, maxply=1024)
    for mv in g.moves:
        bd.push(mv)                      # legal all the way
    assert g.black_wins and g.komi == 6.5


def test_pass():
    seq = BASE[:4] + []                  # insert a pass after 2 moves
    moves = ";B[pd];W[dd];B[];W[pp]" + "".join(
        f";{'B' if i % 2 == 0 else 'W'}[{m}]"
        for i, m in enumerate(BASE[4:10]))
    g = parse_sgf_bytes(sgf(moves=moves))
    assert g is not None
    assert g.moves[2] == PASS
    assert len(g.moves) == 10


def test_filters():
    assert parse_sgf_bytes(seq_sgf(extra="AB[pd][dp]")) is None
    assert parse_sgf_bytes(seq_sgf(res="0")) is None
    assert parse_sgf_bytes(seq_sgf(res="B+R")) is not None
    assert parse_sgf_bytes(seq_sgf(res="W+T")) is not None
    # a move after double pass: TT-terminal, dropped
    assert parse_sgf_bytes(sgf(moves=";B[];W[];B[dd]")) is None
    assert parse_sgf_bytes(seq_sgf()) is not None


def test_supervised_sample_parity():
    g = parse_sgf_bytes(seq_sgf())
    s = supervised_sample(g, 1)          # position before W[dd]
    bd = Board(19, komi=6.5, maxply=1024)
    bd.push(g.moves[0])
    assert (s["tokens"] == bd.encode()).all()
    assert (s["mask"] == F.policy_mask(bd.legal_moves, 19)).all()
    # one-hot policy target on the played move
    assert s["pi_probs"].sum() == 1.0
    hit = s["pi_slots"][s["pi_probs"] > 0]
    assert hit[0] == F.move_to_slot(g.moves[1], 19)
    assert s["mask"][hit[0]]
    # value: white to move, white won
    assert s["wdl"][1] == 1.0 and s["wdl"][0] == 0.0
    # aux = black's actual reply, legal in the after-position
    bd.push(g.moves[1])
    assert s["aux_idx"] == F.move_to_slot(g.moves[2], 19)
    assert s["aux_mask"][s["aux_idx"]]


def test_supervised_sample_symmetry():
    g = parse_sgf_bytes(seq_sgf())
    s0 = supervised_sample(g, 5, sym=0)      # mid-game: stones on board
    s1 = supervised_sample(g, 5, sym=1)
    assert (s0["wdl"] == s1["wdl"]).all()
    assert s0["pi_probs"].sum() == s1["pi_probs"].sum() == 1.0
    assert s1["mask"].sum() == s0["mask"].sum()
    assert not (s0["tokens"] == s1["tokens"]).all()   # actually transformed


@pytest.mark.skipif(not __import__("pathlib").Path("data/kgs/kgs-19-2012")
                    .exists(), reason="KGS data not downloaded")
def test_real_data_smoke():
    from pathlib import Path
    from go_ai.data.sgf import load_dir
    games = load_dir(sorted(Path("data/kgs").glob("kgs-19-*"))[0], b=19)
    assert len(games) > 100
    total = sum(len(g.moves) for g in games[:50])
    assert total > 3000                 # real full-length games
    s = supervised_sample(games[0], 5)
    assert s["tokens"].shape == (361, 18)
