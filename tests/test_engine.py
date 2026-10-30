"""test_engine.py — the cffi Board over the C TT rules + encoder."""
import numpy as np
import pytest

from go_ai.engine import (BLACK, EMPTY, TERM_OVER, TERM_PLAYING, WHITE,
                          Board, IllegalMove, PASS)

GNN_OWN, GNN_OPP = 0, 1
GNN_EDGE, GNN_KOMI, GNN_SIZE = 7, 16, 17


def test_empty_board():
    b = Board(7, komi=9.5, maxply=150)
    assert b.legal_moves == list(range(49)) + [PASS]
    assert b.side_to_move == BLACK and b.ply == 0
    assert (b.stones() == EMPTY).all()
    code, score, own = b.termination()
    assert code == TERM_PLAYING and score is None and own is None


def test_push_capture_and_illegal():
    b = Board(5, komi=5.5, maxply=100)
    for mv in (1, 0, 24, 2, 19, 6):      # ends with W b2 capturing B b1
        b.push(mv)
    st = b.stones()
    assert st[1] == EMPTY
    assert st[0] == WHITE and st[2] == WHITE and st[6] == WHITE
    with pytest.raises(IllegalMove):
        b.push(0)                          # occupied
    with pytest.raises(IllegalMove):
        b.push(6)                          # would be single-stone suicide


def test_double_pass_scoring():
    b = Board(5, komi=5.5, maxply=100)
    b.push(PASS)
    b.push(PASS)
    code, score, own = b.termination()
    assert code == TERM_OVER
    assert score == pytest.approx(-5.5)      # empty board: white by komi
    assert (own == 0.5).all()

    b = Board(5, komi=0.5, maxply=100)
    for x in range(5):
        b.push(x)            # black row 0
        b.push(x + 10)       # white row 2
    b.push(PASS)
    b.push(PASS)
    code, score, own = b.termination()
    assert code == TERM_OVER
    # B: 5 stones, row 1 dame; W: 5 stones + rows 3-4 territory (10)
    assert score == pytest.approx((5.0 - 15.0) - 0.5)
    assert own[0] == 1.0 and own[5] == 0.5 and own[10] == 0.0


def test_encode_spec():
    b = Board(7, komi=9.5, maxply=150)
    t = b.encode()
    assert t.shape == (49, 18) and t.dtype == np.float32
    a1 = t[0]
    assert a1[GNN_OWN] == 0 and a1[GNN_OPP] == 0
    assert a1[GNN_EDGE] == 0 and a1[GNN_EDGE + 2] == 0
    assert a1[GNN_EDGE + 1] == pytest.approx(6 / 7)
    assert a1[GNN_EDGE + 3] == pytest.approx(6 / 7)
    assert a1[GNN_KOMI] == pytest.approx(9.5 / 15)
    assert a1[GNN_SIZE] == pytest.approx(7 / 19)

    b.push(24)                       # black D4; white to move
    t = b.encode()
    assert t[24][GNN_OWN] == 0 and t[24][GNN_OPP] == 1
    b.push(PASS)                     # black to move again
    t = b.encode()
    assert t[24][GNN_OWN] == 1 and t[24][GNN_OPP] == 0
