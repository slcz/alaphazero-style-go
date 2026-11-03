"""
go_ai/train/selfplay_data.py — GOSHARD1 game records → training samples
(the trainer side of the selfplay driver).

One forward replay per game through the C engine (Board), so tokens come
from the C encoder itself — no Python encoder, and the shard's π̄ order
contract (n_pi == legal count, gen_legal order) is asserted ply by ply.

Sample schema (numpy; b = game.b, T = b*b):
    tokens    [T, 18]  f32   C-encoded features of the position
    mask      [362]    bool  legal slots (b²+1 active)
    pi_slots  [K]      i64   the ply's legal moves as slots, gen_legal
                            order, zero-padded by REPEATING THE FIRST slot
    pi_probs  [K]      f32   π̄ renormalized to sum 1 (f16-rounded on
                            disk), 0 on padding
    wdl       [2]      f32   mover-view [W, L] one-hot from the game score
    own       [T]      f32   mover-view ownership {0, .5, 1}
    score     [1]      f32   mover-view margin in points
    aux_idx   [1]      i64   the opponent's actual reply, in OUR frame
    aux_mask  [362]    bool  legal slots of the position after our move
    aux_w     [1]      f32   0 when the game ended before a reply

Playout cap randomization: only plies searched at the game's slow cap
(ply.sims == game.n_sims) become samples. Fast-cap and policy-only plies
live in the shard for replay but are never trained (the agreed PCR
semantics).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from go_ai.engine import BLACK, TERM_OVER, Board
from go_ai.net import features as F
from go_ai.selfplay.shard import MOVE_PASS, SP_END_PAUSED, SpGame
from go_ai.train.symmetry import transform_sample

K = F.PI_SUPPORT_MAX


def ply_trainable(game: SpGame, ply) -> bool:
    """PCR filter: a ply trains iff it was searched at the slow cap."""
    return ply.sims == game.n_sims


def trainable_plies(game: SpGame) -> List[int]:
    return [i for i, p in enumerate(game.plies) if ply_trainable(game, p)]


def selfplay_sample(game: SpGame, i: int, sym: int = 0) -> Dict[str, np.ndarray]:
    """One training sample: the position before `game.plies[i]`, its dense
    π̄, and the outcome labels, under symmetry `sym`. Single-ply form —
    what the training loop's on-the-fly batch sampler calls (a round's
    samples must not be materialized up front). Replays through the C
    engine with trusted pushes (shard moves were legal when recorded)."""
    b = game.b
    ply = game.plies[i]
    assert ply_trainable(game, ply)
    board = Board(b, game.komi, game.n_plies)
    for k in range(i):
        board.push_fast(game.plies[k].move)
    side = board.side_to_move
    legal = board.legal_moves
    assert len(ply.pi) == len(legal), \
        f"n_pi {len(ply.pi)} != {len(legal)} legal — π order contract broken"
    tokens = board.encode()

    slots = [F.move_to_slot(m, b) for m in legal]
    probs = np.asarray(ply.pi, dtype=np.float64)
    probs /= probs.sum()
    pi_slots = np.full(K, slots[0], dtype=np.int64)
    pi_slots[: len(slots)] = slots
    pi_probs = np.zeros(K, dtype=np.float32)
    pi_probs[: len(slots)] = probs

    if i + 1 < len(game.plies):
        after = Board(b, game.komi, game.n_plies)
        for k in range(i + 1):
            after.push_fast(game.plies[k].move)
        aux_idx = F.move_to_slot(game.plies[i + 1].move, b)
        aux_mask = F.policy_mask(after.legal_moves, b)
        aux_w = 1.0
        assert aux_mask[aux_idx], "reply not in its legal set"
    else:
        aux_idx, aux_mask, aux_w = 0, F.policy_mask(legal, b), 0.0

    score_mover = game.score if side == BLACK else -game.score
    own_mover = (np.asarray(game.ownership, dtype=np.float32)
                 if side == BLACK else 1.0 - np.asarray(game.ownership,
                                                        dtype=np.float32))
    base = {
        "tokens": tokens,
        "mask": F.policy_mask(legal, b),
        "pi_slots": pi_slots,
        "pi_probs": pi_probs,
        "wdl": F.wdl_onehot(score_mover > 0),
        "own": own_mover,
        "score": np.float32(score_mover),
        "aux_idx": np.int64(aux_idx),
        "aux_mask": aux_mask,
        "aux_w": np.float32(aux_w),
    }
    return transform_sample(base, b, sym)


def selfplay_samples(game: SpGame, syms=(0,)) -> List[Dict[str, np.ndarray]]:
    """Samples for one game. `syms` selects the D4 symmetries applied per
    trainable ply — the default (0,) is the identity; training uses
    range(8) (the board's 4 rotations × flip), 8× the data."""
    b = game.b
    # maxply = n_plies: identical legality during the replay (a cap game's
    # wall sits exactly at its last ply) and TERM_OVER after it
    board = Board(b, game.komi, game.n_plies)
    per_ply = []
    for ply in game.plies:
        side = board.side_to_move
        legal = board.legal_moves
        assert ply.mover == side, "record mover disagrees with replay"
        assert len(ply.pi) == len(legal), \
            f"n_pi {len(ply.pi)} != {len(legal)} legal — π order contract broken"
        per_ply.append({"side": side, "legal": legal,
                        "tokens": board.encode()})
        board.push(ply.move)

    code, score, own_black = board.termination()
    if game.end_kind == SP_END_PAUSED:
        # selfplay stopped early: the score/ownership labels come from the
        # KataGo pass over this position, not from playing it out
        score = float(game.score)
        own_black = np.asarray(game.ownership, dtype=np.float32)
    else:
        assert code == TERM_OVER
        assert abs(score - game.score) < 1e-4, \
            f"replayed score {score} != recorded {game.score}"
    assert score != 0.0, "jigo in selfplay (fractional komi required)"
    own_black = np.asarray(game.ownership, dtype=np.float32)

    samples = []
    for i, ply in enumerate(game.plies):
        if not ply_trainable(game, ply):
            continue
        s = per_ply[i]
        slots = [F.move_to_slot(m, b) for m in s["legal"]]
        assert 0 < len(slots) <= K
        probs = np.asarray(ply.pi, dtype=np.float64)
        probs /= probs.sum()                        # exact renorm (f16 rounding)

        pi_slots = np.full(K, slots[0], dtype=np.int64)
        pi_slots[: len(slots)] = slots
        pi_probs = np.zeros(K, dtype=np.float32)
        pi_probs[: len(slots)] = probs

        mover_wins = (s["side"] == BLACK) == (score > 0)
        score_mover = score if s["side"] == BLACK else -score
        own_mover = own_black if s["side"] == BLACK else 1.0 - own_black

        if i + 1 < len(per_ply):
            aux_idx = F.move_to_slot(game.plies[i + 1].move, b)
            aux_mask = F.policy_mask(per_ply[i + 1]["legal"], b)
            aux_w = 1.0
            assert aux_mask[aux_idx], "reply not in its legal set"
        else:
            aux_idx, aux_mask, aux_w = 0, F.policy_mask(s["legal"], b), 0.0

        base = {
            "tokens": s["tokens"],
            "mask": F.policy_mask(s["legal"], b),
            "pi_slots": pi_slots,
            "pi_probs": pi_probs,
            "wdl": F.wdl_onehot(mover_wins),
            "own": own_mover.astype(np.float32),
            "score": np.float32(score_mover),
            "aux_idx": np.int64(aux_idx),
            "aux_mask": aux_mask,
            "aux_w": np.float32(aux_w),
        }
        samples.extend(transform_sample(base, b, sym) for sym in syms)
    return samples


def stack_selfplay(samples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.stack([s[k] for s in samples]) for k in samples[0]}
