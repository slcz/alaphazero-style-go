"""
go_ai/net/features.py — move space + label helpers. There is NO Python
input encoder by design: the C encoder (engine/src/nn_encode.c) is the
single source of truth, reached through go_ai.engine.Board.encode().
"""
from __future__ import annotations

import numpy as np

POLICY_MAX = 362          # 361 points + pass (row stride for all boards)
PI_SUPPORT_MAX = 362      # dense π̄ support bound (== max legal count)

PASS = 0xFFFF


def policy_size(b: int) -> int:
    """Active slots for a b×b board: b*b points + pass."""
    return b * b + 1


def pass_slot(b: int) -> int:
    return b * b


def move_to_slot(move: int, b: int) -> int:
    """u16 point (or PASS) → policy slot. Pure arithmetic."""
    return b * b if move == PASS else move


def slot_to_move(slot: int, b: int) -> int:
    return PASS if slot == b * b else slot


def policy_mask(moves, b: int) -> np.ndarray:
    """Bool [POLICY_MAX]: True = legal. Slots beyond b²+1 stay False."""
    m = np.zeros(POLICY_MAX, dtype=bool)
    for mv in moves:
        m[move_to_slot(mv, b)] = True
    return m


def pi_dense(pi_slots, pi_probs, n_legal: int, b: int) -> np.ndarray:
    """(slots, probs) → dense over the policy head's b²+1 slots. The
    padding beyond n_legal repeats slot 0 at prob 0, so it must not be
    scattered."""
    d = np.zeros(policy_size(b), np.float32)
    d[np.asarray(pi_slots)[:n_legal]] = np.asarray(pi_probs)[:n_legal]
    return d


def wdl_onehot(mover_wins: bool) -> np.ndarray:
    """2-way [W, L] one-hot from the mover's view (no ties: fractional
    komi in selfplay)."""
    v = np.zeros(2, dtype=np.float32)
    v[0 if mover_wins else 1] = 1.0
    return v
