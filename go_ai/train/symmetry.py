"""
go_ai/train/symmetry.py — the square board's 8 symmetries (D4: 4
rotations × flip) as data augmentation.

A symmetry maps positions to positions and preserves the TT score, so
every trainable ply yields 8 equivalent (position, π̄, labels) samples.
Legality is preserved pointwise: placement equality survives any fixed
transform (T(A) = T(B) iff A = B), so superko verdicts transport.

What moves, per sample:
  tokens    rows permute; the edge-distance dims (7..10) are the only
            coordinate-derived features and are recomputed at the
            destination rows (one-hot / stone / broadcast dims travel
            with their row)
  pi_slots  π̄ probs ride their move; re-emitted in the transformed
            board's gen_legal order (ascending points, pass last)
  mask      the transformed legal set
  aux_idx   the reply move transforms (pass → pass)
  aux_mask  the after-position's legal set, transformed
  own       point permutation
  wdl / score / aux_w       invariant (score is symmetry-free)
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from go_ai.net import features as F
from go_ai.selfplay.shard import MOVE_PASS

_GNN_EDGE = 7            # dims 7..10: d_left, d_right, d_top, d_bottom (/b)
_maps: Dict[tuple, np.ndarray] = {}


def sym_point_map(b: int, s: int) -> np.ndarray:
    """fwd[p] = image of point p under symmetry s = 4*flip + k
    (k = 90° rotations applied after the flip). Cached per (b, s)."""
    assert 0 <= s < 8
    key = (b, s)
    m = _maps.get(key)
    if m is None:
        f, k = divmod(s, 4)
        fwd = np.empty(b * b, dtype=np.int64)
        for p in range(b * b):
            x, y = p % b, p // b
            if f:
                x = b - 1 - x
            for _ in range(k):
                x, y = y, b - 1 - x
            fwd[p] = y * b + x
        _maps[key] = m = fwd
    return m


def _inv(fwd: np.ndarray) -> np.ndarray:
    inv = np.empty_like(fwd)
    inv[fwd] = np.arange(fwd.shape[0])
    return inv


def transform_move(move: int, b: int, s: int) -> int:
    if move == MOVE_PASS:
        return MOVE_PASS
    return int(sym_point_map(b, s)[move])


def _transform_mask(mask: np.ndarray, b: int, fwd: np.ndarray) -> np.ndarray:
    """Legal-set mask → transformed legal-set mask (gen_legal order is
    ascending points + pass, preserved by sorting the images)."""
    n = b * b
    pts = np.nonzero(mask[:n])[0]
    out = np.zeros_like(mask)
    out[np.sort(fwd[pts])] = True
    out[n] = mask[n]                       # pass
    return out


def _transform_pi(smp: Dict[str, np.ndarray], b: int,
                  fwd: np.ndarray, inv: np.ndarray) -> None:
    n = b * b
    n_legal = int(smp["mask"].sum())
    slots = smp["pi_slots"][:n_legal]      # ascending points, pass last
    probs = smp["pi_probs"][:n_legal]
    assert slots[-1] == n                  # pass terminates the list
    prob_of = np.empty(n + 1, dtype=np.float64)
    prob_of[slots] = probs
    img_pts = np.sort(fwd[slots[:-1]])
    img_slots = np.append(img_pts, n)
    img_probs = prob_of[inv[img_pts]]
    img_probs = np.append(img_probs, prob_of[n])

    pi_slots = np.full(F.PI_SUPPORT_MAX, img_slots[0], dtype=np.int64)
    pi_slots[: len(img_slots)] = img_slots
    pi_probs = np.zeros(F.PI_SUPPORT_MAX, dtype=np.float32)
    pi_probs[: len(img_probs)] = img_probs
    smp["pi_slots"] = pi_slots
    smp["pi_probs"] = pi_probs / pi_probs[: len(img_probs)].sum()


def transform_sample(smp: Dict[str, np.ndarray], b: int,
                     s: int) -> Dict[str, np.ndarray]:
    """One sample (from selfplay_samples) under symmetry s. s = 0 returns
    an equal copy."""
    out = {k: v.copy() for k, v in smp.items()}
    if s == 0:
        return out
    fwd = sym_point_map(b, s)
    inv = _inv(fwd)
    n = b * b

    tok = out["tokens"][inv]               # row p' receives row T⁻¹(p')
    idx = np.arange(n)
    x, y = idx % b, idx // b
    tok[:, _GNN_EDGE] = x / b              # edge dims of the DESTINATION
    tok[:, _GNN_EDGE + 1] = (b - 1 - x) / b
    tok[:, _GNN_EDGE + 2] = y / b
    tok[:, _GNN_EDGE + 3] = (b - 1 - y) / b
    out["tokens"] = tok

    _transform_pi(out, b, fwd, inv)
    out["mask"] = _transform_mask(out["mask"], b, fwd)
    out["aux_mask"] = _transform_mask(out["aux_mask"], b, fwd)
    out["own"] = out["own"][inv]
    if int(out["aux_idx"]) != n:
        out["aux_idx"] = np.int64(fwd[int(out["aux_idx"])])
    # wdl / score / aux_w invariant
    return out
