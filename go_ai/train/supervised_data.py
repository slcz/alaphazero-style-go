"""
go_ai/train/supervised_data.py — SGF game records → training samples.

Same tensor schema as selfplay (selfplay_data), different labels:
    pi_probs   one-hot on the PLAYED move over the legal support (the
               dense-π̄ CE then reduces to plain CE on the human move)
    wdl        the game's RE result (resignations are decisive — fine
               for the 2-way head)
    aux        the opponent's actual reply
    own/score  NO labels (no scoring position exists in resigned games)
               — keys are simply absent; the trainer skips those losses

Samples are built on the fly (a 27k-game cache is ~2 GB as tensors —
never materialized), through the C encoder so the input contract is
byte-identical to selfplay. D4 augmentation applies as in selfplay.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from go_ai.data.sgf import SgfGame
from go_ai.engine import BLACK, PASS, Board
from go_ai.net import features as F
from go_ai.selfplay.shard import MOVE_PASS


def cache_games(games: List[SgfGame], path: str | Path) -> None:
    """Serialize parsed games (fast re-runs: parsing 40k files is ~90 s)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(games, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache(path: str | Path) -> List[SgfGame]:
    with open(path, "rb") as f:
        return pickle.load(f)


def supervised_sample(game: SgfGame, i: int, sym: int = 0) -> Dict[str, np.ndarray]:
    """Sample for the position before `game.moves[i]` under symmetry sym."""
    b = game.b
    board = Board(b, game.komi, 1024)
    for mv in game.moves[:i]:
        board.push_fast(mv)
    legal = board.legal_moves
    played = game.moves[i]
    tokens = board.encode()

    slots = [F.move_to_slot(m, b) for m in legal]
    pi_slots = np.full(F.PI_SUPPORT_MAX, slots[0], dtype=np.int64)
    pi_slots[: len(slots)] = slots
    pi_probs = np.zeros(F.PI_SUPPORT_MAX, dtype=np.float32)
    pi_probs[slots.index(F.move_to_slot(played, b))] = 1.0

    if i + 1 < len(game.moves):
        after = Board(b, game.komi, 1024)
        for mv in game.moves[:i + 1]:
            after.push_fast(mv)
        aux_idx = F.move_to_slot(game.moves[i + 1], b)
        aux_mask = F.policy_mask(after.legal_moves, b)
        aux_w = 1.0
        assert aux_mask[aux_idx], "reply not in its legal set"
    else:
        aux_idx, aux_mask, aux_w = 0, F.policy_mask(legal, b), 0.0

    from go_ai.train.symmetry import transform_sample
    base = {
        "tokens": tokens,
        "mask": F.policy_mask(legal, b),
        "pi_slots": pi_slots,
        "pi_probs": pi_probs,
        "wdl": F.wdl_onehot(game.black_wins == (board.side_to_move == BLACK)),
        "aux_idx": np.int64(aux_idx),
        "aux_mask": aux_mask,
        "aux_w": np.float32(aux_w),
    }
    return transform_sample(base, b, sym)
