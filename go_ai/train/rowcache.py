"""
go_ai/train/rowcache.py — materialized per-round row cache (the trainer's
backing store) + the batch builder.

The old path replayed a game from ply 0 through cffi for EVERY sample, so
a 3072-row step spent ~0.45 s on CPU and left the GPU at ~18% duty. The
cache replays each game ONCE per round and stores only what cannot be
recomputed cheaply:

    bits   u8  [R,12,ceil(T/8)]  the 12 BINARY token dims, bit-packed
    mask   u8  [R,ceil(K/8)]     legal slots of the position (superko!)
    pi     f16 [R,K]             π̄ dense over slots (points, pass last)
    own    u8  [R,T]             {0,1,2} → {0,.5,1}, mover view
    sb     i16 [R]               score bin
    win    u8 [R]                mover won
    aux_i  u16 [R]               opponent reply slot (0 when aux_w = 0)
    aux_w  u8 [R]
    auxm   u8  [R,ceil(K/8)]     legal slots after our move
    game   u32 [R]               the game each row came from

with T = b², K = T+1. The 6 STATIC token dims (4 edge distances, komi,
size) are per-round constants written into the preallocated batch, and
the 12 binary dims are exactly token dims [0..6, 11..15]. ~400 B/row
instead of ~3.5 KB for raw tokens: a round costs ~0.5 GB and a step is a
handful of numpy gathers.

The cache covers a WHOLE round; the replay buffer's selection (which
games the manifest kept) is applied as a row-index array, so the cache
survives the manifest shrinking every iteration. It is invalidated only
when the shard's byte size changes.

Correctness contract: `fill_batch` reproduces `selfplay_sample` up to the
symmetry — tests/test_rowcache.py asserts equality against the C encoder.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from go_ai.engine import Board
from go_ai.net import features as F
from go_ai.net.model import score_to_bin
from go_ai.selfplay.shard import scan_shard
from go_ai.train.symmetry import sym_point_map

CACHE_FILE = "rows-v2.npz"
KEYS = ("bits", "mask", "pi", "own", "sb", "win", "aux_i", "aux_w", "auxm",
        "game")
BIN_DIMS = np.array([0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15])
OWN_LUT = np.array([0.0, 0.5, 1.0], np.float32)


# ---------------------------------------------------------------------------
# static (symmetry-independent) token dims
# ---------------------------------------------------------------------------

_stat: dict = {}


def static_tokens(b: int, komi: float) -> np.ndarray:
    """[T,18] with the 6 static dims filled, the 12 binary dims zero.
    Edge distances are recomputed from the DESTINATION index, so this
    template is the same under every D4 symmetry (the 4 edge channels are
    board directions, not point labels)."""
    key = (b, round(float(komi), 6))
    t = _stat.get(key)
    if t is None:
        T = b * b
        t = np.zeros((T, 18), np.float32)
        idx = np.arange(T)
        x, y = idx % b, idx // b
        fb = np.float32(b)
        t[:, 7] = np.float32(x) / fb
        t[:, 8] = np.float32(b - 1 - x) / fb
        t[:, 9] = np.float32(y) / fb
        t[:, 10] = np.float32(b - 1 - y) / fb
        t[:, 16] = np.float32(komi) / np.float32(15.0)
        t[:, 17] = np.float32(b) / np.float32(19.0)
        _stat[key] = t
    return t


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------

def _pack_slots(moves, b: int, k: int) -> np.ndarray:
    m = np.zeros(k, bool)
    m[[F.move_to_slot(mv, b) for mv in moves]] = True
    return np.packbits(m)


def _bits_of(tok: np.ndarray) -> np.ndarray:
    """[T,18] tokens → [12, ceil(T/8)] packed binary dims."""
    return np.packbits(tok[:, BIN_DIMS].T.astype(bool), axis=1)


def _empty_rows(b: int) -> dict:
    T = b * b
    nb, nbit = (T + 8) // 8, (T + 7) // 8
    return {"bits": np.zeros((0, 12, nbit), np.uint8),
            "mask": np.zeros((0, nb), np.uint8),
            "pi": np.zeros((0, T + 1), np.float16),
            "own": np.zeros((0, T), np.uint8),
            "sb": np.zeros(0, np.int16), "win": np.zeros(0, np.uint8),
            "aux_i": np.zeros(0, np.uint16), "aux_w": np.zeros(0, np.uint8),
            "auxm": np.zeros((0, nb), np.uint8),
            "game": np.zeros(0, np.uint32)}


def _game_rows(g) -> dict:
    """Compact rows for one game's slow-cap plies (one forward replay)."""
    b, K, n = g.b, g.b * g.b + 1, len(g.plies)
    board = Board(b, g.komi, g.n_plies)
    own_black = np.asarray(g.ownership, np.float32)
    score = float(g.score)
    out = {k: [] for k in KEYS}

    legal = board.legal_moves
    for i, ply in enumerate(g.plies):
        train = ply.sims == g.n_sims
        if train:
            side = board.side_to_move
            tok = _bits_of(board.encode())
            legal_here = legal
        board.push_fast(ply.move)
        if i + 1 < n:
            legal = board.legal_moves
        if not train:
            continue
        out["bits"].append(tok)
        out["mask"].append(_pack_slots(legal_here, b, K))
        probs = np.asarray(ply.pi, np.float64)
        p = np.zeros(K, np.float32)
        p[[F.move_to_slot(mv, b) for mv in legal_here]] = probs / probs.sum()
        out["pi"].append(p.astype(np.float16))
        own = own_black if side == 1 else 1.0 - own_black
        out["own"].append(np.rint(own * 2).astype(np.uint8))
        out["sb"].append(score_to_bin(score if side == 1 else -score))
        out["win"].append(1 if (side == 1) == (score > 0) else 0)
        if i + 1 < n:                       # aux: the actual reply
            out["aux_i"].append(F.move_to_slot(g.plies[i + 1].move, b))
            out["aux_w"].append(1)
            out["auxm"].append(_pack_slots(legal, b, K))
        else:                               # no reply: weight 0
            out["aux_i"].append(0)
            out["aux_w"].append(0)
            out["auxm"].append(_pack_slots(legal_here, b, K))
        out["game"].append(g.game_index)
    if not out["bits"]:
        return _empty_rows(b)
    return {k: np.stack(v) for k, v in out.items()}


def build_rows(unit_dir, log=print) -> dict:
    """One round's shard → compact rows (streamed: one game at a time, so
    the parsed game list never has to fit in memory)."""
    t0 = time.time()
    acc = {k: [] for k in KEYS}
    b = komi = None
    games = 0
    for g, _s, _e in scan_shard(Path(unit_dir) / "shard-0.spbin"):
        b, komi = g.b, g.komi
        r = _game_rows(g)
        for k in KEYS:
            acc[k].append(r[k])
        games += 1
    out = {k: np.concatenate(v) for k, v in acc.items()}
    mb = sum(v.nbytes for v in out.values()) / 2**20
    log(f"  rows {Path(unit_dir).name}: {games} games, "
        f"{len(out['sb'])} rows, {mb:.0f} MB, {time.time() - t0:.0f}s")
    return {**out, "b": np.int32(b), "komi": np.float32(komi)}


def load_or_build(unit_dir, log=print) -> dict:
    """Round cache, built on first use; invalidated when the shard's byte
    size changes (a regenerated round)."""
    unit_dir = Path(unit_dir)
    stamp = str((unit_dir / "shard-0.spbin").stat().st_size)
    f = unit_dir / CACHE_FILE
    if f.exists():
        z = np.load(f)
        if str(z["stamp"]) == stamp:
            return {k: z[k] for k in KEYS + ("b", "komi")}
    rows = build_rows(unit_dir, log=log)
    np.savez(f, stamp=np.array(stamp), **rows)
    return rows


def select(cache, game_ids) -> np.ndarray:
    """Row indices belonging to `game_ids` (None = every row)."""
    game = cache["game"]
    if game_ids is None:
        return np.arange(len(game))
    keep = np.zeros(int(game.max()) + 1 if len(game) else 1, bool)
    keep[np.asarray(game_ids, np.int64)] = True
    return np.flatnonzero(keep[game])


# ---------------------------------------------------------------------------
# batch assembly
# ---------------------------------------------------------------------------

def alloc_batch(n: int, b: int) -> dict:
    T = b * b
    return {"tokens": np.zeros((n, T, 18), np.float32),
            "mask": np.zeros((n, F.POLICY_MAX), bool),
            "pi": np.zeros((n, T + 1), np.float32),
            "wdl": np.zeros((n, 2), np.float32),
            "own": np.zeros((n, T), np.float32),
            "score_bin": np.zeros(n, np.int64),
            "aux_idx": np.zeros(n, np.int64),
            "aux_mask": np.zeros((n, F.POLICY_MAX), bool),
            "aux_w": np.zeros(n, np.float32)}


def fill_batch(caches, picks, sym, out, b: int, komi: float) -> int:
    """Rows `picks[k]` of `caches[k]` under D4 symmetry `sym` → `out`
    (in place). One symmetry for the whole batch: rows are drawn ~8 times
    per epoch, so independent per-step draws give the same distribution
    for a fraction of the index arithmetic. Returns the row count."""
    T, K = b * b, b * b + 1
    fwd = sym_point_map(b, sym)
    inv = np.empty(T, np.int64)
    inv[fwd] = np.arange(T)
    slotmap = np.append(fwd, T)            # aux slots: points + pass
    stat = static_tokens(b, komi)
    off = 0
    for c, idx in zip(caches, picks):
        n = len(idx)
        if not n:
            continue
        sl = slice(off, off + n)
        off += n
        tok = out["tokens"][sl]
        tok[:] = stat
        u = np.unpackbits(c["bits"][idx], axis=2)[:, :, inv]      # [n,12,T]
        tok[:, :, BIN_DIMS] = u.transpose(0, 2, 1)

        pi = out["pi"][sl]
        pi[:] = 0.0
        pi[:, :T] = c["pi"][idx][:, inv]
        pi[:, T] = c["pi"][idx][:, T]
        pi /= pi.sum(-1, keepdims=True)

        mk = out["mask"][sl]
        mk[:] = False
        bits = np.unpackbits(c["mask"][idx], axis=1)
        mk[:, :T] = bits[:, inv]
        mk[:, T] = bits[:, T]

        am = out["aux_mask"][sl]
        am[:] = False
        abits = np.unpackbits(c["auxm"][idx], axis=1)
        am[:, :T] = abits[:, inv]
        am[:, T] = abits[:, T]

        out["own"][sl] = OWN_LUT[c["own"][idx][:, inv]]
        out["score_bin"][sl] = c["sb"][idx]
        win = c["win"][idx]
        out["wdl"][sl, 0] = win
        out["wdl"][sl, 1] = 1.0 - win
        out["aux_idx"][sl] = slotmap[c["aux_i"][idx]]
        out["aux_w"][sl] = c["aux_w"][idx]
    return off


def draw(n: int, sels, rng) -> list:
    """n row ids drawn uniformly from the selected rows of every round."""
    sizes = np.array([len(s) for s in sels], np.int64)
    which = np.searchsorted(np.cumsum(sizes),
                            rng.integers(0, int(sizes.sum()), n))
    picks = []
    for i, s in enumerate(sels):
        m = int((which == i).sum())
        picks.append(s[rng.integers(0, len(s), m)] if m
                     else np.zeros(0, np.int64))
    return picks
