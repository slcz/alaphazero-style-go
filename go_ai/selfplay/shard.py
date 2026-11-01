"""
go_ai/selfplay/shard.py — read GOSHARD1 shards written by
engine/build/selfplay (engine/include/go_sp_dump.h is the byte-format
authority). Deliberately minimal: struct parsing plus a light record
type; replaying positions for tensors happens in
go_ai.train.selfplay_data through the C engine.

Kill-safety: records are self-delimiting, so a run killed mid-write has
at most one truncated LAST record — detected as a short read and dropped
with a warning, never silently ingested.
"""
from __future__ import annotations

import json
import mmap
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_SHARD_HDR = struct.Struct("<8sIIQ")       # magic, version, worker, seed
_GAME_HDR = struct.Struct("<IIBBHff")      # idx, plies, b, end, sims, komi, score
_PLY_HDR = struct.Struct("<BHHHf")         # mover, move, n_pi, sims, outlook
MAGIC = b"GOSHARD1"
VERSIONS = (1,)

MOVE_PASS = 0xFFFF

SP_END_NATURAL = 1
SP_END_ADJUDICATED = 3


@dataclass(frozen=True)
class SpPly:
    mover: int                 # BLACK / WHITE
    move: int                  # u16 point, or MOVE_PASS
    sims: int                  # this ply's search budget (PCR filter)
    outlook: float             # mover-view root value
    pi: Tuple[float, ...]      # dense π̄ probs over the legal moves, in
                               # gen_legal order (points ascending, pass
                               # last); sums ~1 (f16 rounding)


@dataclass(frozen=True)
class SpGame:
    game_index: int
    n_plies: int
    b: int
    end_kind: int              # SP_END_*
    n_sims: int                # the slow cap (a ply trains iff sims == n_sims)
    komi: float
    score: float               # final TT margin, black view
    ownership: Tuple[float, ...]   # b*b entries, black view {0, .5, 1}
    plies: Tuple[SpPly, ...]


def scan_shard(path, want=None):
    """Yield (SpGame, start_offset, end_offset) per complete game — the
    offsets let a slicer cut byte-exact parts from a shard that is still
    being written. `want` (a set of game ids) fast-forwards every other
    game instead of parsing it, so a buffer that keeps a slice of a round
    pays only for what it trains on. Raises on a bad header."""
    data = Path(path).read_bytes()
    if len(data) < _SHARD_HDR.size:
        raise ValueError(f"{path}: shard shorter than its header")
    magic, version, _worker, _seed = _SHARD_HDR.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}")
    if version not in VERSIONS:
        raise ValueError(f"{path}: unsupported shard version {version}")
    off = _SHARD_HDR.size
    while off + _GAME_HDR.size <= len(data):
        start = off
        (gidx, n_plies, b, end_kind, n_sims, komi,
         score) = _GAME_HDR.unpack_from(data, off)
        off += _GAME_HDR.size
        n_pts = b * b
        if off + n_pts * 2 > len(data):
            break
        if want is not None and gidx not in want:
            off += n_pts * 2
            for _ in range(n_plies):
                if off + _PLY_HDR.size > len(data):
                    return
                off += _PLY_HDR.size + _PLY_HDR.unpack_from(data, off)[2] * 2
            continue
        own = struct.unpack_from(f"<{n_pts}e", data, off)
        off += n_pts * 2
        plies: List[SpPly] = []
        ok = True
        for _ in range(n_plies):
            if off + _PLY_HDR.size > len(data):
                ok = False
                break
            mover, move, n_pi, sims, outlook = _PLY_HDR.unpack_from(data, off)
            off += _PLY_HDR.size
            if off + n_pi * 2 > len(data):
                ok = False
                break
            pi = struct.unpack_from(f"<{n_pi}e", data, off)
            off += n_pi * 2
            plies.append(SpPly(mover, move, sims, outlook, pi))
        if not ok:
            break
        yield (SpGame(gidx, n_plies, b, end_kind, n_sims, komi, score,
                      own, tuple(plies)), start, off)


def game_extents(path: str | Path, start: Optional[int] = None):
    """Yield (start_offset, end_offset) per complete game, parsing record
    headers only — O(1) memory; identical boundaries to scan_shard."""
    with Path(path).open("rb") as f:
        hdr = f.read(_SHARD_HDR.size)
        if len(hdr) < _SHARD_HDR.size:
            raise ValueError(f"{path}: shard shorter than its header")
        magic, version, _w, _s = _SHARD_HDR.unpack(hdr)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}")
        if version not in VERSIONS:
            raise ValueError(f"{path}: unsupported shard version {version}")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as data:
            n = len(data)
            off = _SHARD_HDR.size if start is None else start
            while off + _GAME_HDR.size <= n:
                s = off
                n_plies, b = _GAME_HDR.unpack_from(data, off)[1:3]
                off += _GAME_HDR.size + b * b * 2
                ok = True
                for _ in range(n_plies):
                    if off + _PLY_HDR.size > n:
                        ok = False
                        break
                    n_pi = _PLY_HDR.unpack_from(data, off)[2]
                    off += _PLY_HDR.size + n_pi * 2
                    if off > n:
                        ok = False
                        break
                if not ok:
                    break
                yield (s, off)


def read_games(path, indices=None) -> List[SpGame]:
    """Only the requested game ids (None = all). Memory holds just the
    returned records: skipped games are fast-forwarded, not parsed."""
    return [g for g, _s, _e in scan_shard(
        path, None if indices is None else set(indices))]


def read_shard(path) -> List[SpGame]:
    """One shard file → game records; drops a truncated trailing record
    with a warning (the kill-mid-write case)."""
    games = []
    consumed = _SHARD_HDR.size
    for g, _, end in scan_shard(path):
        games.append(g)
        consumed = end
    if consumed != Path(path).stat().st_size:
        warnings.warn(f"{path}: truncated trailing game "
                      f"(got {len(games)} complete) — dropped")
    return games


def read_unit(unit_dir) -> List[SpGame]:
    """One work-unit directory → its games. DONE.json gates completeness
    (a unit without it is an interrupted run, not data)."""
    unit_dir = Path(unit_dir)
    done = json.loads((unit_dir / "DONE.json").read_text())
    records: List[SpGame] = []
    seen = set()
    for w in range(done["n_workers"]):
        for g in read_shard(unit_dir / f"shard-{w}.spbin"):
            key = (done["unit_seed"], g.game_index)
            if key not in seen:
                seen.add(key)
                records.append(g)
    assert len(records) == done["games_done"], \
        f"{unit_dir}: DONE says {done['games_done']} games, read {len(records)}"
    return records
