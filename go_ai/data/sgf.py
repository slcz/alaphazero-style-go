"""
go_ai/data/sgf.py — SGF game records → compact games for supervised
training (sgfmill does the format parsing; this module applies OUR
policy: even games only, main line, C-engine legality).

A game is kept iff: SZ matches, no handicap/setup stones (AB/AW), a
decisive result (RE B+/W+, not jigo/void), and every main-line move is
legal in the C engine (TT rules — a few SGF moves differ, e.g. suicide
variants; those games are dropped rather than mislabeled).

Coordinates: SGF (col, row) letters with 'a' row at the TOP; our point
index is y*b + x with y = 0 at GTP row 1 (BOTTOM), so y = b-1-row.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from sgfmill import sgf, boards

from go_ai.engine import PASS, Board, IllegalMove

WHITE, BLACK = 2, 1


@dataclass(frozen=True)
class SgfGame:
    moves: tuple            # u16 points, PASS for a pass
    black_wins: bool
    komi: float
    b: int


def parse_sgf_bytes(data: bytes, b: int = 19) -> Optional[SgfGame]:
    """One SGF file → SgfGame, or None if it fails OUR filters."""
    try:
        game = sgf.Sgf_game.from_bytes(data)
    except ValueError:
        return None
    root = game.get_root()
    if game.get_size() != b:
        return None
    if root.has_property("AB") or root.has_property("AW"):
        return None                       # handicap/setup: different label
    try:
        if root.get("HA") not in (0, None):
            return None
    except KeyError:
        pass
    try:
        re_ = root.get("RE").strip()
    except KeyError:
        return None
    if re_.startswith("B+"):
        black_wins = True
    elif re_.startswith("W+"):
        black_wins = False
    else:                                 # jigo, void, time?, unfinished
        return None
    try:
        komi = float(root.get("KM"))
    except KeyError:
        return None

    raw = []
    for node in game.get_main_sequence()[1:]:
        color, move = node.get_move()
        if color is None:
            continue                      # annotations, not moves
        if move is None:
            raw.append(PASS)
        else:
            row, col = move               # sgfmill: (row, col), row 0
            raw.append(row * b + col)     # at the BOTTOM — same as ours

    # C-engine legality: replay everything; drop games with any illegal
    # move (or a move after the game went terminal in TT terms)
    bd = Board(b, komi=komi, maxply=1024)
    moves = []
    for mv in raw:
        if bd.is_game_over:
            return None
        try:
            bd.push(mv)
        except IllegalMove:
            return None
        moves.append(mv)
    if len(moves) < 10:                   # record noise
        return None
    return SgfGame(tuple(moves), black_wins, komi, b)


def load_dir(path: str | Path, b: int = 19) -> List[SgfGame]:
    """Every *.sgf under `path` (recursive), filtered. Logs nothing."""
    games = []
    for f in sorted(Path(path).rglob("*.sgf")):
        g = parse_sgf_bytes(f.read_bytes(), b)
        if g is not None:
            games.append(g)
    return games
