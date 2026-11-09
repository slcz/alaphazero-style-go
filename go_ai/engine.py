"""
go_ai/engine.py — Python binding over the C TT rules engine (cffi, ABI
mode).

Everything rules-shaped (legality, superko, scoring, feature encoding)
lives in C behind engine/include/go_ffi.h; this module drives it. The C
encoder is the single source of truth for input features — there is no
Python encoder. Moves are u16 points (y*b+x) or PASS; GTP strings never
appear on the training path.

    from go_ai.engine import Board, PASS

    b = Board(7, komi=9.5, maxply=150)
    b.legal_moves          # [0, 1, ..., 48, PASS] in gen_legal order
    b.push(24); ...
    b.termination          # (TERM code, score, ownership) — score/ownership
                           # valid once over
    b.encode()             # float32 [b*b, 18] (go_nn.h layout)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from cffi import FFI

_REPO = Path(__file__).resolve().parents[1]

PASS = 0xFFFF                    # == MOVE_PASS
BLACK, WHITE, EMPTY = 1, 2, 0
TERM_PLAYING, TERM_OVER = 0, 1

_CDEF = """
typedef struct GoGame GoGame;
void go_init(void);
GoGame *go_game_new(int b, float komi, int maxply);
void go_game_free(GoGame *g);
int go_side(const GoGame *g);
int go_ply(const GoGame *g);
int go_board_size(const GoGame *g);
int go_n_pass(const GoGame *g);
void go_board(const GoGame *g, uint8_t *out);
int go_legal(GoGame *g, uint16_t *out, int max);
int go_push(GoGame *g, uint16_t move);
int go_push_fast(GoGame *g, uint16_t move);
int go_termination(const GoGame *g, float *score, float *ownership);
int go_encode(const GoGame *g, float *out);
"""

_MAX_MOVES = 362                 # MAX_MOVES (points + pass)
_NN_TOKENS = 361                 # row stride; b*b active
_NN_TOKEN_DIM = 18               # layout spec: engine/include/go_nn.h


def _load_library():
    candidates = []
    env = os.environ.get("GO_LIB")
    if env:
        candidates.append(Path(env))
    else:
        candidates.append(_REPO / "engine" / "build" / "libgo.dylib")
        candidates.append(_REPO / "engine" / "build" / "libgo.so")

    ffi = FFI()
    ffi.cdef(_CDEF)
    for path in candidates:
        if path.exists():
            lib = ffi.dlopen(str(path))
            lib.go_init()
            return ffi, lib
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"libgo not found (tried: {tried}). Build it first: make -C engine"
    )


ffi, _lib = _load_library()


class IllegalMove(ValueError):
    """push() of a move that is not legal in the current position."""


class Board:
    """One Go game, backed by a C handle. Not thread-safe; any number of
    Boards may coexist. Replay is forward-only (no pop)."""

    def __init__(self, b: int = 7, komi: float = 9.5, maxply: int = 150):
        self._b = b
        self._g = _lib.go_game_new(b, komi, maxply)
        if self._g == ffi.NULL:
            raise ValueError(f"bad game config: b={b} maxply={maxply}")
        self._mv = ffi.new(f"uint16_t[{_MAX_MOVES}]")
        self._bd = ffi.new(f"uint8_t[{b * b}]")
        self._enc = ffi.new(f"float[{_NN_TOKENS * _NN_TOKEN_DIM}]")
        self._score = ffi.new("float *")
        self._own = ffi.new(f"float[{_NN_TOKENS}]")

    def __del__(self):
        g = getattr(self, "_g", None)
        if g is not None and g != ffi.NULL:
            _lib.go_game_free(g)
            self._g = ffi.NULL

    # --- position ------------------------------------------------------

    @property
    def b(self) -> int:
        return self._b

    @property
    def side_to_move(self) -> int:
        """BLACK or WHITE."""
        return _lib.go_side(self._g)

    @property
    def ply(self) -> int:
        return _lib.go_ply(self._g)

    @property
    def n_pass(self) -> int:
        return _lib.go_n_pass(self._g)

    def stones(self) -> np.ndarray:
        """Board as a [b*b] uint8 array (EMPTY/BLACK/WHITE)."""
        _lib.go_board(self._g, self._bd)
        return np.frombuffer(ffi.buffer(self._bd, self._b * self._b),
                             dtype=np.uint8).copy()

    # --- playing ---------------------------------------------------------

    @property
    def legal_moves(self) -> list:
        """Legal moves as u16 points (PASS last), in gen_legal order —
        the same order the shard's dense π̄ uses."""
        n = _lib.go_legal(self._g, self._mv, _MAX_MOVES)
        return [self._mv[i] for i in range(n)]

    def push(self, move: int) -> None:
        if _lib.go_push(self._g, move) != 0:
            raise IllegalMove(f"{move} is not legal at ply {self.ply}")

    def push_fast(self, move: int) -> None:
        """Trusted push (shard replay): occupancy check only, no
        gen_legal scan — for moves the engine itself recorded."""
        if _lib.go_push_fast(self._g, move) != 0:
            raise IllegalMove(f"{move} is not playable at ply {self.ply}")

    # --- observation -----------------------------------------------------

    def termination(self):
        """(code, score, ownership): code TERM_PLAYING while going; once
        over, score is the black-view TT margin and ownership the
        [b*b] black-view per-point vector ({0, 0.5, 1}), else None."""
        code = _lib.go_termination(self._g, self._score, self._own)
        if code == TERM_PLAYING:
            return code, None, None
        own = np.frombuffer(ffi.buffer(self._own, self._b * self._b * 4),
                            dtype=np.float32).copy()
        return code, self._score[0], own

    @property
    def is_game_over(self) -> bool:
        return _lib.go_termination(self._g, ffi.NULL, ffi.NULL) != TERM_PLAYING

    # --- NN input encoding -------------------------------------------------

    def encode(self) -> np.ndarray:
        """Feature tensor of the current position, float32 [b*b, 18]
        (layout spec: engine/include/go_nn.h; the C encoder is the
        authority)."""
        _lib.go_encode(self._g, self._enc)
        n = self._b * self._b
        arr = np.frombuffer(
            ffi.buffer(self._enc, _NN_TOKENS * _NN_TOKEN_DIM * 4),
            dtype=np.float32)
        return arr[:n * _NN_TOKEN_DIM].reshape(n, _NN_TOKEN_DIM).copy()

    def __repr__(self):
        return f"Board(b={self._b}, ply={self.ply})"
