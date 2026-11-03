#!/usr/bin/env python3
"""plot_ply10_states.py — distribution of *board states* at ply 10 across
the it002-vs-it001 gate games, hashed with the engine (captures/side handled
by replay), D4-canonicalized for symmetry-equivalence."""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/sileizhang/go")
from go_ai.engine import Board, BLACK, WHITE, PASS

B = 9
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def sgf_move_to_pt(m: str) -> int:
    x = LETTERS.index(m[0])
    y = B - 1 - LETTERS.index(m[1])
    return y * B + x


def d4_canon(bd: np.ndarray) -> bytes:
    g = bd.reshape(B, B)
    best = None
    for f in (False, True):
        h = np.flip(g, axis=1) if f else g
        for k in range(4):
            r = np.rot90(h, k)
            t = r.tobytes()
            if best is None or t < best:
                best = t
    return best


def d4_canon_color(bd: np.ndarray) -> bytes:
    """canonical key including side-to-move: color-swap symmetric variant"""
    return d4_canon(bd)


exact = Counter()
canon = Counter()
n = 0
for f in Path("/tmp/it002gate/gate-sgf").glob("*.sgf"):
    s = f.read_text()
    moves = re.findall(r";([BW])\[([a-z]{0,2})\]", s)
    if len(moves) < 10:
        continue
    bd = Board(B, 7.5, 250)
    try:
        for _, m in moves[:10]:
            bd.push_fast(PASS if not m else sgf_move_to_pt(m))
    except Exception as e:
        continue
    st = bd.stones()
    n += 1
    exact[st.tobytes()] += 1
    canon[d4_canon(st)] += 1

fig, ax = plt.subplots(figsize=(9, 4.6))
for data, lbl, color in ((canon, f"D4-canonical ({len(canon)} unique states)",
                          "#2c7fb8"),
                         (exact, f"exact ({len(exact)} unique)", "#e6550d")):
    hist = Counter(data.values())
    xs = sorted(hist)
    ax.plot(xs, [hist[x] for x in xs], "o-", label=lbl, color=color, ms=4)
ax.set_xscale("log")
ax.set_xlabel("games sharing this ply-10 state")
ax.set_ylabel("number of states")
ax.set_title(f"ply-10 board-state multiplicity across {n} gate games")
ax.legend()
ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig("/Users/sileizhang/go/gate_sgf_it001/it002_ply10_states.png",
            dpi=130)
top = canon.most_common(5)
print(f"games={n} exact_unique={len(exact)} canonical_unique={len(canon)}")
print("top canonical states:", top)
