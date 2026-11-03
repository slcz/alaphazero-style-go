#!/usr/bin/env python3
"""plot_ply10.py — distribution of stone locations at ply 10 across the
it002-vs-it001 gate games, plotted on the 9x9 board."""
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = 9
LETTERS = "abcdefghijklmnopqrstuvwxyz"
cnt = Counter()
n_games = 0
for f in Path("/tmp/it002gate/gate-sgf").glob("*.sgf"):
    s = f.read_text()
    moves = re.findall(r";([BW])\[([a-z]{0,2})\]", s)
    stones = [(c, m) for c, m in moves[:10] if m]
    if len(moves) < 10:
        continue
    n_games += 1
    for _, m in stones:
        x = LETTERS.index(m[0])
        y = B - 1 - LETTERS.index(m[1])
        cnt[(x, y)] += 1

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2),
                         gridspec_kw={"width_ratios": [1, 1.3]})
ax = axes[0]
ax.set_title(f"stones at ply 10 ({n_games} games)")
for i in range(B):
    for j in range(B):
        ax.add_patch(plt.Rectangle((i - .5, j - .5), 1, 1, fill=False,
                                   edgecolor="#bbb", lw=.5))
mx = max(cnt.values())
for (x, y), c in cnt.items():
    ax.add_patch(plt.Circle((x, y), .12 + .3 * (c / mx) ** .5,
                            color="#c0392b" if True else "k",
                            alpha=.25 + .7 * (c / mx) ** .5))
ax.plot([-.5, 8.5, 8.5, -.5, -.5], [-.5, -.5, 8.5, 8.5, -.5], "k-", lw=1.5)
ax.set_xlim(-1, 9); ax.set_ylim(-1, 9)
ax.set_aspect("equal"); ax.axis("off")

ax = axes[1]
hist = Counter(c for c in cnt.values())
xs = sorted(hist)
ax.bar(xs, [hist[x] for x in xs], color="#2c7fb8")
ax.set_xlabel("games having a stone on a point (of 400)")
ax.set_ylabel("points")
ax.set_title("per-point occupancy histogram @ ply 10")
plt.tight_layout()
plt.savefig("/Users/sileizhang/go/gate_sgf_it001/it002_gate_ply10.png", dpi=130)
print(f"games={n_games} unique_points={len(cnt)} max_count={mx}")
