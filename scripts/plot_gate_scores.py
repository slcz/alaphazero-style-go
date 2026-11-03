#!/usr/bin/env python3
"""plot_gate_scores.py — final TT margin distribution of the it002-vs-it001
gate games (and vs-random for contrast)."""
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def scores_from(d: str) -> list:
    out = []
    for f in Path(d).glob("*.sgf"):
        m = re.search(r"RE\[(?:B|W)\+([0-9.]+)\]", f.read_text())
        if m:
            out.append(float(m.group(1)))
    return out


it2v1 = scores_from("/tmp/it002gate/gate-sgf")
it2vr = scores_from("/tmp/it002gate")   # vs-rand pack extracted flat earlier?

fig, ax = plt.subplots(figsize=(10, 4.6))
bins = [x + 0.5 for x in range(0, 100, 2)]   # 1-point TT lattice, 2-wide bins
ax.hist(it2v1, bins=bins, alpha=.75, label=f"it002 vs it001 ({len(it2v1)})",
        color="#2c7fb8")
ax.set_xlabel("|TT score margin| (1-point lattice)")
ax.set_ylabel("games")
ax.set_title("final score margin, it002-vs-it001 gate (400 games)")
ax.legend()
plt.tight_layout()
plt.savefig("/Users/sileizhang/go/gate_sgf_it001/it002_gate_scores.png",
            dpi=130)
c = Counter(it2v1)
print("it2 vs it1:", len(it2v1), "games")
print("top scores:", c.most_common(8))
print("unique scores:", len(c))
