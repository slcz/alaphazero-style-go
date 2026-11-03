#!/usr/bin/env python3
"""plot_census.py — it001 selfplay: score + ply distributions."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.load(open("/tmp/it001_census.json"))
plies = [d["plies"] for d in data]
scores = [d["score"] for d in data]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.hist(plies, bins=50, color="#4477aa", edgecolor="white")
ax.axvline(250, color="red", ls="--", lw=1, label="ply cap 250")
ax.set_xlabel("plies per game")
ax.set_ylabel("games")
ax.set_title(f"game length (n={len(plies)}, mean={sum(plies)/len(plies):.0f})")
ax.legend()

ax = axes[1]
ax.hist(scores, bins=101, color="#ee7733", edgecolor="white")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("TT score (black view; komi 7.5)")
ax.set_ylabel("games")
ax.set_title(f"final score margin (mean={sum(scores)/len(scores):.1f})")

plt.tight_layout()
plt.savefig("/Users/sileizhang/go/gate_sgf_it001/it001_dist.png", dpi=130)
print("saved")
