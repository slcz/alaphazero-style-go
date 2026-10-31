#!/usr/bin/env python3
"""shard_to_sgf.py — convert games from a selfplay unit to SGF for
inspection in Sabaki.

    python scripts/shard_to_sgf.py --unit data/rand0 --n 10 --random

Shards are written in FINISH order, so the head of a unit is its
shortest games — use --random for a fair sample. Each node carries the
mover-view root value (outlook) as a comment; RE comes from the final
TT margin (black view)."""

import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from go_ai.selfplay.shard import MOVE_PASS, scan_shard

LETTERS = "abcdefghijklmnopqrstuvwxyz"   # plain a+index; the no-i skip
                                         # is a GTP display convention


def sgf_pt(move: int, b: int) -> str:
    if move == MOVE_PASS:
        return ""
    y, x = divmod(move, b)
    return LETTERS[x] + LETTERS[b - 1 - y]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--random", action="store_true",
                    help="sample n uniformly at random (shards are in "
                         "finish order: head = shortest games)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gen0_sgf")
    a = ap.parse_args()

    all_games = [g for shard in sorted(Path(a.unit).glob("shard-*.spbin"))
                 for g, _, _ in scan_shard(shard)]
    games = (random.Random(a.seed).sample(all_games, min(a.n, len(all_games)))
             if a.random else all_games[: a.n])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for g in games:
        margin = g.score
        re_ = (f"B+{margin:.1f}" if margin > 0
               else f"W+{-margin:.1f}" if margin < 0 else "0")
        nodes = []
        for ply in g.plies:
            col = "B" if ply.mover == 1 else "W"
            nodes.append(f";{col}[{sgf_pt(ply.move, g.b)}]"
                         f"C[{ply.outlook:+.3f}]")
        root = (f"(;GM[1]FF[4]SZ[{g.b}]KM[{g.komi}]RE[{re_}]"
                f"PB[gen0-{g.game_index}]PW[gen0-{g.game_index}]"
                f"C[end={'natural' if g.end_kind == 1 else 'cap'}; "
                f"sims={g.n_sims}]")
        (out / f"{g.game_index:05d}.sgf").write_text(
            root + "".join(nodes) + ")\n")
    print(f"wrote {len(games)} of {len(all_games)} games -> {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
