#!/usr/bin/env python3
"""shard_to_sgf.py — convert the first N games of a selfplay unit to
SGF for inspection in Sabaki.

    python scripts/shard_to_sgf.py --unit data/gen0 --n 10 --out gen0_sgf

Each node carries the mover-view root value (outlook) as a comment, so
Sabaki shows the winrate the search saw at every move. RE comes from the
final TT margin (black view)."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from go_ai.selfplay.shard import MOVE_PASS, scan_shard

LETTERS = "abcdefghijklmnopqrstuvwxyz"   # plain a+index; the no-i skip is GTP display only


def sgf_pt(move: int, b: int) -> str:
    if move == MOVE_PASS:
        return ""
    y, x = divmod(move, b)
    return LETTERS[x] + LETTERS[b - 1 - y]     # SGF rows run top-down


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="gen0_sgf")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for shard in sorted(Path(a.unit).glob("shard-*.spbin")):
        if written >= a.n:
            break
        for g, _, _ in scan_shard(shard):
            if written >= a.n:
                break
            margin = g.score                    # black view, komi included
            re_ = (f"B+{margin:.1f}" if margin > 0
                   else f"W+{-margin:.1f}" if margin < 0 else "0")
            nodes = []
            for ply in g.plies:
                col = "B" if ply.mover == 1 else "W"
                c = f"C[{ply.outlook:+.3f}]"    # mover-view root value
                nodes.append(f";{col}[{sgf_pt(ply.move, g.b)}]{c}")
            root = (f"(;GM[1]FF[4]SZ[{g.b}]KM[{g.komi}]RE[{re_}]"
                    f"PB[gen0-{g.game_index}]PW[gen0-{g.game_index}]"
                    f"C[end={'natural' if g.end_kind == 1 else 'cap'}; "
                    f"sims={g.n_sims}]")
            (out / f"{g.game_index:05d}.sgf").write_text(root + "".join(nodes) + ")\n")
            written += 1
    print(f"wrote {written} games -> {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
