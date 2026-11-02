#!/usr/bin/env python3
"""endgame_report.py — do games end with dead stones still on the board?

Under Tromp-Taylor (area) scoring an uncaptured dead stone still scores its
own point, and capturing it is a +2 swing. This report measures how often
that actually matters: for every game that ends by double pass, replay to
the final board, run a deterministic "cleanup" sweep (capture every group
left in atari, alternating sides), re-score under TT, and compare.

Only groups with exactly one liberty are swept, so this is a LOWER BOUND on
the dead-stone cost: a dead clump with two or more liberties is left alone
(telling a live group from a dead one needs life-and-death, which is exactly
the domain knowledge the engine does not have).

    python3 scripts/endgame_report.py t/shard-0.spbin [--games N] [--verbose]

Columns: plies, TT margin before/after cleanup, captures swept, whether the
winner changed.
"""
from __future__ import annotations

import argparse
import random
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from go_ai.engine import BLACK, PASS, TERM_OVER, WHITE, Board  # noqa: E402
from go_ai.selfplay.shard import SP_END_NATURAL, read_shard  # noqa: E402


def _groups(stones, b, colour):
    """[(points, liberties)] for every group of `colour`."""
    seen = [False] * (b * b)
    out = []
    for start in range(b * b):
        if seen[start] or stones[start] != colour:
            continue
        pts, libs, stack = [], set(), [start]
        seen[start] = True
        while stack:
            p = stack.pop()
            pts.append(p)
            x, y = p % b, p // b
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if not (0 <= nx < b and 0 <= ny < b):
                    continue
                q = ny * b + nx
                if stones[q] == 0:
                    libs.add(q)
                elif stones[q] == colour and not seen[q]:
                    seen[q] = True
                    stack.append(q)
        out.append((pts, libs))
    return out


def _replay(game, extra=0):
    """Board at the last non-pass ply. The trailing passes carry no
    placement, and a finished game rejects further moves, so they are
    dropped and re-added when a score is wanted."""
    moves = [p.move for p in game.plies]
    while moves and moves[-1] == PASS:
        moves.pop()
    bd = Board(game.b, game.komi, game.n_plies + extra)
    for m in moves:
        bd.push_fast(m)
    return bd


def _finish(bd):
    """Pass it out and score: TT only reports a score once the game ends."""
    while not bd.is_game_over:
        bd.push(PASS)
    return bd.termination()


def cleanup(bd):
    """Capture every group left in atari, alternating sides. Returns the
    number of stones captured."""
    b, caps = bd.b, 0
    for _ in range(200):
        stones = bd.stones()
        legal = set(bd.legal_moves)
        opp = WHITE if bd.side_to_move == BLACK else BLACK
        target = None
        for pts, libs in _groups(stones, b, opp):
            if len(libs) == 1:
                lib = next(iter(libs))
                if lib in legal:
                    target = lib
                    break
        if target is not None:
            bd.push(target)             # captures that group
            after = bd.stones()
            caps += sum(1 for p in range(b * b)
                        if stones[p] == opp and after[p] == 0)
            continue
        if PASS in legal:               # hand the turn over; two passes end it
            bd.push(PASS)
            continue
        break
    return caps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("--games", type=int, default=400,
                    help="natural games to analyse (0 = all)")
    ap.add_argument("--verbose", action="store_true", help="per-game rows")
    ap.add_argument("--first", action="store_true",
                    help="take the first N instead of a random sample "
                         "(shards are in finish order, so the first games "
                         "are the shortest — biased)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    games = read_shard(args.shard)
    nat = [g for g in games if g.end_kind == SP_END_NATURAL]
    cap = len(games) - len(nat)
    if not args.games or args.games >= len(nat):
        sample = nat
    elif args.first:
        sample = nat[:args.games]
    else:
        sample = random.Random(args.seed).sample(nat, args.games)
    print(f"{Path(args.shard).name}: {len(games)} games — {len(nat)} double "
          f"pass, {cap} at the ply cap; analysing {len(sample)}")

    flips, swept, margins, deltas = 0, 0, [], []
    close = close_flips = 0
    settled = settled_flips = 0
    buckets = {}                        # ply bucket -> [games, flips]
    rows = []
    for g in sample:
        code, s0, _ = _finish(_replay(g))
        if code != TERM_OVER:
            continue
        if abs(s0 - g.score) > 1e-4:
            print(f"  score mismatch game {g.game_index}: {s0} vs {g.score}")
        bd = _replay(g, extra=200)
        atari = sum(1 for c in (BLACK, WHITE)
                    for _, libs in _groups(bd.stones(), bd.b, c)
                    if len(libs) == 1)
        caps = cleanup(bd)
        _, s1, _ = _finish(bd)
        flip = (s0 > 0) != (s1 > 0)
        flips += flip
        if abs(s0) <= 10.0:
            close += 1
            close_flips += flip
        if atari == 0:
            settled += 1
            settled_flips += flip
        lo = (g.n_plies // 40) * 40
        b_ = buckets.setdefault(lo, [0, 0])
        b_[0] += 1
        b_[1] += flip
        swept += caps > 0
        margins.append(abs(s0))
        deltas.append(abs(s1 - s0))
        if args.verbose and (flip or caps):
            rows.append(f"  game {g.game_index:6d} plies {g.n_plies:3d} "
                        f"TT {s0:+6.1f} -> {s1:+6.1f} ({caps} captured)"
                        f"{'  FLIP' if flip else ''}")
    for r in rows[:40]:
        print(r)

    n = len(margins)
    if not n:
        print("no natural games to analyse")
        return 1
    print(f"\n  end in atari           {n-settled:5d} / {n}  "
          f"({100*(n-settled)/n:.0f}% never settled)")
    print(f"  swept >0 stones        {swept:5d} / {n}  ({100*swept/n:.0f}%)")
    print(f"  result changed         {flips:5d} / {n}  ({100*flips/n:.0f}%)")
    print(f"  plies (median)         {st.median(g.n_plies for g in sample):5.0f}")
    print(f"  close games (|TT|<=10) {close:5d} / {n}  "
          f"-> flipped {close_flips} ({100*close_flips/max(close,1):.0f}%)")
    print(f"  settled at the pass    {settled:5d} / {n}  "
          f"(no group in atari) -> flipped {settled_flips}")
    for lo in sorted(buckets):
        c, f = buckets[lo]
        print(f"  plies {lo:3d}-{lo+39:3d}       {c:5d} games, "
              f"{f:3d} flipped ({100*f/c:.0f}%)")
    print(f"  mean |TT margin|       {st.mean(margins):5.1f}")
    print(f"  mean |change|          {st.mean(deltas):5.1f} points")
    print(f"  max  |change|          {max(deltas):5.1f} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
