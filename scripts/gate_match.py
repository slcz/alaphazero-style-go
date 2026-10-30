#!/usr/bin/env python3
"""gate_match.py — two-network gate matches in net-lockstep waves
(ported from the chess gate; the referee drives the C rules via cffi).

Two `engine_mux` processes (one per checkpoint, one network each) host
ALL of a wave's games through their own landing-zone mux, so every phase
batches all live games on the GPU. Wave 0 gives network A Black in every
game; wave 1 swaps colors. Openings: the referee plays r uniformly random
legal point moves per game (r ~ geometric over 1/4 of the board area's
square root scale — kept simple: r = rng.integers(0, 4) on 7x7, scaled by
b/7), then the engines think.

Resignation is the DRIVER's call (AGENTS.md): when an engine's root value
exceeds --resign (0.95 default) against it and the game has passed
--resign-ply plies, the referee resigns it. Scores are from network A's
perspective: 1 win, 0 loss (jigo cannot occur — fractional komi).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from go_ai.engine import PASS, TERM_OVER, Board

DRIVER = REPO / "engine" / "build" / "engine_mux"


class Engine:
    """One engine_mux process = one network hosting all of a wave's games."""

    label = "?"

    def __init__(self, ckpt, device, games, rows, walkers, seed, label="?"):
        self.label = label
        env = dict(os.environ, PUCT_CKPT=str(ckpt))
        if walkers:
            env["PUCT_WALKERS"] = str(walkers)
        argv = [str(DRIVER), str(games), str(rows), device, str(seed)]
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True,
                                  env=env, cwd=str(REPO))
        while True:
            line = self.p.stdout.readline().strip()
            if line == "ready":
                break
            if not line:
                print(f"engine {ckpt} failed to attach", file=sys.stderr)
                sys.exit(1)

    def send(self, line):
        if os.environ.get("GATE_DEBUG"):
            print(f"> [{self.label}] {line}", file=sys.stderr, flush=True)
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def read(self):
        line = self.p.stdout.readline()
        if not line:
            print(f"engine {self.label} died", file=sys.stderr)
            sys.exit(1)
        return line.strip()

    def quit(self):
        try:
            self.send("quit")
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def phase(eng, sims, games, boards, moves, sent):
    """One think round for every live game on `eng`'s turn: batched by the
    driver (all `go`s, then `run`), one bestmove per game. `moves` is the
    referee's per-game move list (our Board is forward-only, so history
    lives here); `sent` counts moves already fed to this engine. Returns
    {gid: (move, value)}."""
    pending = [gid for gid in games if boards[gid].side_to_move ==
               eng.side and not boards[gid].is_game_over]
    for gid in pending:
        suffix = moves[gid][sent[gid]:]
        if suffix:
            tokens = [str(m) if m != PASS else "pass" for m in suffix]
            eng.send(f"game {gid} moves " + " ".join(tokens))
        eng.send(f"game {gid} go nodes {sims}")
    eng.send("run")
    out = {}
    for gid in pending:
        parts = eng.read().split()
        # game <gid> bestmove <m> value <v>
        if (len(parts) != 6 or parts[0] != "game" or parts[2] != "bestmove"
                or int(parts[1]) != gid or parts[4] != "value"):
            print(f"protocol error: {parts}", file=sys.stderr)
            sys.exit(1)
        m = PASS if parts[3] == "pass" else int(parts[3])
        out[gid] = (m, float(parts[5]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--a-sims", type=int, default=200)
    ap.add_argument("--b-sims", type=int, default=200)
    ap.add_argument("--games", type=int, default=16,
                    help="games PER WAVE (total = 2x; colors swap per wave)")
    ap.add_argument("--b", type=int, default=7)
    ap.add_argument("--komi", type=float, default=9.5)
    ap.add_argument("--max-plys", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rows", type=int, default=0, help="0 = 8*games")
    ap.add_argument("--walkers", type=int, default=0)
    ap.add_argument("--openings", type=int, default=1,
                    help="max random opening moves per game (0..N drawn "
                         "uniformly per game; 0 = empty-board starts only)")
    ap.add_argument("--resign", type=float, default=0.95,
                    help="mover-view loss value that resigns (0 = off)")
    ap.add_argument("--resign-ply", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.randrange(1 << 31)
    rng = random.Random(seed)
    print(f"GATE seed={seed} b={args.b} komi={args.komi} games/wave={args.games} "
          f"a_sims={args.a_sims} b_sims={args.b_sims} resign={args.resign}",
          flush=True)

    eng_a = Engine(args.ckpt_a, args.device, args.games, args.rows,
                   args.walkers, (seed * 2 + 1) % (1 << 64), label="A")
    eng_a.side = 1                                   # BLACK in wave 0
    eng_b = Engine(args.ckpt_b, args.device, args.games, args.rows,
                   args.walkers, (seed * 2 + 2) % (1 << 64), label="B")
    eng_b.side = 2

    t0 = time.time()
    total = {"w": 0, "l": 0}
    waves = []
    for wave in (0, 1):
        if wave == 1:                                # swap colors
            eng_a.side, eng_b.side = eng_b.side, eng_a.side
        boards = [Board(args.b, args.komi, args.max_plys)
                  for _ in range(args.games)]
        moves = [[] for _ in range(args.games)]
        sent_a = [0] * args.games
        sent_b = [0] * args.games
        resigned = [False] * args.games
        # plant every game at the match config (fresh tree per wave)
        for gid in range(args.games):
            eng_a.send(f"game {gid} config {args.b} {args.komi} "
                       f"{args.max_plys}")
            eng_b.send(f"game {gid} config {args.b} {args.komi} "
                       f"{args.max_plys}")
        # opening randomization: r random legal point moves per game
        for gid, bd in enumerate(boards):
            r = rng.randint(0, args.openings)
            for _ in range(r):
                legal = [m for m in bd.legal_moves if m != PASS]
                if not legal:
                    break
                m = rng.choice(legal)
                bd.push(m)
                moves[gid].append(m)
        alive = list(range(args.games))
        w = l = 0
        t_wave = time.time()
        while alive:
            for eng, sent in ((eng_a, sent_a), (eng_b, sent_b)):
                sims = args.a_sims if eng is eng_a else args.b_sims
                for gid, (mv, val) in phase(eng, sims, alive, boards,
                                            moves, sent).items():
                    if (args.resign > 0 and -val >= args.resign
                            and boards[gid].ply >= args.resign_ply):
                        resigned[gid] = True         # mover resigns
                        continue
                    boards[gid].push(mv)
                    moves[gid].append(mv)
                    # the engine played its own bestmove into its tree
                    # already: mark it sent so the next suffix holds only
                    # the opponent's replies
                    sent[gid] = len(moves[gid])
            still = []
            a_is_black = eng_a.side == 1
            for gid in alive:
                bd = boards[gid]
                if resigned[gid]:
                    sc = 0.0 if bd.side_to_move == eng_a.side else 1.0
                    score = None
                else:
                    code, score, _ = bd.termination()
                    if code != TERM_OVER:
                        still.append(gid)
                        continue
                    sc = 1.0 if (score > 0) == a_is_black else 0.0
                if sc == 1.0:
                    w += 1
                else:
                    l += 1
                print(f"GAME wave={wave} game={gid} "
                      f"{'resign' if resigned[gid] else 'over'} "
                      f"plies={bd.ply} score={score}", flush=True)
            alive = still
        total["w"] += w
        total["l"] += l
        n = w + l
        waves.append({"wave": wave, "games": n, "w": w, "l": l,
                      "score_a": w / n if n else 0.0})
        print(f"SUMMARY wave={wave} games={n} w={w} l={l} "
              f"score_a={w / n if n else 0.0:.3f} "
              f"elapsed={time.time() - t_wave:.0f}s", flush=True)
    n = total["w"] + total["l"]
    score_a = total["w"] / n if n else 0.0
    print(f"TOTAL games={n} w={total['w']} l={total['l']} "
          f"score_a={score_a:.3f} elapsed={time.time() - t0:.0f}s")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "games": n, "w": total["w"], "l": total["l"], "score_a": score_a,
            "waves": waves, "ckpt_a": args.ckpt_a, "ckpt_b": args.ckpt_b,
            "a_sims": args.a_sims, "b_sims": args.b_sims, "seed": seed,
            "elapsed_s": round(time.time() - t0, 1),
        }, indent=2))
    eng_a.quit()
    eng_b.quit()


if __name__ == "__main__":
    main()
