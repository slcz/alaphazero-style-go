#!/usr/bin/env python3
"""kata_label.py — fill score/ownership labels for the games selfplay paused.

Selfplay stops a game once its result is certain (|value| >= 0.95) and writes
the record with end_kind = SP_END_PAUSED and neutral labels. This pass asks
KataGo about that final position and patches the record in place:

    score      black-view margin  <- sign-flipped KataGo scoreLead
    ownership  black-view [0,1]   <- KataGo's per-point ownership

Both are reported by KataGo from the side to move, so they are flipped when
White is to move. Games that ended on the board (double pass or the ply
wall) keep their exact TT labels and are not touched.

    python3 scripts/kata_label.py t/shard-0.spbin [--out FILE] [--visits N]
        [--katago-cmd "katago gtp -model ... -config ..."] [--games N]

KATA_CMD overrides the KataGo command line.
"""
from __future__ import annotations

import argparse
import os
import queue
import shlex
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from go_ai.engine import BLACK, PASS, WHITE, Board  # noqa: E402
from go_ai.selfplay.shard import SP_END_PAUSED, scan_shard  # noqa: E402

_GAME_HDR = struct.Struct("<IIBBHff")     # idx, plies, b, end, sims, komi, score
SCORE_OFF = 16                           # score field inside the record
OWN_OFF = _GAME_HDR.size                 # ownership right after the header


def gtp_vertex(move: int, b: int) -> str:
    if move == PASS:
        return "pass"
    cols = "ABCDEFGHJKLMNOPQRST"
    return f"{cols[move % b]}{move // b + 1}"


class KataGo:
    """One long-lived GTP session; one position per call."""

    def __init__(self, cmd: str):
        self.p = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True,
                                  bufsize=1)
        # a reader thread: select() must not be mixed with a buffered pipe
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        if self.cmd("protocol_version") is None:
            raise SystemExit("katago did not answer")
        # our rules are Tromp-Taylor: area scoring, positional superko and
        # legal suicide — KataGo must replay the history under the same
        # ruleset or it rejects moves we allow
        self.cmd("kata-set-rules tromp-taylor")

    def _reader(self):
        for line in self.p.stdout:
            self.q.put(line)
        self.q.put(None)                     # EOF

    def _line(self, timeout: float):
        """-> line, "" on timeout, None on EOF."""
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return ""

    def cmd(self, line: str, timeout: float = 30.0):
        if self.p.poll() is not None:
            return None
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = []
        deadline = time.time() + timeout
        while True:
            l = self._line(max(0.1, deadline - time.time()))
            if l is None or l == "":
                return None
            if l.strip() == "":
                return "".join(out).strip()
            out.append(l)

    def analyse(self, moves, b: int, komi: float, visits: int):
        """-> (black-view margin, black-view ownership[0,1]) or None.

        kata-raw-nn is one response per position with no streaming and a
        fixed perspective (White), so no per-turn sign juggling."""
        self.cmd(f"komi {komi}", timeout=30)
        self.cmd(f"boardsize {b}", timeout=30)
        self.cmd("clear_board", timeout=30)
        # one play per move: set_position chokes on passes and on the
        # suicide moves our rules allow
        for i, m in enumerate(moves):
            colour = "b" if i % 2 == 0 else "w"
            r = self.cmd(f"play {colour} {gtp_vertex(m, b)}", timeout=30)
            if r is None or r.startswith("?"):
                print(f"    play #{i} rejected: {repr(r)[:100]}")
                return None
        # the first call pays for backend init (CUDA/cuDNN), so be patient
        r = self.cmd("kata-raw-nn 0", timeout=180)
        if r is None or r.startswith("?"):
            print(f"    raw-nn failed: {repr(r)[:120]}")
            return None
        # KataGo may wrap the key/value stream anywhere, so tokenise the
        # whole response instead of reading lines
        toks = r.split()

        def nums_after(key, n):
            try:
                i = toks.index(key)
                return [float(x) for x in toks[i + 1:i + 1 + n]]
            except (ValueError, IndexError):
                return None

        lead_v, own = nums_after("whiteLead", 1), nums_after("whiteOwnership",
                                                            b * b)
        lead = lead_v[0] if lead_v else None
        if lead is None or own is None or len(own) != b * b:
            print(f"    raw-nn parse failed: lead={lead} "
                  f"own={None if own is None else len(own)} "
                  f"b2={b * b} first={r.splitlines()[:2]}")
            return None
        return -lead, [(1.0 - v) / 2.0 for v in own]

    def close(self):
        try:
            self.cmd("quit")
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def label(shard: Path, out: Path, cmd: str, visits: int, limit: int) -> int:
    data = bytearray(shard.read_bytes())
    n_paused = sum(1 for g, _s, _e in scan_shard(shard)
                   if g.end_kind == SP_END_PAUSED)
    if n_paused == 0:
        print(f"{shard.name}: no paused games, nothing to label")
        if out != shard:
            out.write_bytes(bytes(data))
        return 0
    kat = KataGo(cmd)
    patched = seen = 0
    t0 = time.time()
    for g, start, _end in scan_shard(shard):
        if g.end_kind != SP_END_PAUSED:
            continue
        seen += 1
        if limit and seen > limit:
            break
        moves = [p.move for p in g.plies]
        r = kat.analyse(moves, g.b, g.komi, visits)
        if r is None:
            print(f"  game {g.game_index}: katago failed, left as-is")
            continue
        score_black, own_black = r
        struct.pack_into("<f", data, start + SCORE_OFF, score_black)
        struct.pack_into(f"<{g.b * g.b}e", data, start + OWN_OFF,
                         *[min(1.0, max(0.0, o)) for o in own_black])
        patched += 1
        if patched % 200 == 0:
            print(f"  {patched} labelled ({time.time() - t0:.0f}s)", flush=True)
    kat.close()
    out.write_bytes(data)
    dt = time.time() - t0
    print(f"{shard.name}: {seen} paused games, {patched} labelled in {dt:.0f}s"
          f" -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("--out", default=None, help="default: patch in place")
    ap.add_argument("--visits", type=int, default=200)
    ap.add_argument("--games", type=int, default=0, help="0 = all")
    ap.add_argument("--katago-cmd", default=os.environ.get("KATA_CMD"))
    args = ap.parse_args()
    shard = Path(args.shard)
    out = Path(args.out) if args.out else shard
    if not args.katago_cmd:
        print("no KataGo command (--katago-cmd or KATA_CMD)", file=sys.stderr)
        return 2
    if out == shard:                      # patch a copy, then swap
        tmp = Path(tempfile.mkstemp(dir=str(shard.parent))[1])
        rc = label(shard, tmp, args.katago_cmd, args.visits, args.games)
        if rc == 0:
            tmp.replace(shard)
        else:
            tmp.unlink(missing_ok=True)
        return rc
    return label(shard, out, args.katago_cmd, args.visits, args.games)


if __name__ == "__main__":
    sys.exit(main())
