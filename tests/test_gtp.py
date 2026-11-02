"""test_gtp.py — the GTP engine driven as a subprocess (dummy device:
no net needed; uniform priors, balanced value)."""
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GTP = REPO / "engine" / "build" / "gtp"


class Gtp:
    def __init__(self, simss=8, device="dummy", ckpt=None, extra_env=None):
        env = {"GO_SIMS": str(simss)}
        if ckpt:
            env["PUCT_CKPT"] = ckpt
            # keep an analyze session open long enough to exercise the
            # abort path (it otherwise stops at its own playout budget)
            env["GO_ANALYZE_SIMS"] = "100000"
        env.update(extra_env or {})
        self.p = subprocess.Popen(
            [str(GTP), device, "64"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, env=env, cwd=str(REPO))

    def cmd(self, line, expect_ok=True):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = []
        while True:
            l = self.p.stdout.readline()
            if l == "":
                raise AssertionError("engine died: " + "".join(out))
            out.append(l)
            if l == "\n":
                break
        body = "".join(out[:-1])
        assert body.startswith("=") == expect_ok, body
        return body[1:].strip()

    def stream(self, line, lines, abort="protocol_version"):
        """Analysis streams: `=` alone, `info ...` lines, blank terminator
        only after the abort command (which Sabaki itself uses)."""
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        head = self.p.stdout.readline()
        assert head.rstrip() == "=", repr(head)
        out, closed = [], False
        while len(out) < lines:
            l = self.p.stdout.readline()
            if l == "":
                raise AssertionError("engine died: " + "".join(out))
            if l == "\n":                 # session hit its playout budget
                closed = True
                break
            out.append(l)
        if not closed:                     # still streaming: abort like Sabaki
            self.p.stdin.write(abort + "\n")
            self.p.stdin.flush()
            while True:
                l = self.p.stdout.readline()
                if l == "":
                    raise AssertionError("engine died in stream")
                if l == "\n":
                    break
                out.append(l)
            assert self.p.stdout.readline().strip().startswith("=2")
            assert self.p.stdout.readline() == "\n"  # its own terminator
        return "".join(out)

    def close(self):
        self.p.stdin.write("quit\n")
        self.p.stdin.flush()
        self.p.wait(timeout=10)


def test_required_subset():
    g = Gtp()
    try:
        assert g.cmd("protocol_version") == "2"
        assert g.cmd("name") == "gogo"
        assert g.cmd("version") == "0.1"
        assert g.cmd("known_command genmove") == "true"
        assert g.cmd("known_command kata-analyze") == "false"
        listing = g.cmd("list_commands")
        for c in ("protocol_version", "boardsize", "clear_board", "komi",
                  "play", "genmove", "quit"):
            assert c in listing
        assert g.cmd("boardsize 7") == ""
        assert g.cmd("komi 9.5") == ""
        assert g.cmd("clear_board") == ""
    finally:
        g.close()


def test_play_genmove_game():
    g = Gtp()
    try:
        g.cmd("boardsize 7")
        g.cmd("komi 9.5")
        assert g.cmd("play black D4") == ""      # GTP vertex accepted
        assert g.cmd("play E3") == ""            # white's reply by hand
        mv = g.cmd("genmove")                    # black again
        assert mv and (mv == "pass" or mv[0] in "ABCDEFGH")
        # id-prefixed commands (the id echoes in the response)
        assert g.cmd("17 name") == "17 gogo"
        # illegal play errors
        g.cmd("play D4", expect_ok=False)        # occupied
        g.cmd("play I5", expect_ok=False)        # no I column
        # unknown command errors
        g.cmd("kata-raw-nn 0", expect_ok=False)
    finally:
        g.close()


def test_analysis_commands_are_advertised():
    """Sabaki probes these names (engines.analyze_commands) before it can
    show the winrate bar / ownership heatmap."""
    g = Gtp()
    try:
        for c in ("analyze", "showstate"):
            assert g.cmd(f"known_command {c}") == "true", c
        for c in ("kata-analyze", "genmove_analyze", "lz-analyze"):
            assert g.cmd(f"known_command {c}") == "false", c
    finally:
        g.close()


def test_analyze_and_ownership():
    """`analyze` streams Sabaki's format: '=' alone, then one `info` line
    per update with an INTEGER winrate in ten-thousandths; the response
    stays open until the GUI sends something. `showstate` is our own
    text report of the score head and the ownership map."""
    ck = os.environ.get("GO_TEST_CKPT")
    if not ck or not os.path.exists(ck):
        pytest.skip("set GO_TEST_CKPT to a checkpoint")
    g = Gtp(simss=8, device="cpu", ckpt=ck)
    try:
        g.cmd("boardsize 9")
        g.cmd("clear_board")
        g.cmd("play b E5")
        txt = g.stream("analyze b 1", 3)
        frames = [l for l in txt.splitlines() if l.strip()]
        assert len(frames) >= 2, frames            # it streams, not one-shot
        for line in frames:
            assert line.startswith("info move"), line
            assert "pv" in line
            wr = int(line.split("winrate")[1].split()[0])
            assert 0 <= wr <= 10000, wr            # ten-thousandths
        # state dump: score line, then a 9x9 map with aligned columns
        txt = g.cmd("showstate")
        assert "score" in txt and "side to move:" in txt
        grid = [l for l in txt.splitlines() if l.strip()[:1].isdigit()]
        assert len(grid) == 9, grid
        for i, row in enumerate(grid):
            assert row.strip().split()[0] == str(9 - i)
            cells = row.strip().split()[1:]
            assert len(cells) == 9 and set(cells) <= {".", "X", "O"}, row
        assert "X = black" in txt
        assert g.cmd("name") == "gogo"             # in sync after the abort
    finally:
        g.close()


def test_short_color_forms():
    """GTP colors are "b"/"w" — Sabaki sends those, not "black"/"white"."""
    g = Gtp()
    try:
        g.cmd("boardsize 7")
        g.cmd("play b D4")
        g.cmd("play w E3")
        assert g.cmd("genmove b")             # short, upper/lower both
        assert g.cmd("genmove W")
        g.cmd("genmove w", expect_ok=False)   # out of turn is still an error
        # a vertex starting with a color letter must stay a vertex
        g.cmd("play B6")
    finally:
        g.close()


def test_genmove_color_mismatch():
    g = Gtp()
    try:
        g.cmd("boardsize 7")
        g.cmd("genmove white", expect_ok=False)  # black to move
        g.cmd("genmove black")
    finally:
        g.close()
