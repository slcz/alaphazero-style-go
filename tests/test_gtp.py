"""test_gtp.py — the GTP engine driven as a subprocess (dummy device:
no net needed; uniform priors, balanced value)."""
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GTP = REPO / "engine" / "build" / "gtp"


class Gtp:
    def __init__(self, simss=8):
        self.p = subprocess.Popen(
            [str(GTP), "dummy", "64"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, env={"GO_SIMS": str(simss)},
            cwd=str(REPO))

    def cmd(self, line, expect_ok=True):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = []
        while True:
            l = self.p.stdout.readline()
            out.append(l)
            if l == "\n":
                break
        body = "".join(out[:-1])
        assert body.startswith("=") == expect_ok, body
        return body[1:].strip()

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
        g.cmd("kata-analyze", expect_ok=False)
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
