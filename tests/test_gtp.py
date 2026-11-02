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
            env["GO_ANALYZE_SIMS"] = str(simss)
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
        for c in ("analyze", "showownership"):
            assert g.cmd(f"known_command {c}") == "true", c
        for c in ("kata-analyze", "genmove_analyze", "lz-analyze"):
            assert g.cmd(f"known_command {c}") == "false", c
    finally:
        g.close()


def test_analyze_and_ownership():
    """`analyze` answers once in Sabaki's format; showownership dumps the
    net's ownership head as a map plus the numbers."""
    ck = os.environ.get("GO_TEST_CKPT")
    if not ck or not os.path.exists(ck):
        pytest.skip("set GO_TEST_CKPT to a checkpoint")
    g = Gtp(simss=8, device="cpu", ckpt=ck)
    try:
        g.cmd("boardsize 9")
        g.cmd("clear_board")
        g.cmd("play b E5")
        a = g.cmd("analyze b 10")
        line = a.strip()
        assert line.startswith("info move"), line
        for key in ("visits", "winrate", "prior", "order", "pv", "scoreLead"):
            assert f" {key} " in line, key
        assert line.count("info move") >= 2          # whole info groups
        assert "ownership" not in line               # not Sabaki's format
        wr = float(line.split("winrate")[1].split()[0])
        assert 0.0 <= wr <= 1.0
        # ownership dump: 9 map rows, a legend, then the numbers
        rows = [l for l in g.cmd("showownership").splitlines() if l.strip()]
        assert len(rows) == 2 * 9 + 1, rows
        assert all(len(r.strip()) == 9 for r in rows[:9])
        assert g.cmd("name") == "gogo"               # still in sync
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
