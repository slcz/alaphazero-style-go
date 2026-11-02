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

    def stream(self, line, lines):
        """Send an analysis command, read `lines` report lines, interrupt
        with a raw newline (KataGo's cancel signal), drain the response."""
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = []
        for _ in range(lines):
            l = self.p.stdout.readline()
            if l == "":
                raise AssertionError("engine died: " + "".join(out))
            out.append(l)
        self.p.stdin.write("\n")
        self.p.stdin.flush()
        while True:
            l = self.p.stdout.readline()
            if l == "":
                raise AssertionError("engine died in stream")
            out.append(l)
            if l == "\n":
                break
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
        assert g.cmd("known_command kata-analyze") == "true"
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
        for c in ("kata-analyze", "analyze", "genmove_analyze",
                  "showownership"):
            assert g.cmd(f"known_command {c}") == "true", c
    finally:
        g.close()


def test_analysis_with_a_net():
    ck = os.environ.get("GO_TEST_CKPT")
    if not ck or not os.path.exists(ck):
        pytest.skip("set GO_TEST_CKPT to a checkpoint")
    g = Gtp(simss=8, device="cpu", ckpt=ck)
    try:
        g.cmd("boardsize 9")
        g.cmd("clear_board")
        g.cmd("play b E5")
        # ownership dump: b map rows, then the numbers
        rows = [l for l in g.cmd("showownership").splitlines() if l.strip()]
        assert len(rows) == 2 * 9 + 1, rows      # map, legend, numbers
        assert all(len(r.strip()) == 9 for r in rows[:9])
        # kata-analyze streams report blocks until interrupted
        a = g.stream("kata-analyze interval 1", 12)
        assert "info move" in a and "rootInfo visits" in a
        assert a.count("ownership") >= 1
        vals = a.split("ownership")[-1].split()
        assert len(vals) == 81
        assert all(-1.0 <= float(v) <= 1.0 for v in vals)
        wr = float(a.split("winrate")[1].split()[0])
        assert 0.0 <= wr <= 1.0
        # the engine is still healthy and in sync afterwards
        assert g.cmd("name") == "gogo"
    finally:
        g.close()


def test_analyze_sabaki_dialect():
    """Sabaki's spec (docs/guides/engine-analysis-integration.md): one
    line per update, space-separated `info` groups, positional color and
    interval, floating-point winrate, optional scoreLead, no ownership."""
    ck = os.environ.get("GO_TEST_CKPT")
    if not ck or not os.path.exists(ck):
        pytest.skip("set GO_TEST_CKPT to a checkpoint")
    g = Gtp(simss=8, device="cpu", ckpt=ck)
    try:
        g.cmd("boardsize 9")
        g.cmd("clear_board")
        g.cmd("play b E5")
        txt = g.stream("analyze b 1", 2)
        first = txt.splitlines()[0].lstrip("= ")
        assert first.startswith("info move"), first
        for key in ("visits", "winrate", "pv"):
            assert f" {key} " in first, key
        assert "ownership" not in txt
        # every update is one line carrying whole info groups
        updates = [l for l in txt.splitlines() if l.lstrip("= ").strip()]
        assert len(updates) >= 2, updates
        assert all(l.lstrip("= ").startswith("info move") for l in updates)
        assert g.cmd("name") == "gogo"          # still in sync
    finally:
        g.close()


def test_genmove_analyze_plays_a_move():
    ck = os.environ.get("GO_TEST_CKPT")
    if not ck or not os.path.exists(ck):
        pytest.skip("set GO_TEST_CKPT to a checkpoint")
    g = Gtp(simss=8, device="cpu", ckpt=ck)
    try:
        g.cmd("boardsize 9")
        g.cmd("clear_board")
        g.cmd("play b E5")
        txt = g.stream("genmove_analyze w 1", 2)
        last = [l for l in txt.splitlines() if l.strip()][-1]
        assert last.startswith("play "), last      # white's reply
        assert g.cmd("name") == "gogo"
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
