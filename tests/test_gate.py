"""test_gate.py — the two-engine gate end to end with dummy evaluators:
protocol, waves, referee legality, resign path, JSON output."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_gate_match(tmp_path):
    out = tmp_path / "eval.json"
    r = subprocess.run(
        [sys.executable, "scripts/batch_match.sh",
         "--ckpt-a", "dummy-a", "--ckpt-b", "dummy-b",
         "--games", "1", "--a-sims", "4", "--b-sims", "4",
         "--b", "7", "--komi", "9.5", "--max-plys", "40",
         "--device", "dummy", "--rows", "64",
         "--resign", "0.95", "--resign-ply", "6",
         "--openings", "2", "--seed", "7", "--json", str(out)],
        capture_output=True, text=True, cwd=str(REPO), timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    data = json.loads(out.read_text())
    assert data["games"] == 2                     # 1 per wave, 2 waves
    assert data["w"] + data["l"] == data["games"]
    assert 0.0 <= data["score_a"] <= 1.0
    assert set(data["waves"][0]) >= {"wave", "games", "w", "l", "score_a"}
    # per-game referee lines: one GAME per finished game
    games = [l for l in r.stdout.splitlines() if l.startswith("GAME ")]
    assert len(games) == 2
