"""conftest.py — the shared dummy-selfplay shard fixture (runs the real
C driver with the stub evaluator once per test session)."""
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SELFPLAY = REPO / "engine" / "build" / "selfplay"

# small but non-trivial: PCR on, 4 games, 40-ply cap
ARGS = ["none", "{dir}", "4", "7", "9.5", "8", "4", "16", "6", "40", "1",
        "dummy", "7", "4", "0.5"]


@pytest.fixture(scope="session")
def sp_dir(tmp_path_factory):
    if not SELFPLAY.exists():
        pytest.fail(f"{SELFPLAY} not built — run make -C engine")
    d = tmp_path_factory.mktemp("sp")
    subprocess.run([str(SELFPLAY)] + [a.format(dir=d) for a in ARGS],
                   check=True, capture_output=True, cwd=REPO)
    return d
