"""The row cache must reproduce the reference per-sample path exactly:
same tokens, superko mask, π̄, ownership, labels, and the same D4 mapping.
The reference is `selfplay_sample`/`transform_sample` (cffi replay + the
C encoder), i.e. what the trainer used before the cache existed.

    GO_TEST_SHARD=it002/selfplay/shard-0.spbin python3 -m pytest \
        tests/test_rowcache.py
"""
import os
from itertools import islice

import numpy as np
import pytest

pytest.importorskip("torch")

from go_ai.net import features as F  # noqa: E402
from go_ai.selfplay.shard import scan_shard  # noqa: E402
from go_ai.train import rowcache  # noqa: E402
from go_ai.train.selfplay_data import selfplay_sample, trainable_plies  # noqa: E402
from go_ai.train.symmetry import transform_sample  # noqa: E402

SHARD = os.environ.get("GO_TEST_SHARD")
GAMES = 3


def _games():
    if not SHARD or not os.path.exists(SHARD):
        pytest.skip("set GO_TEST_SHARD to a selfplay shard")
    return [g for g, _s, _e in islice(scan_shard(SHARD), GAMES)]


def _check(out, ref, mask_key="mask"):
    for k in ("tokens", mask_key, "own", "score", "wdl", "aux_idx",
              "aux_mask", "aux_w"):
        a, e = out[k][0], np.asarray(ref[k])
        if k == "aux_w":
            assert a == pytest.approx(float(e), abs=1e-6), k
        elif k in ("own", "tokens", "mask", "aux_mask"):
            assert np.array_equal(a, e), k
        else:
            assert a == e, k
    # π̄: dense over the policy head's slots, renormalized
    want = F.pi_dense(ref["pi_slots"], ref["pi_probs"],
                      int(ref["mask"].sum()), out["pi"].shape[1] - 1)
    assert np.allclose(out["pi"][0], want, atol=1e-3), "pi"


@pytest.mark.parametrize("game_i", range(GAMES))
def test_identity_matches_reference(game_i):
    g = _games()[game_i]
    cache = rowcache._game_rows(g)
    idx = trainable_plies(g)
    assert len(idx) == len(cache["sb"])
    assert np.all(cache["game"] == g.game_index)
    for j, i in enumerate(idx):
        ref = selfplay_sample(g, i)
        out = rowcache.alloc_batch(1, g.b)
        assert rowcache.fill_batch([cache], [np.array([j])], 0,
                                   out, g.b, g.komi) == 1
        _check(out, ref)


@pytest.mark.parametrize("game_i", range(GAMES))
def test_symmetry_matches_reference(game_i):
    g = _games()[game_i]
    cache = rowcache._game_rows(g)
    idx = trainable_plies(g)
    for j, i in enumerate(idx[:40]):
        ref = selfplay_sample(g, i)
        for s in range(1, 8):
            want = transform_sample(ref, g.b, s)
            out = rowcache.alloc_batch(1, g.b)
            rowcache.fill_batch([cache], [np.array([j])], s,
                                out, g.b, g.komi)
            _check(out, want)


def test_cache_round_trip(tmp_path):
    """load_or_build caches to disk and reloads identical rows."""
    if not SHARD or not os.path.exists(SHARD):
        pytest.skip("set GO_TEST_SHARD to a selfplay shard")
    g, _s, end = next(iter(scan_shard(SHARD)))
    unit = tmp_path / "selfplay"
    unit.mkdir()
    raw = open(SHARD, "rb").read(end)          # header + one game record
    (unit / "shard-0.spbin").write_bytes(raw)
    quiet = lambda *_: None                    # noqa: E731
    a = rowcache.load_or_build(unit, log=quiet)
    b = rowcache.load_or_build(unit, log=quiet)   # now from the npz
    assert int(b["b"]) == g.b and len(a["sb"]) == len(b["sb"])
    for k in rowcache.KEYS:
        assert np.array_equal(a[k], b[k]), k


def test_select_keeps_only_named_games():
    g = _games()[0]
    cache = rowcache._game_rows(g)
    n = len(cache["sb"])
    if n == 0:
        pytest.skip("game has no slow-cap ply")
    assert len(rowcache.select(cache, None)) == n
    assert len(rowcache.select(cache, [g.game_index])) == n
    assert len(rowcache.select(cache, [g.game_index + 999])) == 0


def test_draw_is_uniform():
    rng = np.random.default_rng(0)
    sels = [np.arange(10_000), np.arange(20_000)]
    picks = rowcache.draw(30_000, sels, rng)
    got = np.array([len(p) for p in picks], float) / 30_000
    assert np.allclose(got, (1 / 3, 2 / 3), atol=0.02), got
    for p, s in zip(picks, sels):
        assert p.min() >= s[0] and p.max() <= s[-1]
    assert sum(len(p) for p in picks) == 30_000
