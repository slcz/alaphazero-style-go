"""test_nn_worker.py — the worker's synchronous (CPU) protocol against a
real checkpoint: host buffers by pointer, three-phase calls, masking
contract on the delivered logp, softmaxed value rows."""
import numpy as np
import pytest
import torch

import go_ai.net.nn_worker as W
from go_ai.engine import Board
from go_ai.net import features as F
from go_ai.net.model import GoNet, NetConfig
from go_ai.train.trainer import TrainConfig, Trainer

TINY = NetConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64, rope_pairs=2)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    torch.manual_seed(3)
    tr = Trainer(GoNet(TINY), TrainConfig(device="cpu", compile=False))
    p = tmp_path_factory.mktemp("nnw") / "tiny.pt"
    tr.save(p)
    return str(p)


def test_cpu_protocol(ckpt):
    b = 7
    out = W.setup(ckpt, "cpu", rows=8, banks=2)
    assert len(out["planes"]) == 2
    sets = W._S["sets"]

    # stage two rows by writing the host tensors (the C side would fill
    # them via nn_encode + the mask builder)
    boards = [Board(b, komi=9.5, maxply=150), Board(b, komi=9.5, maxply=150)]
    boards[0].push(24)
    boards[1].push(3)
    boards[1].push(11)
    n = 2
    for i, bd in enumerate(boards):
        sets[0]["x"][i, :b * b] = torch.from_numpy(bd.encode())
        sets[0]["m"][i] = torch.from_numpy(F.policy_mask(bd.legal_moves, b))

    W.push(0, n)
    W.pull(0, n)
    W.wait_ready(0, n)

    for i, bd in enumerate(boards):
        lp = sets[0]["lp"][i]
        legal = sets[0]["m"][i]
        assert torch.isfinite(lp[:b * b + 1][legal[:b * b + 1]]).all()
        # illegal active slots: prob exactly 0
        act = ~legal[:b * b + 1]
        if act.any():
            assert (lp[:b * b + 1][act] == torch.finfo(torch.float32).min).all()
        assert float(lp[:b * b + 1].exp().sum()) == pytest.approx(1.0, abs=1e-5)
        assert (lp[b * b + 1:] == 0).all()          # stride tail untouched
        w = sets[0]["w"][i]
        assert w.shape == (2,) and float(w.sum()) == pytest.approx(1.0)
        assert 0.0 <= float(w[0]) <= 1.0
