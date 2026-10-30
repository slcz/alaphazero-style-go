"""test_net_model.py — GoNet: trunk/heads shapes, the masking contract,
board-size adaptivity, score bins, checkpoint round trip."""
import math

import pytest
import torch

from go_ai.net import features as F
from go_ai.net.model import (SCORE_BINS, GoNet, NetConfig, count_parameters,
                             score_to_bin)
from go_ai.train.trainer import TrainConfig, Trainer

TINY = NetConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64, rope_pairs=2)


def _tokens(b, batch=3):
    """Random tokens with the right active size."""
    return torch.randn(batch, b * b, TINY.token_dim)


def _mask(b, batch=3):
    m = torch.zeros(batch, F.POLICY_MAX, dtype=torch.bool)
    m[:, :b * b + 1] = True
    m[:, torch.randint(0, b * b, (batch,))] = False   # some illegal slots
    return m


def test_shapes_and_contract():
    torch.manual_seed(0)
    net = GoNet(TINY).eval()
    for b in (5, 7, 9):                 # same weights, three board sizes
        x, m = _tokens(b), _mask(b)
        with torch.no_grad():
            out = net(x, m)
        T = b * b
        assert out["policy_logp"].shape == (3, T + 1)
        assert out["aux_logp"].shape == (3, T + 1)
        assert out["wdl"].shape == (3, 2)
        assert out["own"].shape == (3, T)
        assert out["score"].shape == (3, SCORE_BINS)
        # the masking contract: illegal slots have probability EXACTLY 0
        lp = out["policy_logp"]
        legal = m[:, :T + 1]
        assert torch.isfinite(lp[legal]).all()
        assert (lp[~legal] == torch.finfo(lp.dtype).min).all()
        p = lp.exp()
        assert (p[~legal] == 0).all()
        assert torch.allclose(p.sum(-1), torch.ones(3), atol=1e-5)
        # value = pW ∈ [0,1]; illegal slots never carry mass
        assert (out["value"] >= 0).all() and (out["value"] <= 1).all()


def test_aux_mask_independent():
    torch.manual_seed(1)
    net = GoNet(TINY).eval()
    b = 5
    x, m = _tokens(b), _mask(b)
    am = _mask(b)
    with torch.no_grad():
        out = net(x, m, am)
    legal = am[:, :b * b + 1]
    assert torch.isfinite(out["aux_logp"][legal]).all()
    assert (out["aux_logp"][~legal] == torch.finfo(torch.float32).min).all()


def test_default_scale():
    net = GoNet(NetConfig())            # the training trunk
    n = count_parameters(net)
    assert 5_000_000 < n < 8_000_000, n


def test_score_bins():
    assert score_to_bin(-380.0) == 0
    assert score_to_bin(0.0) == 380
    assert score_to_bin(380.0) == 760
    assert score_to_bin(-9999) == 0          # clamped at the rails
    assert score_to_bin(9999) == SCORE_BINS - 1
    assert score_to_bin(7.4) == score_to_bin(7.0)   # 1-point bins round


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(2)
    net = GoNet(TINY)
    tr = Trainer(net, TrainConfig(device="cpu", compile=False))
    ck = tmp_path / "tiny.pt"
    tr.save(ck)
    tr2 = Trainer.load(ck, device="cpu")
    b = 7
    x, m = _tokens(b), _mask(b)
    net.eval()
    tr2.model.eval()
    with torch.no_grad():
        o1 = net(x, m)
        o2 = tr2.model(x, m)
    assert torch.allclose(o1["policy_logp"], o2["policy_logp"], atol=1e-6)
    assert torch.allclose(o1["own"], o2["own"], atol=1e-6)
    assert tr2.steps == tr.steps
