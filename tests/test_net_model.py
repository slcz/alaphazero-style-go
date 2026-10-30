"""test_net_model.py — GoNet: trunk/heads shapes, the masking contract,
board-size adaptivity, score bins, checkpoint round trip."""
import math

import torch.nn as nn

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


def test_fp8_linear_matches_reference():
    from go_ai.net.model import FP8Linear
    torch.manual_seed(0)
    lin = nn.Linear(64, 32)
    fp8 = FP8Linear(lin)
    x = torch.randn(16, 64)
    y, ref = fp8(x), lin(x)
    rel = (y - ref).norm() / ref.norm()
    assert rel < 0.05, f"fp8 rel err {rel:.3f}"


def test_fp8_gonet_roundtrip(tmp_path):
    from go_ai.net.model import GoNet
    cfg = NetConfig(**{**TINY.__dict__, "fp8": True})
    net = GoNet(cfg)
    assert isinstance(net.blocks[0].attn.qkv.wq,
                      torch.Tensor)
    payload = {"model": net.state_dict(),
               "net_config": dict(cfg.__dict__)}
    p = tmp_path / "fp8.pt"
    torch.save(payload, p)
    net2 = GoNet(NetConfig(**payload["net_config"]))
    net2.load_state_dict(payload["model"])
    tok = torch.randn(2, 49, TINY.token_dim)
    mask = torch.ones(2, 50, dtype=torch.bool)
    out = net2(tok, mask, mask)
    assert out["policy_logp"].shape == (2, 50)


def test_legacy_conv_ffn_ckpt_loads(tmp_path):
    """Pre-MLP checkpoints stored ff up/down as [out,in,1,1]; the load
    shim squeezes them into the Linear weights (outputs identical)."""
    from go_ai.train.trainer import Trainer, TrainConfig
    torch.manual_seed(3)
    tr = Trainer(GoNet(TINY), TrainConfig(device="cpu", compile=False))
    tr.save(tmp_path / "a.pt")
    payload = torch.load(tmp_path / "a.pt", weights_only=False)
    for k in ("ff.up.weight", "ff.down.weight"):
        for i in range(TINY.n_layers):
            payload["model"][f"blocks.{i}.{k}"] = \
                payload["model"][f"blocks.{i}.{k}"].unsqueeze(-1).unsqueeze(-1)
    torch.save(payload, tmp_path / "legacy.pt")

    tok = torch.randn(2, 49, TINY.token_dim)
    mask = torch.ones(2, 50, dtype=torch.bool)
    a = Trainer.load(tmp_path / "a.pt", device="cpu").model(tok, mask, mask)
    b = Trainer.load(tmp_path / "legacy.pt", device="cpu").model(tok, mask, mask)
    assert (a["policy_logp"] - b["policy_logp"]).abs().max() < 1e-5
