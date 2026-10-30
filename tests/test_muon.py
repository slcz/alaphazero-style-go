"""test_muon.py — native torch.optim.Muon (torch >= 2.9) split + fit."""
import math

import pytest
import torch

from go_ai.net.model import GoNet, NetConfig
from go_ai.train.trainer import TrainConfig, Trainer

pytestmark = pytest.mark.skipif(
    not hasattr(torch.optim, "Muon"), reason="torch < 2.9")


def test_muon_minimizes_least_squares():
    torch.manual_seed(0)
    A = torch.randn(256, 32)
    target = torch.randn(256, 16)
    w = torch.nn.Parameter(torch.zeros(32, 16))
    opt = torch.optim.Muon([w], lr=0.02, adjust_lr_fn="original")
    first = None
    for _ in range(30):
        loss = ((A @ w - target) ** 2).mean()
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_trainer_muon_split():
    torch.manual_seed(0)
    cfg = TrainConfig(device="cpu", compile=False, batch_size=8,
                      optimizer="muon", lr=0.01, own_weight=0.0,
                      score_pdf_weight=0.0, score_cdf_weight=0.0)
    tr = Trainer(GoNet(NetConfig(d_model=64, n_layers=2, n_heads=4,
                                 d_ff=128, rope_pairs=2)), cfg)
    n_mu = sum(len(g["params"]) for g in tr.opt.param_groups)
    n_ad = sum(len(g["params"]) for g in tr.opt_adam.param_groups)
    assert n_mu > 0 and n_ad > 0
    b, T = 8, 49
    tok = torch.randn(b, T, 18)
    mask = torch.ones(b, T + 1, dtype=torch.bool)
    batch = {"tokens": tok, "mask": mask,
             "pi_slots": torch.randint(0, T + 1, (b, 8)),
             "pi_probs": torch.rand(b, 8),
             "wdl": torch.ones(b, 2) * 0.5,
             "aux_idx": torch.zeros(b, dtype=torch.int64),
             "aux_mask": mask, "aux_w": torch.ones(b)}
    m = tr.train_step(batch)
    assert math.isfinite(m["loss"])
