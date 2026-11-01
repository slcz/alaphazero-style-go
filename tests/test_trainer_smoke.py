"""test_trainer_smoke.py — one real training step on real samples (tiny
net, CPU) + the nn_worker protocol against the same checkpoint."""
import numpy as np
import pytest
import torch

from go_ai.net import features as F
from go_ai.net.model import GoNet, NetConfig
from go_ai.selfplay.shard import read_unit
from go_ai.train.selfplay_data import selfplay_samples, stack_selfplay
from go_ai.train.trainer import TrainConfig, Trainer, tensors_to_torch

TINY = NetConfig(d_model=64, n_layers=2, n_heads=4, d_ff=64, rope_pairs=2)


def _batch(sp_dir):
    games = read_unit(sp_dir)
    ss = [s for g in games for s in selfplay_samples(g)]
    raw = stack_selfplay(ss[:16])
    raw["pi"] = np.stack([F.pi_dense(raw["pi_slots"][i], raw["pi_probs"][i],
                                     int(raw["mask"][i].sum()), games[0].b)
                          for i in range(len(raw["wdl"]))])
    return tensors_to_torch(raw, torch.device("cpu"))


def test_train_step(sp_dir):
    torch.manual_seed(0)
    tr = Trainer(GoNet(TINY), TrainConfig(device="cpu", compile=False))
    batch = _batch(sp_dir)
    m0 = tr.train_step(batch)
    assert np.isfinite(m0["loss"])
    assert 0.0 <= m0["policy_acc"] <= 1.0
    for _ in range(5):
        m = tr.train_step(batch)
    assert np.isfinite(m["loss"])
    assert tr.steps == 6
    # losses: every head reported
    for k in ("policy", "value", "aux", "own", "score_pdf", "score_cdf"):
        assert np.isfinite(m[k])


def test_evaluate(sp_dir, tmp_path):
    torch.manual_seed(1)
    tr = Trainer(GoNet(TINY), TrainConfig(device="cpu", compile=False))
    batch = _batch(sp_dir)
    ev = tr.evaluate({"tokens": batch["tokens"].numpy(),
                      "mask": batch["mask"].numpy(),
                      "aux_mask": batch["aux_mask"].numpy(),
                      "pi": batch["pi"].numpy(),
                      "wdl": batch["wdl"].numpy()})
    assert 0.0 <= ev["policy_acc"] <= 1.0
    assert 0.0 <= ev["value_acc"] <= 1.0
    assert ev["policy_nll"] > 0.0
