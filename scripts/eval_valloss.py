#!/usr/bin/env python3
"""eval_valloss.py — held-out val losses for a supervised checkpoint.

    python scripts/eval_valloss.py sup/base.pt [--n 8192]

Same by-game split as training (seed 1, val-frac 0.02). Forward passes
only: policy CE (nats + ppl + top-1/top-5), value CE + acc, aux CE."""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from go_ai.train import supervised_data as SD
from go_ai.train.trainer import Trainer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    games = pickle.load(open(REPO / "data/kgs/games.pkl", "rb"))
    h = np.random.default_rng(a.seed).permutation(len(games))
    n_val = max(1, int(len(games) * 0.02))
    val_games = [games[i] for i in h[:n_val]]
    val_rows = [(vi, i) for vi, g in enumerate(val_games)
                for i in range(len(g.moves))]

    tr = Trainer.load(a.ckpt, device="cuda")
    rng = np.random.default_rng(a.seed + 7)
    pick = rng.permutation(len(val_rows))[:a.n]
    B = a.batch

    # build all batches first so the timed loop measures pure forward
    batches = []
    for s in range(0, len(pick), B):
        rows = [val_rows[i] for i in pick[s:s + B]]
        samples = [SD.supervised_sample(val_games[gi], i,
                                        int(rng.integers(0, 8)))
                   for gi, i in rows]
        batches.append({k: torch.from_numpy(
            np.stack([x[k] for x in samples])).cuda() for k in samples[0]})

    agg = {"pol": 0.0, "val": 0.0, "aux": 0.0, "w_aux": 0.0,
           "t1": 0, "t5": 0, "vacc": 0, "n": 0}
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    t0 = time.time()
    ev0.record()
    with torch.no_grad(), tr._autocast():
        for bt in batches:
            out = tr.model(bt["tokens"], bt["mask"], bt["aux_mask"])

            slot_logp = out["policy_logp"].gather(1, bt["pi_slots"])
            pol = -(bt["pi_probs"] * slot_logp).sum(-1)
            vlp = torch.log_softmax(out["wdl"], dim=-1)
            vl = -(vlp * bt["wdl"]).sum(-1)
            anll = -out["aux_logp"].gather(
                1, bt["aux_idx"][:, None]).squeeze(1) * bt["aux_w"]

            tgt = bt["pi_probs"].argmax(-1)
            top5 = out["policy_logp"].topk(5, dim=-1).indices
            agg["pol"] += pol.sum().item()
            agg["val"] += vl.sum().item()
            agg["aux"] += anll.sum().item()
            agg["w_aux"] += bt["aux_w"].sum().item()
            agg["t1"] += (top5[:, 0] == tgt).sum().item()
            agg["t5"] += (top5 == tgt[:, None]).any(-1).sum().item()
            agg["vacc"] += (out["wdl"].argmax(-1)
                            == bt["wdl"].argmax(-1)).sum().item()
            agg["n"] += len(bt["tokens"])
    ev1.record()
    torch.cuda.synchronize()
    wall = time.time() - t0
    gpu = ev0.elapsed_time(ev1) / 1e3
    n = agg["n"]
    gflop = 6.01e9                      # 16.65 MFLOPs/token x 361
    print(f"policy CE {agg['pol']/n:.4f} nats  (ppl {np.exp(agg['pol']/n):.2f})  "
          f"top1 {agg['t1']/n:.4f}  top5 {agg['t5']/n:.4f}")
    print(f"value   CE {agg['val']/n:.4f} nats  acc {agg['vacc']/n:.4f}")
    print(f"aux     CE {agg['aux']/max(agg['w_aux'],1.0):.4f} nats")
    print(f"throughput: {n/gpu:,.0f} samples/s forward (gpu {gpu:.2f}s)  "
          f"= {n*gflop/gpu/1e12:.1f} TFLOP/s   "
          f"end-to-end {n/wall:,.0f} samples/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
