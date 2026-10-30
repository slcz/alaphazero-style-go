#!/usr/bin/env python3
"""train_supervised.py — supervised phase: fit human moves (KGS).

    python scripts/train_supervised.py --data-dir data/kgs \
        --out sup/ckpt.pt --steps 20000 --batch 512 --device cuda

Parses (or loads cached) SGF games, holds out a val split BY GAME, and
draws on-the-fly batches: (game, ply) uniform + a random D4 symmetry
(8× augmentation). Labels: played move (policy), RE result (value),
actual reply (aux). Ownership/score have no human labels and are
skipped — the heads ride along untrained until selfplay.

Ablation flags (the loss-head test surface): --no-value, --no-aux set
the corresponding weight to 0 (heads stay attached, no gradient).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from go_ai.data.sgf import load_dir
from go_ai.net.model import GoNet, NetConfig
from go_ai.train import supervised_data as SD
from go_ai.train.trainer import (TrainConfig, Trainer, tensors_to_torch)


def games_from_args(args) -> list:
    if args.cache and Path(args.cache).exists():
        games = SD.load_cache(args.cache)
        print(f"cache: {len(games)} games from {args.cache}")
        return games
    games = []
    for d in args.data_dir:
        p = REPO / d if not d.startswith("/") else Path(d)
        g = load_dir(p, b=args.b)
        print(f"  {p}: {len(g)} games kept")
        games.extend(g)
    if not games:
        print("no games (run scripts/download_kgs.py first)", file=sys.stderr)
        sys.exit(1)
    if args.cache:
        SD.cache_games(games, args.cache)
        print(f"cached {len(games)} games -> {args.cache}")
    return games


def eval_val(trainer, val_games, val_rows, rng, n=2048, batch=256):
    """Policy top-1 / value acc over a random val subsample."""
    trainer.model.eval()
    pol, val, n_seen = 0, 0, 0
    with torch.no_grad():
        for start in range(0, min(n, len(val_rows)), batch):
            rows = val_rows[start:start + batch]
            samples = [SD.supervised_sample(val_games[gi], i,
                                            int(rng.integers(0, 8)))
                       for gi, i in rows]
            batch_t = {k: np.stack([s[k] for s in samples])
                       for k in samples[0]}
            bt = tensors_to_torch(batch_t, trainer.device)
            out = trainer.model(bt["tokens"], bt["mask"], bt["aux_mask"])
            pred = out["policy_logp"].argmax(-1)
            pol += (pred == bt["pi_probs"].argmax(-1)).sum().item()
            val += (out["wdl"].argmax(-1) == bt["wdl"].argmax(-1)
                    ).sum().item()
            n_seen += len(samples)
    trainer.model.train()
    return pol / n_seen, val / n_seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", nargs="+", default=["data/kgs"])
    ap.add_argument("--cache", default="data/kgs/games.pkl")
    ap.add_argument("--out", default="sup/ckpt.pt")
    ap.add_argument("--base-ckpt", default=None)
    ap.add_argument("--b", type=int, default=19)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--no-value", action="store_true",
                    help="ablation: value weight 0")
    ap.add_argument("--no-aux", action="store_true",
                    help="ablation: aux weight 0")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tiny", action="store_true",
                    help="smoke-test config (d=64, 2 layers)")
    ap.add_argument("--val-n", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    t0 = time.time()
    games = games_from_args(args)
    rng = np.random.default_rng(args.seed)
    # deterministic by-game split (hash of index + seed)
    h = np.random.default_rng(args.seed).permutation(len(games))
    n_val = max(1, int(len(games) * args.val_frac))
    val_games = [games[i] for i in h[:n_val]]
    train_games = [games[i] for i in h[n_val:]]
    train_rows = [(gi, i) for gi, g in enumerate(train_games)
                  for i in range(len(g.moves))]
    val_rows = [(vi, i) for vi, g in enumerate(val_games)
                for i in range(len(g.moves))]
    print(f"{len(train_games)} train / {len(val_games)} val games, "
          f"{len(train_rows)} train positions, "
          f"{time.time() - t0:.0f}s total", flush=True)

    cfg = TrainConfig(lr=args.lr, batch_size=args.batch, device=args.device,
                      seed=args.seed,
                      own_weight=0.0, score_pdf_weight=0.0,
                      score_cdf_weight=0.0,
                      value_weight=0.0 if args.no_value else 1.0,
                      aux_reply_weight=0.0 if args.no_aux else 0.15,
                      grad_clip=1.0)
    if args.base_ckpt and Path(args.base_ckpt).exists():
        trainer = Trainer.load(args.base_ckpt, device=args.device)
        print(f"base: {args.base_ckpt}")
    else:
        net = (GoNet(NetConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128,
                               rope_pairs=2))
               if args.tiny else GoNet(NetConfig()))
        trainer = Trainer(net, cfg)
        print(f"random init ({sum(p.numel() for p in trainer.model.parameters()):,} "
              f"params)")

    p0, v0 = eval_val(trainer, val_games, val_rows, rng, n=args.val_n)
    print(f"val BEFORE: policy top1 {p0:.3f}  value {v0:.3f}", flush=True)
    history = []
    for step in range(1, args.steps + 1):
        pick = rng.integers(0, len(train_rows), size=args.batch)
        syms = rng.integers(0, 8, size=args.batch)
        proto = SD.supervised_sample(train_games[train_rows[pick[0]][0]],
                                     train_rows[pick[0]][1], int(syms[0]))
        batch = {k: np.zeros((args.batch,) + v.shape, v.dtype)
                 for k, v in proto.items()}
        for j, r in enumerate(pick):
            s = SD.supervised_sample(train_games[train_rows[r][0]],
                                     train_rows[r][1], int(syms[j]))
            for k, v in s.items():
                batch[k][j] = v
        m = trainer.train_step(tensors_to_torch(batch, trainer.device))
        history.append(m)
        if step % args.val_every == 0 or step == args.steps:
            p1, v1 = eval_val(trainer, val_games, val_rows, rng,
                              n=args.val_n)
            print(f"step {step}/{args.steps} loss {m['loss']:.4f} "
                  f"policy {m['policy']:.4f} val top1 {p1:.3f} "
                  f"value {v1:.3f} ({time.time() - t0:.0f}s)", flush=True)
            history[-1]["val_policy_top1"] = p1
            history[-1]["val_value_acc"] = v1

    out = Path(args.out)
    trainer.save(out)
    if args.json:
        tail = history[-min(20, len(history)):]
        Path(args.json).write_text(json.dumps({
            "games": len(train_games), "steps": args.steps,
            "final": {k: (float(np.mean([h[k] for h in tail]))
                          if not isinstance(tail[0][k], (int, float))
                          else float(np.mean([h[k] for h in tail])))
                      for k in tail[0]},
            "elapsed_s": round(time.time() - t0, 1)}, indent=2))
    print(f"saved {out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
