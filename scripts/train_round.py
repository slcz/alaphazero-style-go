#!/usr/bin/env python3
"""train_round.py — train one round's checkpoint from self-play shards.

    python scripts/train_round.py --data-dir it001/selfplay [more dirs] \
        --out it001/train.pt [--base-ckpt best.pt] --epochs 1 \
        --batch 512 --lr 2e-4 --device cuda

Games come from the units' DONE-gated shards; every trainable ply (PCR
slow-cap filter) yields syms=8 D4-augmented samples drawn ON THE FLY —
the round's samples are never materialized (tokens for a 30k-game round
would be ~20 GB; the replay is C-side and cheap). Steps = ceil(trainable
plies × 8 × epochs / batch). No --base-ckpt = random init.
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

from go_ai.net.model import GoNet, NetConfig
from go_ai.selfplay.shard import read_unit
from go_ai.train.selfplay_data import K, selfplay_sample, trainable_plies
from go_ai.train.trainer import (Trainer, TrainConfig, resolve_device,
                                 tensors_to_torch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--adam-lr", type=float, default=2e-4,
                    help="companion AdamW lr for non-Muon params")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500,
                    help="checkpoint to <out>-ck.pt every N steps")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    t0 = time.time()
    games = []
    for d in args.data_dir:
        games.extend(read_unit(d))
    if not games:
        print("no games", file=sys.stderr)
        return 1
    bs = {g.b for g in games}
    assert len(bs) == 1, f"mixed board sizes in one round: {bs}"
    rows = [(gi, i) for gi, g in enumerate(games)
            for i in trainable_plies(g)]
    steps = max(1, -(-len(rows) * 8 * args.epochs // args.batch))
    print(f"train: {len(games)} games, {len(rows)} trainable plies, "
          f"steps={steps} (8x D4 aug, batch {args.batch})", flush=True)

    if args.base_ckpt and Path(args.base_ckpt).exists():
        dev = resolve_device(args.device)   # returns a device STRING
        trainer = Trainer.load(
            args.base_ckpt, device=args.device,
            cfg_overrides=dict(lr=args.lr, batch_size=args.batch,
                               adam_lr=args.adam_lr, seed=args.seed,
                               compile=str(dev).startswith("cuda")))
        print(f"base: {args.base_ckpt} (steps {trainer.steps}) — "
              f"muon lr={trainer.cfg.lr} "
              f"batch={trainer.cfg.batch_size} compile={trainer.cfg.compile}",
              flush=True)
    else:
        model = GoNet(NetConfig())
        cfg = TrainConfig(lr=args.lr, batch_size=args.batch,
                          device=args.device, seed=args.seed)
        trainer = Trainer(model, cfg)
        print(f"random init ({sum(p.numel() for p in model.parameters()):,} "
              f"params)", flush=True)

    rng = np.random.default_rng(args.seed)
    b = games[0].b
    gi0, i0 = rows[0]        # game 0 may have no trainable (slow-cap) plies
    proto = selfplay_sample(games[gi0], i0)   # per-sample shapes
    batch = {k: np.zeros((args.batch,) + v.shape, v.dtype)
             for k, v in proto.items()}
    history = []
    for step in range(steps):
        pick = rng.integers(0, len(rows), size=args.batch)
        syms = rng.integers(0, 8, size=args.batch)
        for j, r in enumerate(pick):
            gi, i = rows[r]
            s = selfplay_sample(games[gi], i, int(syms[j]))
            for k, v in s.items():
                batch[k][j] = v
        m = trainer.train_step(tensors_to_torch(batch, trainer.device))
        history.append(m)
        if args.save_every and (step + 1) % args.save_every == 0:
            trainer.save(Path(args.out).with_suffix(".ck.pt"))
        if (step + 1) % args.log_every == 0 or step == steps - 1:
            print(f"step {step + 1}/{steps} loss {m['loss']:.4f} "
                  f"policy {m['policy']:.4f} (acc {m['policy_acc']:.3f}) "
                  f"value {m['value']:.4f} own {m['own']:.4f} "
                  f"score {m['score_pdf']:.4f}", flush=True)

    out = Path(args.out)
    trainer.save(out)
    if args.json:
        n_log = min(len(history), 20)
        tail = history[-n_log:]
        Path(args.json).write_text(json.dumps({
            "games": len(games), "trainable_plies": len(rows), "steps": steps,
            "final": {k: float(np.mean([h[k] for h in tail]))
                      for k in tail[0]},
            "elapsed_s": round(time.time() - t0, 1),
        }, indent=2))
    print(f"saved {out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
