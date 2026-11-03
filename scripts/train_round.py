#!/usr/bin/env python3
"""train_round.py — train one checkpoint from the self-play replay buffer.

    python scripts/train_round.py --buffer work/buffer.json \\
        --out it004/train.pt [--base-ckpt best.pt] --epochs 1 \\
        --batch 3072 --lr 0.01 --device cuda

The buffer manifest (buffer.json, written and trimmed by selfplay_loop.py)
lists, per preserved round, the shard and the game ids to train on:
`{"rounds": [{"round": 3, "shard": "it003/selfplay/shard-0.spbin",
"games": [...]}]}`. Rounds take no weight of their own — they are already
thinned to the intended mix by the manifest.

Each round's shard is materialized ONCE into a compact row cache
(go_ai/train/rowcache.py: ~400 B/row, one forward replay per game) and the
manifest's game ids select rows out of it. A step is then a handful of
numpy gathers instead of one cffi replay per sample — the old path spent
~0.45 s of CPU per 3072-row step and left the GPU at ~18% duty.

Steps = ceil(rows × 8 × epochs / batch) with rows the SELECTED rows, and
each step draws one D4 symmetry for the whole batch. `--data-dir` trains
whole rounds instead (tests, one-off runs). No --base-ckpt = random init.
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
from go_ai.train import rowcache
from go_ai.train.trainer import (Trainer, TrainConfig, resolve_device,
                                 tensors_to_torch)


def load_buffer(manifest):
    """buffer.json → [(round, unit_dir, game_ids)], dead rounds dropped (a
    restored node may not have every shard)."""
    m = json.loads(Path(manifest).read_text())
    base = Path(manifest).resolve().parent
    out = []
    for r in m["rounds"]:
        p = base / r["shard"]
        if not p.exists():
            print(f"  round {r['round']:03d}: shard missing ({p}) — dropped",
                  flush=True)
            continue
        out.append((int(r["round"]), p.parent, r["games"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", nargs="*", default=[],
                    help="whole rounds (test/one-off path)")
    ap.add_argument("--buffer", default=None,
                    help="buffer.json manifest: preserved game ids per "
                         "round (preferred)")
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
    ap.add_argument("--max-steps", type=int, default=0,
                    help="cap the step count (debug runs; 0 = no cap)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    t0 = time.time()
    rounds = []                       # (label, cache, selected rows, games)
    if args.buffer:
        for round_no, unit, ids in load_buffer(args.buffer):
            c = rowcache.load_or_build(unit)
            sel = rowcache.select(c, ids)
            rounds.append((f"it{round_no:03d}", c, sel, len(ids)))
            print(f"  round {round_no:03d}: {len(ids)} games, {len(sel)} rows",
                  flush=True)
    for d in args.data_dir:
        c = rowcache.load_or_build(d)
        sel = rowcache.select(c, None)
        n = int(c["game"].max()) + 1 if len(c["game"]) else 0
        rounds.append((Path(d).parent.name, c, sel, n))
        print(f"  {d}: {n} games, {len(sel)} rows", flush=True)
    if not rounds or not any(len(r[2]) for r in rounds):
        print("no trainable rows", file=sys.stderr)
        return 1

    # group by (board size, komi): a batch has one token width, so a step
    # is one micro-batch per size and they interleave inside the step
    groups = {}
    for lab, c, sel, _g in rounds:
        groups.setdefault((int(c["b"]), float(c["komi"])), []).append(
            (lab, c, sel))
    rows = sum(len(s) for s in (r[2] for r in rounds))
    games = sum(r[3] for r in rounds)
    steps = max(1, -(-rows * 8 * args.epochs // args.batch))
    if args.max_steps:
        steps = min(steps, args.max_steps)
    # per-size row counts, and how many rows of each size one step takes:
    # keep the TOKENS per step constant, so a 15x15 micro-batch is small in
    # rows but the same cost as the 9x9 one
    share = {}
    for k, gs in groups.items():
        share[k] = sum(len(s) for _l, _c, s in gs) / max(rows, 1)
    per_step = {}
    for k in groups:
        b_k = k[0]
        per_step[k] = max(1, int(round(share[k] * args.batch * 81
                                       / (b_k * b_k))))
    print(f"train: {games} games over {len(rounds)} round(s), {rows} rows, "
          f"steps={steps} (8x D4, {args.batch} 9x9-equivalent rows/step)",
          flush=True)
    for k in sorted(groups):
        print(f"  {k[0]}x{k[0]} komi {k[1]}: {len(groups[k])} round(s), "
              f"{int(share[k] * rows)} rows ({100 * share[k]:.0f}%), "
              f"{per_step[k]} rows/step", flush=True)

    if args.base_ckpt and Path(args.base_ckpt).exists():
        dev = resolve_device(args.device)   # returns a device STRING
        trainer = Trainer.load(
            args.base_ckpt, device=args.device,
            cfg_overrides=dict(lr=args.lr, batch_size=args.batch,
                               adam_lr=args.adam_lr, seed=args.seed,
                               compile=str(dev).startswith("cuda")))
        print(f"base: {args.base_ckpt} (steps {trainer.steps}) — "
              f"muon lr={trainer.cfg.lr} "
              f"batch={trainer.cfg.batch_size} compile={trainer.cfg.compile} "
              f"bf16={trainer.cfg.bf16}", flush=True)
    else:
        model = GoNet(NetConfig())
        cfg = TrainConfig(lr=args.lr, batch_size=args.batch,
                          device=args.device, seed=args.seed)
        trainer = Trainer(model, cfg)
        print(f"random init ({sum(p.numel() for p in model.parameters()):,} "
              f"params)", flush=True)

    rng = np.random.default_rng(args.seed)
    bufs = {k: rowcache.alloc_batch(per_step[k], k[0]) for k in groups}
    history = []
    for step in range(steps):
        chunks = []
        for k, gs in groups.items():
            caches_k = [c for _l, c, _s in gs]
            sels_k = [s for _l, _c, s in gs]
            n_k = per_step[k]
            picks = rowcache.draw(n_k, sels_k, rng)
            buf = bufs[k]
            n = rowcache.fill_batch(caches_k, picks, int(rng.integers(0, 8)),
                                    buf, k[0], k[1])
            if n:
                chunks.append(tensors_to_torch(
                    {kk: v[:n] for kk, v in buf.items()}, trainer.device))
        if not chunks:
            break
        m = trainer.train_step_multi(chunks)
        history.append(m)
        if args.save_every and (step + 1) % args.save_every == 0:
            trainer.save(Path(args.out).with_suffix(".ck.pt"))
        if (step + 1) % args.log_every == 0 or step == steps - 1:
            print(f"step {step + 1}/{steps} loss {m['loss']:.4f} "
                  f"policy {m['policy']:.4f} (acc {m['policy_acc']:.3f}) "
                  f"value {m['value']:.4f} own {m['own']:.4f} "
                  f"score {m['score']:.3f}", flush=True)

    out = Path(args.out)
    trainer.save(out)
    if args.json:
        n_log = min(len(history), 20)
        tail = history[-n_log:]
        Path(args.json).write_text(json.dumps({
            "games": games, "trainable_plies": rows, "steps": steps,
            "rounds": [(r[0], r[3], len(r[2])) for r in rounds],
            "final": {k: float(np.mean([h[k] for h in tail]))
                      for k in tail[0]},
            "elapsed_s": round(time.time() - t0, 1),
        }, indent=2))
    print(f"saved {out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
