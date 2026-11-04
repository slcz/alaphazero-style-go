"""Score/ownership accuracy at the moment the value crosses 0.95,
compared with the game's actual TT end result."""
import sys, random
sys.path.insert(0, "/Users/sileizhang/go")
import numpy as np
from go_ai.selfplay.shard import read_shard, SP_END_NATURAL, MOVE_PASS
from go_ai.engine import Board, BLACK, WHITE
from go_ai.net import nn_worker

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--shard", default=".hf-cache/dl/9x9/it018/sp9/shard-0.spbin")
_ap.add_argument("--ckpt", default=".hf-cache/dl/9x9/it018-train.pt")
_ap.add_argument("--games", type=int, default=250)
_args = _ap.parse_args()
SHARD, CKPT = _args.shard, _args.ckpt
nn_worker.setup(CKPT, device="mps", rows=8, banks=1)
B = 9; T = B * B

def raw(bd):
    mask = np.zeros(T + 1, np.uint8)
    for m in bd.legal_moves:
        mask[T if m == MOVE_PASS else m] = 1
    planes = np.zeros((361, 18), np.float32)
    planes[:T] = np.frombuffer(bd.encode(), np.float32).reshape(T, 18)
    _, wdl, own, sc = nn_worker.raw(planes.tobytes(), mask.tobytes(), B)
    w = np.frombuffer(wdl, np.float32)
    o = np.frombuffer(own, np.float32)
    s = float(np.frombuffer(sc, np.float32)[0])
    return (w[0] - w[1]), o.astype(np.float64), s      # all mover-view

def main(n=250, seed=0):
    games = [g for g in read_shard(SHARD) if g.end_kind == SP_END_NATURAL]
    sample = random.Random(seed).sample(games, min(n, len(games)))
    out = {}
    for label, off in (("at the 0.95 crossing", 0),
                       ("20 plies later", 20),
                       ("at the last trained ply", -1)):
        rows = []
        for g in sample:
            bd = Board(B, g.komi, g.n_plies)
            hit = None
            for i, p in enumerate(g.plies):
                v, _, _ = raw(bd)
                if hit is None and abs(v) >= 0.95:
                    hit = i
                if i == (hit + off if hit is not None else -99):
                    rows.append((i, bd.side_to_move, raw(bd), g))
                    break
                bd.push(p.move)
            else:
                if off == -1 and hit is not None:      # last ply
                    bd2 = Board(B, g.komi, g.n_plies)
                    for p in g.plies[:-1]:
                        bd2.push(p.move)
                    rows.append((g.n_plies - 1, bd2.side_to_move, raw(bd2), g))
        if not rows:
            continue
        se, oe, omae, agree, tot = [], [], [], 0, 0
        for i, mover, (v, own_m, sc_m), g in rows:
            sign = 1.0 if mover == BLACK else -1.0
            pred_score = sign * sc_m                 # black view
            pred_own = own_m if mover == BLACK else 1.0 - own_m
            true_score = g.score
            true_own = np.asarray(g.ownership, np.float64)
            se.append(pred_score - true_score)
            omae.append(np.abs(pred_own - true_own).mean())
            m = np.abs(true_own - 0.5) > 0.1         # ignore dame
            if m.any():
                agree += int(((pred_own[m] > 0.5) == (true_own[m] > 0.5)).sum())
                tot += int(m.sum())
        se = np.array(se); om = np.array(omae)
        out[label] = (len(rows), float(np.abs(se).mean()), float(se.mean()),
                      float(np.sqrt((se**2).mean())), float(om.mean()),
                      100.0 * agree / max(tot, 1))
    print(f"{len(sample)} natural 9x9 games of it018\n")
    print(f"{'when':24s} {'n':>4s} {'score MAE':>10s} {'bias':>7s} "
          f"{'RMSE':>7s} {'own MAE':>8s} {'owner acc':>10s}")
    for k, (n_, mae, bias, rmse, omae, acc) in out.items():
        print(f"{k:24s} {n_:4d} {mae:10.1f} {bias:+7.1f} {rmse:7.1f} "
              f"{omae:8.3f} {acc:9.1f}%")
main(_args.games)
