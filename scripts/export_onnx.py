"""export_onnx.py — export GoNet checkpoints to ONNX for the web GUI
(gui branch).

The browser runs the net through onnxruntime-web; this is the offline
half. One .onnx PER BOARD SIZE: the transformer derives b = isqrt(T)
inside forward (rope tables, edge dims), which tracing bakes in — a
dynamic-T graph would silently apply the traced size's tables. The
batch axis is dynamic (the search batches parked eval rows).

The exported graph mirrors the wasm eval seam (engine/src/wasm_glue.c;
go_ai/net/nn_worker.py is the native twin):

    inputs : tokens [B, b*b, 18] float32 (the C nn_encode planes,
             token-major)
             mask   [B, 362]      bool (True = legal; b*b+1 active)
    outputs: policy_logp [B, b*b+1]  illegal slots exp→ EXACTLY 0 (the
                                      search's hard contract, model.py)
             wdl        [B, 2]      softmax probs [pW, pL], mover view
             own        [B, b*b]    sigmoid probs, mover view
             score      [B]         predicted margin, points, mover view

The aux opponent-reply head is training-time only — dropped (smaller
file, no aux_mask input, the GUI never consumes it).

Parity gate: identical inputs through torch and onnxruntime-CPU on REAL
positions — random playouts through the FFI Board, encoded by the C
encoder (Board.encode), masks built from move_to_slot over legal moves.

    python scripts/export_onnx.py --ckpt ckpts/best.pt --sizes 7,9
    python scripts/export_onnx.py --random --sizes 7,9,13,19
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from go_ai.net.model import GoNet, NetConfig      # noqa: E402
from go_ai.net.features import PI_SUPPORT_MAX, move_to_slot  # noqa: E402
from go_ai.engine import PASS, TERM_OVER, Board   # noqa: E402

# Tolerances of the gate (fp32 torch vs fp32 ORT-CPU is op-scheduling
# noise; these sit far below anything PUCT can feel).
TOL_LOGP = 2e-4
TOL_WDL = 1e-4
TOL_OWN = 1e-4
TOL_SCORE = 1e-3


class InferenceWrapper(nn.Module):
    """Trimmed forward for export: the four heads the search and the GUI
    consume, in the seam's semantics. Softmax/sigmoid applied HERE so
    the wasm glue feeds ORT output straight to C unchanged."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        x = self.model.trunk(tokens)
        logp = self.model._masked_logp(self.model.policy(x), mask)
        wdl = torch.softmax(self.model.value(x), dim=-1)
        own = torch.sigmoid(self.model.own(x))
        score = self.model.score(x)
        return logp, wdl, own, score


def export_onnx(model: nn.Module, b: int, out_path: Path,
                opset: int = 17) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    n = b * b
    tokens = torch.zeros(4, n, model.cfg.token_dim)
    mask = torch.zeros(4, PI_SUPPORT_MAX, dtype=torch.bool)
    mask[:, :n + 1] = True                 # any non-empty mask traces fine
    with torch.inference_mode():
        torch.onnx.export(
            InferenceWrapper(model), (tokens, mask), str(out_path),
            opset_version=opset, dynamo=False,
            input_names=["tokens", "mask"],
            output_names=["policy_logp", "wdl", "own", "score"],
            dynamic_axes={name: {0: "batch"} for name in
                          ("tokens", "mask", "policy_logp", "wdl", "own",
                           "score")},
        )
    return out_path


def random_positions(b: int, n_pos: int, seed: int = 0, max_plies: int = 120):
    """n random legal positions at b×b as (tokens [n, b*b, 18], mask
    [n, 362]) numpy arrays — random playouts through the FFI Board,
    encoded by the C encoder. Terminal positions are skipped: the engine
    never evaluates those, so the gate shouldn't either."""
    rng = random.Random(seed)
    komi = 9.5 if b == 7 else 7.5
    maxply = min(3 * b * b, 1024)          # the engine's MAX_PLY wall
    tokens, masks = [], []
    while len(tokens) < n_pos:
        bd = Board(b, komi, maxply)
        for _ in range(rng.randrange(max_plies)):
            if bd.termination()[0] == TERM_OVER:
                break
            bd.push(rng.choice(bd.legal_moves))
        if bd.termination()[0] == TERM_OVER:
            continue
        m = np.zeros(PI_SUPPORT_MAX, dtype=bool)
        for mv in bd.legal_moves:
            m[move_to_slot(mv, b)] = True
        tokens.append(bd.encode())
        masks.append(m)
    return np.stack(tokens), np.stack(masks)


def parity_gate(model: nn.Module, b: int, onnx_path: Path,
                n_pos: int = 48, seed: int = 0) -> dict:
    import onnxruntime as ort

    tokens, mask = random_positions(b, n_pos, seed)
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    o_logp, o_wdl, o_own, o_score = sess.run(
        None, {"tokens": tokens, "mask": mask})

    with torch.inference_mode():
        t_logp, t_wdl, t_own, t_score = InferenceWrapper(model)(
            torch.from_numpy(tokens), torch.from_numpy(mask))
    t_logp, t_wdl = t_logp.numpy(), t_wdl.numpy()
    t_own, t_score = t_own.numpy(), t_score.numpy()

    active = mask[:, : b * b + 1]
    d_logp = np.abs(t_logp - o_logp)
    with np.errstate(under="ignore"):      # exp(-inf-ish) underflows to 0
        illegal_prob = np.exp(o_logp[~active])
    assert d_logp[active].max() <= TOL_LOGP, \
        f"policy_logp drift {d_logp[active].max():.2e} > {TOL_LOGP}"
    assert np.abs(t_wdl - o_wdl).max() <= TOL_WDL, "wdl drift"
    assert np.abs(t_own - o_own).max() <= TOL_OWN, "own drift"
    assert np.abs(t_score - o_score).max() <= TOL_SCORE, "score drift"
    assert illegal_prob.max(initial=0.0) == 0.0, \
        "illegal slot has nonzero probability in the ONNX graph"
    return {
        "b": b, "positions": n_pos,
        "max_d_logp_legal": float(d_logp[active].max()),
        "max_d_wdl": float(np.abs(t_wdl - o_wdl).max()),
        "max_d_own": float(np.abs(t_own - o_own).max()),
        "max_d_score": float(np.abs(t_score - o_score).max()),
        "argmax_agree": int((t_logp.argmax(-1) == o_logp.argmax(-1)).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="trainer checkpoint (fp8 payloads dequantized)")
    ap.add_argument("--random", action="store_true",
                    help="export a random-init net (GUI development)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", default="7,9",
                    help="board sizes to export, comma-separated")
    ap.add_argument("--out-dir", default="web/assets/net")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--gate", type=int, default=48,
                    help="parity positions per size (0 = skip)")
    args = ap.parse_args()

    if args.random:
        torch.manual_seed(args.seed)
        model = GoNet(NetConfig())
        origin = "random"
    else:
        from go_ai.train.trainer import Trainer
        model = Trainer.load(args.ckpt).model
        origin = args.ckpt
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {origin} ({n_params:,} params)")

    out_dir = REPO / args.out_dir
    for b in (int(x) for x in args.sizes.split(",")):
        out = out_dir / f"net-{b}.onnx"
        export_onnx(model, b, out, args.opset)
        size_mb = out.stat().st_size / 1e6
        line = f"{out.name}: {size_mb:.1f} MB"
        if args.gate:
            res = parity_gate(model, b, out, n_pos=args.gate,
                              seed=args.seed + b)
            line += (f"  parity ok (d_logp {res['max_d_logp_legal']:.1e},"
                     f" d_wdl {res['max_d_wdl']:.1e},"
                     f" d_own {res['max_d_own']:.1e},"
                     f" d_score {res['max_d_score']:.1e},"
                     f" argmax {res['argmax_agree']}/{res['positions']})")
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
