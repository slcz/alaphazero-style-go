"""
go_ai/net/nn_worker.py — the Python half of the NN evaluator
(engine/src/nn_eval.c). Ported from the chess worker; shapes are Go's:
rows are strided at the MAX board (361×18 tokens, 362 policy slots) and
sliced to the run's b before the forward.

Owns the model and the device tensors; host staging tensors are allocated
here (pinned where possible) and handed to C by raw pointer: walkers fill
planes/mask with nn_encode, the io thread drives the three-phase
protocol, delivery reads logp/wdl straight out of the host buffers. One
tensor set per landing-zone bank.

Three-phase protocol (as in chess):
  push(bank, n)       H2D + launch compute — ASYNC on CUDA (three
                      streams: s_h2d / s_compute / s_d2h, the forward
                      gated on the H2D event).
  pull(bank, n)       enqueue D2H for the results, gated on the forward.
  wait_ready(bank)    block until the bank's D2H completed.

On backends without streams (MPS, CPU) push runs the whole
H2D + forward + D2H synchronously and pull/wait_ready are no-ops. Runs
entirely on the landing zone's single io thread (the only Python entrant).
"""
from __future__ import annotations

import math
import os

import torch

from go_ai.net.model import GoNet, NetConfig
from go_ai.train.trainer import Trainer

_S: dict = {}
_MODELS: dict = {}          # (ckpt, device) -> loaded model (shared)

NN_TOKENS, NN_TOKEN_DIM, NN_POLICY = 361, 18, 362


def setup(ckpt: str, device: str, rows: int, banks: int = 1) -> dict:
    import sys as _sys
    print(f"SETUP ckpt={ckpt!r} device={device!r} rows={rows} banks={banks} "
          f"cuda_avail={torch.cuda.is_available()} "
          f"count={torch.cuda.device_count()}", file=_sys.stderr)
    key = (ckpt, device)
    if key not in _MODELS:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = GoNet(NetConfig(**payload["net_config"]))
        model.load_state_dict(payload["model"])
        _MODELS[key] = model.to(device).eval()
    model = _MODELS[key]

    cuda = device == "cuda"
    if cuda:
        _S.update(s_h2d=torch.cuda.Stream(), s_compute=torch.cuda.Stream(),
                  s_d2h=torch.cuda.Stream(),
                  ev_fwd=[torch.cuda.Event() for _ in range(banks)],
                  ev_free=[torch.cuda.Event() for _ in range(banks)])

    sets = []
    out = {"planes": [], "mask": [], "logp": [], "wdl": []}
    for _ in range(banks):
        try:
            pin = cuda
            x = torch.zeros(rows, NN_TOKENS, NN_TOKEN_DIM, pin_memory=pin)
        except (RuntimeError, NotImplementedError):     # no pinned memory
            pin = False
            x = torch.zeros(rows, NN_TOKENS, NN_TOKEN_DIM)
        s = {"x": x,
             "m": torch.zeros(rows, NN_POLICY, dtype=torch.bool,
                              pin_memory=pin),
             "lp": torch.empty(rows, NN_POLICY, pin_memory=pin),
             "w": torch.empty(rows, 2, pin_memory=pin)}
        if cuda:
            # device-side buffers per bank (static addresses), sized at
            # the MAX stride: the active board size is sliced per batch
            # (GTP boardsize may change between pushes)
            s["xd"] = torch.empty(rows, NN_TOKENS, NN_TOKEN_DIM,
                                  device=device,
                                  dtype=next(model.parameters()).dtype)
            s["md"] = torch.empty(rows, NN_POLICY, dtype=torch.bool,
                                  device=device)
            s["polf"] = torch.empty(rows, NN_POLICY, device=device)
            s["wdlf"] = torch.empty(rows, 2, device=device)
        else:
            s["xd"] = torch.empty(rows, NN_TOKENS, NN_TOKEN_DIM,
                                  device=device,
                                  dtype=next(model.parameters()).dtype)
            s["md"] = torch.empty(rows, NN_POLICY, dtype=torch.bool,
                                  device=device)
        sets.append(s)
        out["planes"].append(s["x"].data_ptr())
        out["mask"].append(s["m"].data_ptr())
        out["logp"].append(s["lp"].data_ptr())
        out["wdl"].append(s["w"].data_ptr())
    # compile defaults on for the CUDA path (NN_COMPILE=0 disables)
    if os.environ.get("NN_COMPILE", "1" if cuda else "0") == "1":
        model = torch.compile(model, dynamic=True)   # variable batch n
    _S.update(model=model, device=device, cuda=cuda, sets=sets,
              bf16=os.environ.get("NN_BF16", "1") == "1",
              adtype=(torch.float16 if os.environ.get("NN_AUTOCAST") == "fp16"
                      else torch.bfloat16))
    return out


def _active_b(bank: int, n: int) -> int:
    """Board size of the staged rows: every row's legal set includes the
    pass slot (b²), so the last active mask column is exactly b²."""
    m = _S["sets"][bank]["m"][:n]
    return math.isqrt(int(torch.nonzero(m.any(dim=0))[-1].item()))


def push(bank: int, n: int) -> int:
    s = _S["sets"][bank]
    b = _active_b(bank, n)
    T = b * b                  # active tokens
    if not _S["cuda"]:
        # synchronous backend: the whole forward at push
        with torch.inference_mode():
            s["xd"][:n, :T].copy_(s["x"][:n, :T])
            s["md"][:n, :b * b + 1].copy_(s["m"][:n, :b * b + 1])
            out = _S["model"](s["xd"][:n, :T], s["md"][:n, :b * b + 1])
            s["lp"][:n].zero_()                  # tail stays 0 for C
            s["lp"][:n, :b * b + 1].copy_(out["policy_logp"].cpu())
            s["w"][:n] = torch.softmax(out["wdl"], dim=-1).cpu()
        return n
    # CUDA: H2D on the copy stream, forward on the compute stream, gated
    # on the H2D event — everything async, returns before the GPU runs it
    ev_h2d = torch.cuda.Event()
    with torch.cuda.stream(_S["s_h2d"]):
        s["xd"][:n, :T].copy_(s["x"][:n, :T], non_blocking=True)
        s["md"][:n, :b * b + 1].copy_(s["m"][:n, :b * b + 1],
                                      non_blocking=True)
        ev_h2d.record(_S["s_h2d"])
    _S["s_compute"].wait_event(ev_h2d)
    with torch.cuda.stream(_S["s_compute"]):
        with torch.inference_mode():
            if _S.get("bf16", True):
                with torch.autocast("cuda", dtype=_S["adtype"]):
                    out = _S["model"](s["xd"][:n, :T], s["md"][:n, :b * b + 1])
            else:
                out = _S["model"](s["xd"][:n, :T], s["md"][:n, :b * b + 1])
            s["polf"][:n].zero_()
            s["polf"][:n, :b * b + 1].copy_(out["policy_logp"])
            # wdl leaves the model as raw logits; the search backs up
            # w[0]-w[1], which is 2pW-1 only after a softmax
            s["wdlf"][:n].copy_(torch.softmax(out["wdl"].float(), dim=-1))
            _S["ev_fwd"][bank].record(_S["s_compute"])
    return n


def pull(bank: int, n: int) -> int:
    if not _S["cuda"]:
        return n                      # synchronous backend: already on host
    s = _S["sets"][bank]
    _S["s_d2h"].wait_event(_S["ev_fwd"][bank])
    with torch.cuda.stream(_S["s_d2h"]):
        s["lp"][:n].copy_(s["polf"][:n], non_blocking=True)
        s["w"][:n].copy_(s["wdlf"][:n], non_blocking=True)
        _S["ev_free"][bank].record(_S["s_d2h"])
    return n


def wait_ready(bank: int, n: int = 0) -> int:
    if _S["cuda"]:
        _S["ev_free"][bank].synchronize()
    return bank
