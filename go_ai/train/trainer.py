"""
go_ai/train/trainer.py — the training loop (ported from the chess
trainer; losses extended with the Go targets).

    total = policy_weight · policy_loss        (dense π̄ target-weighted CE)
          + value_weight · value_loss          (2-way W/L CE)
          + aux_reply_weight · aux_loss        (opponent-reply CE)
          + own_weight · ownership_loss        (per-point BCE, mean over
                                                points; KataGo wo=1.5/b²
                                                is the mean form)
          + score_pdf_weight · score_pdf_loss  (bin CE; KataGo 0.02)
          + score_cdf_weight · score_cdf_loss  (cdf MSE; KataGo 0.02)

Adam (fused on CUDA), gradient clipping, bf16 autocast, torch.compile —
all as in chess. Checkpoints are atomic (tmp + rename) and carry the
net config so any load rebuilds the exact architecture.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from go_ai.net.model import GoNet, NetConfig, to_half_


@dataclass
class TrainConfig:
    lr: float = 3e-4
    adam_lr: float = 2e-4             # companion AdamW: non-Muon params
                                      # (heads, norms, embeddings)
    batch_size: int = 256
    value_weight: float = 1.0
    policy_weight: float = 1.0
    aux_reply_weight: float = 0.15    # KataGo wopp
    own_weight: float = 1.5           # KataGo wo (mean-over-points form)
    score_pdf_weight: float = 0.02    # KataGo wspdf
    score_cdf_weight: float = 0.02    # KataGo wscdf
    grad_clip: Optional[float] = 1.0
    weight_decay: float = 0.0
    bf16: bool = True              # CUDA autocast (no-op off CUDA)
    compile: bool = True           # torch.compile (CUDA only)
    device: str = "auto"
    seed: int = 0


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_device(name: str) -> str:
    """A checkpoint's saved device is a TRAINING-time fact, not a
    load-time requirement (resuming a CUDA run on the Mac must work)."""
    if name == "cuda" and not torch.cuda.is_available():
        return "auto"
    if name == "mps" and not torch.backends.mps.is_available():
        return "auto"
    return name


_TORCH_DTYPES = {
    "tokens": torch.float32,
    "mask": torch.bool,
    "pi_slots": torch.int64,
    "pi_probs": torch.float32,
    "pi": torch.float32,
    "wdl": torch.float32,
    "own": torch.float32,
    "score_bin": torch.int64,
    "aux_idx": torch.int64,
    "aux_mask": torch.bool,
    "aux_w": torch.float32,
}


def tensors_to_torch(batch: Dict[str, np.ndarray],
                     device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: torch.from_numpy(np.ascontiguousarray(v)).to(
            device=device, dtype=_TORCH_DTYPES[k])
        for k, v in batch.items()
    }


class Trainer:
    """Owns model + optimizer + step count; one train_step per batch."""

    def __init__(self, model: GoNet, cfg: TrainConfig):
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        self.model = model.to(self.device)
        self._raw_model = model          # unwrapped: state_dict stays clean
        self._maybe_compile()
        params = list(model.parameters())
        mu = [p for p in params
              if p.dim() == 2 and min(p.shape) >= 64]
        mu_ids = {id(p) for p in mu}
        ad = [p for p in params if id(p) not in mu_ids]
        self.opt = torch.optim.Muon(mu, lr=cfg.lr,
                                    weight_decay=cfg.weight_decay,
                                    adjust_lr_fn="original")
        try:                             # fused Adam: one update kernel
            self.opt_adam = torch.optim.Adam(
                ad, lr=cfg.adam_lr, fused=self.device.type == "cuda",
                weight_decay=cfg.weight_decay)
        except TypeError:
            self.opt_adam = torch.optim.Adam(ad, lr=cfg.adam_lr,
                                             weight_decay=cfg.weight_decay)
        self.steps = 0
        torch.manual_seed(cfg.seed)

    def _maybe_compile(self) -> None:
        if self.cfg.compile and self.device.type == "cuda":
            torch._inductor.config.triton.cudagraphs = False
            # eval (256) then train (512) on the same code otherwise
            # flips dynamo into symbolic-shape mode: minutes of compile
            torch._dynamo.config.automatic_dynamic_shapes = False
            self.model = torch.compile(self._raw_model)
        else:
            self.model = self._raw_model

    # --- one optimization step ------------------------------------------

    def _autocast(self):
        if self.cfg.bf16 and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def _losses(self, batch: Dict[str, torch.Tensor]):
        """Forward + loss terms for one (micro-)batch; backward is the
        caller's job."""
        with self._autocast():
            out = self.model(batch["tokens"], batch["mask"],
                             batch["aux_mask"])

            # dense π̄ target: −Σ_a π(a)·logp_θ(a) over the policy row
            # stride. π is 0 outside the legal set, so the masked slots
            # contribute 0·(large negative) = 0.
            logp = out["policy_logp"]
            policy_loss = -(batch["pi"] * logp).sum(-1).mean()
            policy_acc = ((logp.argmax(-1)
                           == batch["pi"].argmax(-1)).float().mean())

            wdl_logp = F.log_softmax(out["wdl"], dim=-1)
            value_loss = -(wdl_logp * batch["wdl"]).sum(-1).mean()

            if self.model.aux is not None:
                aux_nll = -out["aux_logp"].gather(
                    1, batch["aux_idx"][:, None]).squeeze(1)
                w = batch["aux_w"]
                aux_loss = (aux_nll * w).sum() / w.sum().clamp(min=1.0)
            else:
                aux_loss = torch.zeros((), device=self.device)

            # ownership/score labels exist only in selfplay (human games
            # end by resignation — no scoring position); absent keys skip
            # the loss, so the supervised phase trains policy/value/aux
            # only and the heads ride along untrained
            if "own" in batch:
                own_loss = F.binary_cross_entropy_with_logits(
                    out["own"], batch["own"])
            else:
                own_loss = torch.zeros((), device=self.device)

            if "score_bin" in batch:
                score_logp = F.log_softmax(out["score"], dim=-1)
                score_pdf_loss = F.nll_loss(score_logp, batch["score_bin"])

                with torch.no_grad():        # target cdf from the one-hot
                    tgt = torch.zeros_like(score_logp)
                    tgt.scatter_(1, batch["score_bin"][:, None], 1.0)
                    tgt_cdf = tgt.cumsum(dim=1)
                score_cdf_loss = ((score_logp.exp().cumsum(dim=1) - tgt_cdf)
                                  ** 2).mean()
            else:
                score_pdf_loss = torch.zeros((), device=self.device)
                score_cdf_loss = torch.zeros((), device=self.device)

            loss = (self.cfg.policy_weight * policy_loss
                    + self.cfg.value_weight * value_loss
                    + self.cfg.aux_reply_weight * aux_loss
                    + self.cfg.own_weight * own_loss
                    + self.cfg.score_pdf_weight * score_pdf_loss
                    + self.cfg.score_cdf_weight * score_cdf_loss)

            scalars = {
                "loss": loss.detach(),
                "policy": policy_loss.detach(),
                "value": value_loss.detach(),
                "aux": aux_loss.detach(),
                "own": own_loss.detach(),
                "score_pdf": score_pdf_loss.detach(),
                "score_cdf": score_cdf_loss.detach(),
                "policy_acc": policy_acc.detach(),
                "value_acc": (out["wdl"].argmax(-1)
                              == batch["wdl"].argmax(-1)).float().mean(),
            }
        return loss, scalars

    def train_step(self, batch: Dict[str, torch.Tensor],
                   micro: int = 0) -> Dict[str, float]:
        """One optimization step over `batch`. micro>0 splits it into
        micro-sized chunks (grad accumulation) — one opt.step at the
        end; metric scalars are sample-weighted means over the chunks."""
        n = batch["tokens"].shape[0]
        starts = list(range(0, n, micro or n))
        self.opt.zero_grad(set_to_none=True)
        if self.opt_adam is not None:
            self.opt_adam.zero_grad(set_to_none=True)
        acc = None
        for s in starts:
            e = s + (micro or n)
            sub = {k: v[s:e] for k, v in batch.items()}
            loss, sc = self._losses(sub)
            (loss / len(starts)).backward()   # backward OUTSIDE autocast
            w = sub["tokens"].shape[0]
            if acc is None:
                acc = {k: v * w for k, v in sc.items()}
            else:
                for k, v in sc.items():
                    acc[k] += v * w
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.cfg.grad_clip)
        self.opt.step()
        if self.opt_adam is not None:
            self.opt_adam.step()
        self.steps += 1
        return {k: (v / n).item() for k, v in acc.items()}

    # --- eval without training ------------------------------------------

    @torch.no_grad()
    def evaluate(self, tensors: Dict[str, np.ndarray],
                 batch_size: int = 512) -> Dict[str, float]:
        self.model.eval()
        sums = {"policy_acc": 0.0, "value_acc": 0.0, "policy_nll": 0.0}
        n = tensors["tokens"].shape[0]
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            batch = tensors_to_torch({k: v[sl] for k, v in tensors.items()},
                                     self.device)
            out = self.model(batch["tokens"], batch["mask"], batch["aux_mask"])
            sums["policy_acc"] += (out["policy_logp"].argmax(-1)
                                   == batch["pi"].argmax(-1)
                                   ).float().sum().item()
            sums["value_acc"] += (out["wdl"].argmax(-1)
                                  == batch["wdl"].argmax(-1)
                                  ).float().sum().item()
            sums["policy_nll"] += (-(out["policy_logp"]
                                     * batch["pi"]).sum(-1)).sum().item()
        self.model.train()
        return {k: v / n for k, v in sums.items()}

    # --- checkpointing (atomic) ------------------------------------------

    def save(self, path: os.PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._raw_model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "optimizer_adam": (self.opt_adam.state_dict()
                               if self.opt_adam is not None else None),
            "steps": self.steps,
            "config": asdict(self.cfg),
            "net_config": asdict(self._raw_model.cfg),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)             # atomic on the same filesystem
        return path

    @classmethod
    def load(cls, path: os.PathLike, model: Optional[GoNet] = None,
             device: Optional[str] = None, strict: bool = True,
             cfg_overrides: Optional[dict] = None) -> "Trainer":
        payload = torch.load(Path(path), map_location="cpu",
                             weights_only=False)
        # legacy configs may carry removed keys (e.g. optimizer="adam")
        known = TrainConfig.__dataclass_fields__
        cfg = TrainConfig(**{k: v for k, v in payload["config"].items()
                             if k in known})
        if device is not None:
            cfg.device = device
        cfg.device = resolve_device(cfg.device)
        cfg.compile = False    # training-time fact; inference must not
                               # require a build toolchain
        if model is None:
            model = GoNet(NetConfig(**payload["net_config"]))
        # pre-MLP checkpoints stored ff up/down as 1x1-conv [out,in,1,1]
        sd = {k: (v.squeeze(-1).squeeze(-1) if v.dim() == 4
                  and k.endswith(("ff.up.weight", "ff.down.weight")) else v)
              for k, v in payload["model"].items()}
        model.load_state_dict(sd, strict=strict)
        # fp8 tensors cannot move to MPS/CPU backends — ship them bf16:
        # dequantize the FP8 linears and run the whole network bf16
        if payload["net_config"].get("fp8") \
                and torch.device(cfg.device).type != "cuda":
            to_half_(model)
            model.to(torch.float16)
        # caller overrides win (e.g. train_round's --lr/--opt/--batch);
        # applied before __init__ so the optimizer is built to match
        for k, v in (cfg_overrides or {}).items():
            setattr(cfg, k, v)
        trainer = cls(model, cfg)
        if strict and payload.get("optimizer"):
            try:      # pre-Muon checkpoints stored an all-Adam state
                trainer.opt.load_state_dict(payload["optimizer"])
                if trainer.opt_adam is not None \
                        and payload.get("optimizer_adam"):
                    trainer.opt_adam.load_state_dict(
                        payload["optimizer_adam"])
            except (ValueError, RuntimeError):
                print("[trainer] optimizer state not restored (new params)")
        elif strict:
            print("[trainer] optimizer state not restored (new params)")
        trainer.steps = payload["steps"]
        return trainer
