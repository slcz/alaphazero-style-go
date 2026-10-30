"""
go_ai/net/model.py — the Go transformer (trunk ported from the chess
engine's ChessTransformer; heads and RoPE are Go-specific).

    tokens [B, b*b, 18]                  (go_nn.h layout, C-encoded)
      │ input projection  Linear(18 → d)   (no absolute position embedding)
      ▼
    8 × pre-LN encoder blocks
        ├─ LN → multi-head attention (8 heads): axial 2D RoPE on q/k
        │      (parameter-free; scores depend on (Δx, Δy) only)
        └─ LN → conv FFN: 1x1 (d → 4d) → 3x3 depthwise → GELU
                                    → 1x1 (4d → d)   (MBConv-style; the
                                      depthwise adds a Go-locality prior)
      │ final LN
      ├─ policy head: Linear(d→1) per point + a pass logit from the
      │    mean-pool → b*b+1 logits (pass = slot b*b)
      ├─ value head: mean-pool → MLP → 2 logits [pW, pL]; the scalar the
      │    search consumes is v = pW (= 1 at pure win)
      ├─ ownership head: Linear(d→1) per point (no pass) — mover-view
      │    own-probability via sigmoid in the loss
      ├─ score head: mean-pool → MLP → SCORE_BINS logits (1-point bins
      │    covering ±380, size-agnostic: the same head serves 7x7..19x19)
      └─ aux opponent-reply head: a second policy head (KataGo aux
           target), same masking contract

Board-size adaptivity: tokens are per-point with relative positions only
(RoPE uses raw integer coordinates so distance-1 = adjacency at every
size), so the same weights run on any square board — T = b*b tokens,
b = isqrt(T). Illegal-move masking is the module's hard contract:
log-probabilities are log_softmax over legal slots only, illegal slots
exactly zero (superko legality is handled HERE, at the output).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Score belief: 1-point bins from -380..+380 (covers 19x19 + komi).
SCORE_MIN = -380.0
SCORE_BINS = 761


@dataclass
class NetConfig:
    token_dim: int = 18
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    aux_reply: bool = True
    rope_pairs: int = 4       # 2 banks (x, y) × 4 pairs = 16 of 32 head
                              # dims rotated; λ = 8,16,32,64 — the short
                              # pairs buy near-origin resolution (adjacent
                              # points), the long ones stay alias-free at 19
    rope_lambda: float = 8.0


# ---------------------------------------------------------------------------
# Axial 2D RoPE (dirs = 2: x and y banks; parameter-free, cached per size)
# ---------------------------------------------------------------------------

_rope_cache: dict = {}


def _rope_tables(b: int, pairs: int, lam0: float, device, dtype):
    key = (b, pairs, lam0, str(device), dtype)
    t = _rope_cache.get(key)
    if t is None:
        idx = torch.arange(b * b, device=device, dtype=torch.float64)
        x = idx % b
        y = idx // b
        theta = 2 * math.pi / (lam0 * 2.0 ** torch.arange(pairs,
                                                          device=device))
        # [b*b, 2*pairs]: x bank then y bank
        ang = torch.cat(
            [x[:, None] * theta[None, :], y[:, None] * theta[None, :]],
            dim=1)
        t = (ang.cos().to(dtype), ang.sin().to(dtype))
        _rope_cache[key] = t
    return t


def apply_rope(x: torch.Tensor, b: int, pairs: int, lam0: float) -> torch.Tensor:
    """Rotate x [..., b*b, d_head]: dims [0, 2p) = x bank, [2p, 4p) = y
    bank (half-split pairing inside each bank — the GPT-NeoX convention);
    the rest is unrotated content. q and k use the same tables."""
    assert 4 * pairs <= x.shape[-1], "axial RoPE needs 4·pairs ≤ d_head"
    p = pairs
    cos, sin = _rope_tables(b, p, lam0, x.device, x.dtype)
    banks = []
    for m in range(2):
        xb = x[..., m * 2 * p:(m + 1) * 2 * p]
        x1, x2 = xb.chunk(2, dim=-1)
        cb, sb = cos[:, m * p:(m + 1) * p], sin[:, m * p:(m + 1) * p]
        banks.append(torch.cat([x1 * cb - x2 * sb, x2 * cb + x1 * sb], dim=-1))
    return torch.cat(banks + [x[..., 4 * p:]], dim=-1)


class RopeAttention(nn.Module):
    """Plain multi-head attention with axial 2D RoPE on q/k; residual
    branch zero-init (each block starts as the identity)."""

    def __init__(self, cfg: NetConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.pairs = cfg.rope_pairs
        self.lam0 = cfg.rope_lambda
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        b = math.isqrt(T)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = apply_rope(q.view(B, T, self.h, self.dh).transpose(1, 2),
                       b, self.pairs, self.lam0)
        k = apply_rope(k.view(B, T, self.h, self.dh).transpose(1, 2),
                       b, self.pairs, self.lam0)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        if q.is_cuda:
            # flash never materializes the T×T scores; the math fallback
            # needs B·h·T² bf16 live (17 GB/layer at B=8192) — OOM/hang
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, -1))


class ConvFFN(nn.Module):
    """MBConv-style FFN: the usual up/down projections as 1x1 convs with
    a 3x3 depthwise conv between them — a Go-locality prior (contacts,
    liberties, captures) in the residual branch. The depthwise adds only
    9·d_ff params and ~2% FLOPs per layer; size-agnostic (padding=1
    slides over any board, zero border padding matches the edge-distance
    features). Zero-init projection: the branch starts as the identity."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up = nn.Conv2d(d_model, d_ff, 1)
        self.dw = nn.Conv2d(d_ff, d_ff, 3, padding=1, groups=d_ff)
        self.down = nn.Conv2d(d_ff, d_model, 1)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        b = math.isqrt(T)
        h = x.transpose(1, 2).reshape(B, C, b, b)
        h = F.gelu(self.dw(self.up(h)))
        h = self.down(h)
        return h.reshape(B, C, T).transpose(1, 2)


class EncoderBlock(nn.Module):
    """Pre-LN transformer encoder block (chess trunk; the FFN is the Go
    conv variant above)."""

    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = RopeAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = ConvFFN(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """Linear(d→1) per point + a pass logit from the mean pool → b²+1."""

    def __init__(self, d_model: int):
        super().__init__()
        self.pts = nn.Linear(d_model, 1)
        self.pas = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.pts(x).squeeze(-1)                 # [B, b*b]
        return torch.cat([logits, self.pas(x.mean(1))], dim=-1)  # [B, b²+1]


class ValueHead(nn.Module):
    """2-way win/loss (no tie: fractional komi); v = pW."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.mean(dim=1))                   # [B, 2] [pW, pL]


class OwnershipHead(nn.Module):
    """Linear(d→1) per point, no pass — mover-view own logit."""

    def __init__(self, d_model: int):
        super().__init__()
        self.lin = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).squeeze(-1)                   # [B, b*b]


class ScoreHead(nn.Module):
    """Mean-pool → MLP → SCORE_BINS logits (pdf target; cdf derives)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, SCORE_BINS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.mean(dim=1))                   # [B, SCORE_BINS]


def score_to_bin(score_mover: float) -> int:
    """Mover-view margin → bin index (clamped at the ±380 rails)."""
    return min(max(int(round(score_mover - SCORE_MIN)), 0), SCORE_BINS - 1)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class GoNet(nn.Module):
    """Encoder-only trunk + Go heads.

    forward(tokens, mask, aux_mask) → dict with:
      "policy_logp" [B, b²+1]  log-probs, illegal slots = −inf (prob
                              exactly 0 — the hard contract; superko
                              masking lives here)
      "wdl"         [B, 2]    [pW, pL] logits (apply softmax in the loss)
      "value"       [B]       scalar v = pW ∈ [0, 1]
      "own"         [B, b²]   ownership logits (mover view, sigmoid in loss)
      "score"       [B, SCORE_BINS]  score-belief logits (mover view)
      "aux_logp"    [B, b²+1] opponent-reply head, same masking contract
                              (only when NetConfig.aux_reply)

    `mask`/`aux_mask` are bool [B, 362] (row stride = max board; only the
    first b²+1 slots are meaningful). Rows must have ≥1 legal slot.
    """

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or NetConfig()
        self.input_proj = nn.Linear(cfg.token_dim, cfg.d_model)
        self.blocks = nn.ModuleList(EncoderBlock(cfg)
                                    for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.policy = PolicyHead(cfg.d_model)
        self.value = ValueHead(cfg.d_model)
        self.own = OwnershipHead(cfg.d_model)
        self.score = ScoreHead(cfg.d_model)
        self.aux = PolicyHead(cfg.d_model) if cfg.aux_reply else None

    def trunk(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(tokens)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    @staticmethod
    def _masked_logp(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """log_softmax over legal slots only; illegal = −inf. The mask
        row stride is POLICY_MAX; slice to this board's active width. The
        mask is non-empty by construction (pass is always legal unless the
        game is over, and the engine never evaluates terminal positions)."""
        mask = mask[:, : logits.size(-1)]
        neg_inf = torch.finfo(logits.dtype).min
        return F.log_softmax(logits.masked_fill(~mask, neg_inf), dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        aux_mask: torch.Tensor | None = None,
    ) -> dict:
        x = self.trunk(tokens)
        out = {
            "policy_logp": self._masked_logp(self.policy(x), mask),
            "wdl": self.value(x),
            "own": self.own(x),
            "score": self.score(x),
        }
        w = F.softmax(out["wdl"], dim=-1)
        out["value"] = w[:, 0]                    # v = pW
        if self.aux is not None:
            if aux_mask is None:
                aux_mask = mask
            out["aux_logp"] = self._masked_logp(self.aux(x), aux_mask)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
