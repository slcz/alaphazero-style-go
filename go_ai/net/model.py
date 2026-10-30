"""
go_ai/net/model.py — the Go transformer (trunk ported from the chess
engine's ChessTransformer; heads and RoPE are Go-specific).

    tokens [B, b*b, 18]                  (go_nn.h layout, C-encoded)
      │ input projection  Linear(18 → d)   (no absolute position embedding)
      ▼
    8 × encoder blocks
        ├─ LN → multi-head attention (8 heads): axial 2D RoPE on q/k
        │      (parameter-free; scores depend on (Δx, Δy) only)
        └─ LN → up-GEMM (d → 4d) → 3x3 depthwise → GELU
           → down-GEMM (4d → d) → residual  (channels-last dw: the
             layout permutes are zero-copy views; fp8 GEMMs at inference)
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
    fp8: bool = False         # E4M3 weight quantization (inference)


# ---------------------------------------------------------------------------
# Axial 2D RoPE (dirs = 2: x and y banks; parameter-free, cached per size)
# ---------------------------------------------------------------------------

_rope_cache: dict = {}


def _rope_tables(b: int, pairs: int, lam0: float, device, dtype, d_head: int):
    """Full-width tables [b*b, d_head]: cos/sin per dim; passthrough dims
    carry (1, 0). Within a bank the sign is folded into sin so the
    rotation reduces to x·cos + roll(x)·sin (no cat, no per-pair loop)."""
    key = (b, pairs, lam0, str(device), dtype, d_head)
    t = _rope_cache.get(key)
    if t is None:
        # built on CPU in float64 (exact); MPS has no float64 — the
        # tables are tiny and cached, the move is free
        idx = torch.arange(b * b, dtype=torch.float64)
        x, y = idx % b, idx // b
        theta = 2 * math.pi / (lam0 * 2.0 ** torch.arange(pairs))
        ang = torch.cat(
            [x[:, None] * theta[None, :], y[:, None] * theta[None, :]],
            dim=1)                                     # [T, 2p]
        c = torch.ones(b * b, d_head, dtype=torch.float64)
        s = torch.zeros(b * b, d_head, dtype=torch.float64)
        p = pairs
        for m in (0, 1):                               # build-time only
            a = ang[:, m * p:(m + 1) * p]
            sl = slice(m * 2 * p, m * 2 * p + 2 * p)
            c[:, sl] = torch.cat([a.cos(), a.cos()], dim=1)
            s[:, sl] = torch.cat([-a.sin(), a.sin()], dim=1)
        t = (c.to(device=device, dtype=dtype),
             s.to(device=device, dtype=dtype))
        _rope_cache[key] = t
    return t


def apply_rope(x: torch.Tensor, b: int, pairs: int, lam0: float) -> torch.Tensor:
    """Rotate x [..., b*b, d_head]: dims [0, 2p) = x bank, [2p, 4p) = y
    bank (half-split pairing inside each bank — the GPT-NeoX convention);
    the rest is unrotated content. q and k use the same tables."""
    assert 4 * pairs <= x.shape[-1], "axial RoPE needs 4·pairs ≤ d_head"
    p = pairs
    cos, sin = _rope_tables(b, p, lam0, x.device, x.dtype, x.shape[-1])
    # out = x·cos + roll_within_bank(x)·sin — 4-5 kernels total; the two
    # banks form a (2, 2p) view so one roll rotates each independently
    out = x * cos
    x4 = x[..., :4 * p].unflatten(-1, (2, 2 * p))
    out[..., :4 * p] += (x4.roll(p, dims=-1)
                         * sin[..., :4 * p].unflatten(-1, (2, 2 * p))
                         ).flatten(-2)
    return out


class FP8Linear(nn.Module):
    """E4M3 weight-quantized linear. CUDA computes through _scaled_mm
    with FP8 activations: x is cast DIRECTLY (scale_a = 1) — every
    quantized linear consumes LN-bounded input (|x| <= sqrt(T) ~ 19,
    far under E4M3's 448), so no activation scale or clamp is needed;
    fp8 relative precision is scale-invariant. Elsewhere a dequantized
    matmul runs as the reference path (tests, MPS)."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.data.float()
        self.scale = nn.Buffer((w.abs().amax() / 448.0).clamp(min=1e-12))
        self.wq = nn.Buffer((w / self.scale).clamp(-448, 448)
                            .to(torch.float8_e4m3fn))
        self.one = nn.Buffer(torch.ones(()))       # scale_a, rides .to()
        # bf16 bias: an fp32 bias would promote _scaled_mm's bf16 output
        # to fp32 and break the flash-attention dtype contract downstream
        self.bias = (nn.Parameter(linear.bias.data.to(torch.bfloat16))
                     if linear.bias is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.is_cuda:
            # cast saturating-free: inputs are LN-bounded (see class doc)
            xq = x2.to(torch.float8_e4m3fn)
            y = torch._scaled_mm(xq, self.wq.t(), scale_a=self.one,
                                 scale_b=self.scale, bias=self.bias,
                                 out_dtype=torch.bfloat16,
                                 use_fast_accum=True)
        else:
            # non-CUDA (M4 play, CPU tests): dequant to bf16 once — the
            # FP8 checkpoint then runs as a plain bf16 network
            wdq = getattr(self, "_wdq", None)
            if wdq is None or wdq.device != x2.device:
                wdq = (self.wq.to(torch.bfloat16)
                       * self.scale.to(torch.bfloat16))
                object.__setattr__(self, "_wdq", wdq)   # cache only
            y = F.linear(x2.to(wdq.dtype), wdq, self.bias)
        # keep the stream dtype stable (fp32 tests, bf16 serving)
        return y.reshape(*shape[:-1], -1).to(x2.dtype)


def to_fp8_(root: nn.Module) -> None:
    """Quantize Linear weights to E4M3 where cuBLASLt allows it (K and N
    multiples of 16 — the token projection and the d→1 head projections
    stay bf16); the depthwise conv and norms stay as-is."""
    for name, child in root.named_children():
        if isinstance(child, nn.Linear):
            if child.in_features % 16 == 0 and child.out_features % 16 == 0:
                setattr(root, name, FP8Linear(child))
        else:
            to_fp8_(child)


def to_half_(root: nn.Module) -> None:
    """Replace FP8Linears with plain fp16 Linears (dequantized once).
    Non-CUDA targets (MPS) cannot hold the fp8 dtype at all, and MPS
    bf16 matmul asserts on the fused bias — fp16 is the stable fast
    path (same 2x rate). Conversion happens on CPU at load time."""

    def conv(m: FP8Linear) -> nn.Linear:
        lin = nn.Linear(m.wq.shape[1], m.wq.shape[0], bias=m.bias is not None)
        with torch.no_grad():
            lin.weight.data = (m.wq.float() * m.scale).to(torch.float16)
            if m.bias is not None:
                lin.bias.data = m.bias.data
        return lin

    for name, child in root.named_children():
        if isinstance(child, FP8Linear):
            setattr(root, name, conv(child))
        else:
            to_half_(child)


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
        if q.is_cuda and not torch.compiler.is_compiling():
            # flash never materializes the T×T scores; the math fallback
            # needs B·h·T² bf16 live (17 GB/layer at B=8192) — OOM/hang.
            # under compile the context breaks dynamo's fake-tensor
            # dispatch — inductor picks flash itself
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, -1))


class ConvFFN(nn.Module):
    """MBConv-style FFN: up/down MLP linears, 3x3 depthwise conv between
    them (the Go-locality prior; ~2% of FLOPs). The dw conv runs
    channels_last so the [B,N,C] <-> [B,C,H,W] transitions are zero-copy
    views — and under torch.compile the surrounding norm/cast/GELU fuse
    into neighbor kernels. Zero-init down: the branch starts as the
    identity."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.dw = nn.Conv2d(d_ff, d_ff, 3, padding=1, groups=d_ff)
        self.dw.to(memory_format=torch.channels_last)
        self.down = nn.Linear(d_ff, d_model)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        b = math.isqrt(N)
        h = self.up(x).view(B, b, b, -1).permute(0, 3, 1, 2)
        h = F.gelu(F.conv2d(h, self.dw.weight.to(h.dtype),
                            self.dw.bias.to(h.dtype), padding=1,
                            groups=self.dw.groups))
        h = h.permute(0, 2, 3, 1).reshape(B, N, -1)
        return self.down(h)


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
        if cfg.fp8:
            to_fp8_(self)

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
