"""Supra2-IMG DiT, matching the Hub ``inference.py`` checkpoint contract.

Port of the architecture shipped at
https://huggingface.co/SupraLabs/Supra2-IMG (``inference.py`` +
``model_final_ema.pt``). Defaults are the live card (~104.1M). Tests and
``--dummy`` construct a smaller ``SupraDiT(...)``; ``SupraDiT()`` is the
checkpoint shape.

No Hub download in this module.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 256
LATENT_SIZE = 32  # 256 / 8 (f8 VAE)
LATENT_CH = 4
PATCH = 2
NUM_TOKENS = (LATENT_SIZE // PATCH) ** 2  # 256

D_MODEL = 576
DEPTH = 14
N_HEADS = 9
HEAD_DIM = 64
MLP_RATIO = 4.0
D_CTX = 768  # Flan-T5-Base
MAX_CTX_LEN = 128
T5_NAME = "google/flan-t5-base"
VAE_NAME = "stabilityai/sd-vae-ft-mse"
VAE_SCALE = 0.18215

HF_REPO = "SupraLabs/Supra2-IMG"
CKPT_FILENAME = "model_final_ema.pt"
# ``SupraDiT()`` numel. Hub card rounds this to 104.1M.
SUPRA_PARAM_COUNT = 104_094_736


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN: ``x * (1 + scale) + shift``."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None] * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))


class Attention(nn.Module):
    """Multi-head self- or cross-attention (Hub ``inference.py`` names).

    Self-attn linears: ``qkv``, ``proj``.
    Cross-attn linears: ``q``, ``kv``, ``proj``.
    ``proj`` is shared by both, so LoRA targets must be path suffixes
    (``self_attn.proj`` vs ``cross_attn.proj``), not a bare ``proj``.
    """

    def __init__(self, dim: int, n_heads: int, ctx_dim: int | None = None) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} must divide n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.is_self = ctx_dim is None
        if self.is_self:
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
        else:
            self.q = nn.Linear(dim, dim, bias=True)
            self.kv = nn.Linear(ctx_dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor | None = None,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        if self.is_self:
            qkv = self.qkv(x).view(batch, tokens, 3, self.n_heads, self.head_dim)
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        else:
            if ctx is None:
                raise ValueError("cross-attention requires ctx")
            length = ctx.shape[1]
            q = self.q(x).view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)
            kv = self.kv(ctx).view(batch, length, 2, self.n_heads, self.head_dim)
            k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)

        attn_mask = None
        if ctx_mask is not None:
            attn_mask = ctx_mask.bool()[:, None, None, :]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(out)


class DiTBlock(nn.Module):
    """AdaLN-Zero self-attn + cross-attn + MLP."""

    def __init__(self, dim: int, n_heads: int, ctx_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        del ctx_dim  # Hub script projects context to ``dim`` before the block.
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, n_heads)
        self.norm_ca = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(dim, n_heads, ctx_dim=dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.adaln(c).chunk(
            6, dim=1
        )
        x = x + gate_sa.unsqueeze(1) * self.self_attn(
            modulate(self.norm1(x), shift_sa, scale_sa)
        )
        x = x + self.cross_attn(self.norm_ca(x), ctx=ctx, ctx_mask=ctx_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_ch: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_ch, bias=True)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class SupraDiT(nn.Module):
    """Rectified-flow text-to-image DiT. Default ctor matches the Hub card."""

    def __init__(
        self,
        latent_ch: int = LATENT_CH,
        d_model: int = D_MODEL,
        depth: int = DEPTH,
        n_heads: int = N_HEADS,
        ctx_dim: int = D_CTX,
        mlp_ratio: float = MLP_RATIO,
        num_tokens: int = NUM_TOKENS,
        patch: int = PATCH,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must divide n_heads {n_heads}")
        if patch < 1:
            raise ValueError(f"patch must be >= 1, got {patch}")
        self.num_tokens = int(num_tokens)
        self.patch = int(patch)
        self.latent_ch = int(latent_ch)
        self.x_embed = nn.Linear(latent_ch * self.patch * self.patch, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))
        self.t_embed = TimestepEmbedder(d_model)
        self.ctx_proj = nn.Linear(ctx_dim, d_model)
        self.blocks = nn.ModuleList(
            [DiTBlock(d_model, n_heads, d_model, mlp_ratio) for _ in range(int(depth))]
        )
        self.final = FinalLayer(d_model, latent_ch * self.patch * self.patch)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``z`` is ``[B,C,H,W]`` latent. ``t`` is in ``[0, 1]`` (Hub Euler)."""
        batch, _channels, height, width = z.shape
        patch = self.patch
        if height % patch or width % patch:
            raise ValueError(
                f"latent spatial {(height, width)} must be divisible by patch {patch}"
            )
        grid_h, grid_w = height // patch, width // patch
        tokens = z.view(batch, self.latent_ch, grid_h, patch, grid_w, patch)
        tokens = tokens.permute(0, 2, 4, 1, 3, 5).reshape(
            batch, grid_h * grid_w, self.latent_ch * patch * patch
        )
        if tokens.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"token count {tokens.shape[1]} != pos_embed {self.pos_embed.shape[1]} "
                "(construct SupraDiT with num_tokens=(H/patch)*(W/patch))"
            )
        hidden = self.x_embed(tokens) + self.pos_embed
        cond = self.t_embed(t)
        context = self.ctx_proj(ctx)
        for block in self.blocks:
            hidden = block(hidden, cond, context, ctx_mask)
        hidden = self.final(hidden, cond)
        restored = hidden.view(batch, grid_h, grid_w, self.latent_ch, patch, patch)
        return restored.permute(0, 3, 1, 4, 2, 5).reshape(batch, self.latent_ch, height, width)


def supra_weights_from_checkpoint(state: Any) -> dict[str, torch.Tensor]:
    """Unwrap Hub ``model_final_ema.pt`` (``ema`` or ``model`` or raw)."""
    if isinstance(state, dict) and "ema" in state:
        weights = state["ema"]
        cfg = state.get("config") or {}
        patch = cfg.get("patch", PATCH) if isinstance(cfg, dict) else PATCH
        if patch != PATCH:
            raise ValueError(f"Checkpoint PATCH={patch} != script PATCH={PATCH}")
    elif isinstance(state, dict) and "model" in state and not _looks_like_dit_state(state):
        weights = state["model"]
    else:
        weights = state
    if not isinstance(weights, dict):
        raise ValueError(f"Supra checkpoint weights must be a state dict, got {type(weights)}")
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in weights.items():
        name = str(key)
        if name.startswith("module."):
            name = name[len("module.") :]
        cleaned[name] = value
    return cleaned


def _looks_like_dit_state(state: dict) -> bool:
    return any(str(key).startswith(("blocks.", "x_embed.", "ctx_proj.")) for key in state)


def load_supra_weights(model: SupraDiT, state: Any, *, strict: bool = True) -> SupraDiT:
    weights = supra_weights_from_checkpoint(state)
    model.load_state_dict(weights, strict=strict)
    return model


class LoRALinear(nn.Module):
    """Zero-init up-projection so step 0 matches the frozen base linear."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        self.scale = float(alpha) / float(rank)
        self.multiplier = 1.0
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.down(x))
        return self.base(x) + self.multiplier * self.scale * delta


def _module_by_path(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent: Any = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def attach_supra_lora(
    root: nn.Module,
    rank: int,
    alpha: float,
    targets: tuple[str, ...] | list[str],
) -> list[LoRALinear]:
    """Wrap path-suffix linears. Bare ``proj`` is rejected (self/cross collide).

    Base weights are frozen. Only ``down`` / ``up`` stay trainable.
    """
    target_names = tuple(targets)
    if any(name in {"proj", "q"} for name in target_names):
        raise ValueError(
            "Supra LoRA targets must be path suffixes "
            "(cross_attn.proj / self_attn.proj); bare 'proj' or 'q' hits both attns"
        )
    named = dict(root.named_modules())
    wrapped: list[LoRALinear] = []
    for target in target_names:
        matches = [
            name for name in named if name == target or name.endswith("." + target)
        ]
        if not matches:
            raise RuntimeError(
                f"Supra LoRA target {target!r} not found on {type(root).__name__}"
            )
        for name in matches:
            parent, attr = _module_by_path(root, name)
            base = getattr(parent, attr) if not attr.isdigit() else parent[int(attr)]
            if isinstance(base, LoRALinear):
                wrapped.append(base)
                continue
            if not isinstance(base, nn.Linear):
                raise RuntimeError(f"{name} is {type(base).__name__}, expected Linear")
            lora = LoRALinear(base, rank=rank, alpha=alpha)
            if attr.isdigit():
                parent[int(attr)] = lora
            else:
                setattr(parent, attr, lora)
            wrapped.append(lora)
    for param in root.parameters():
        param.requires_grad_(False)
    for lora in wrapped:
        lora.down.weight.requires_grad_(True)
        lora.up.weight.requires_grad_(True)
    return wrapped
