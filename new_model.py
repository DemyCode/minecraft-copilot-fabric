"""
DiT3D — 3-D Diffusion Transformer for masked block diffusion.

Architecture (faithful to the DiT paper, Peebles & Xie 2022):
  1. Embed each voxel token (block id + condition bit).
  2. Patchify non-overlapping (patch_size)³ cubes → flat sequence of N patch tokens.
  3. Add learned 3-D positional embeddings.
  4. Pass through `depth` DiT blocks, each conditioned on the diffusion timestep via
     adaLN-Zero (scale/shift/gate produced by a zero-initialised MLP so the model
     starts as an identity residual).
  5. Final adaLN-conditioned output layer projects each patch token back to
     (patch_size³ × vocab_size) logits.
  6. Unpatchify → [B, vocab_size, D, H, W].

Forward signature: (x_t, condition_mask, t) → logits
Attributes:        .vocab_size, .mask_idx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Time embedding ─────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.pow(10000.0, -torch.arange(0, half).float() / half)
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = (t * 1000.0).float().unsqueeze(1) * self.freqs.unsqueeze(0)
        emb = torch.cat([x.sin(), x.cos()], dim=-1)
        return self.proj(emb)


# ── Attention ──────────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    """Multi-head self-attention using F.scaled_dot_product_attention (flash-attn when available)."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, heads, N, head_dim]
        out = F.scaled_dot_product_attention(q, k, v)    # flash-attn path in PT 2.0+
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ── adaLN-Zero conditioning ────────────────────────────────────────────────────

class AdaLNZero(nn.Module):
    """
    Maps time embedding → 6 per-channel modulation scalars
    (shift1, scale1, gate1, shift2, scale2, gate2).

    The output linear is zero-initialised so gates start at 0 and each block
    is an identity at initialisation (DiT paper §3.2).
    """

    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t_emb: torch.Tensor):
        return self.mlp(t_emb).chunk(6, dim=-1)


# ── DiT block ──────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, time_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_dim, num_heads)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.adaLN = AdaLNZero(time_dim, hidden_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = [
            v.unsqueeze(1) for v in self.adaLN(t_emb)
        ]
        h = self.norm1(x) * (1.0 + scale1) + shift1
        x = x + gate1 * self.attn(h)
        h = self.norm2(x) * (1.0 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x


# ── Final output layer ─────────────────────────────────────────────────────────

class FinalLayer(nn.Module):
    """adaLN-conditioned output projection, zero-initialised (DiT paper §3.2)."""

    def __init__(self, hidden_dim: int, time_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * hidden_dim))
        self.proj = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(t_emb).unsqueeze(1).chunk(2, dim=-1)
        return self.proj(self.norm(x) * (1.0 + scale) + shift)


# ── DiT3D ──────────────────────────────────────────────────────────────────────

class DiT3D(nn.Module):
    """
    3-D Diffusion Transformer.

    Args:
        vocab_size:  Number of block types (mask token is at index vocab_size).
        embed_dim:   Per-voxel block embedding dimension before patchification.
        hidden_dim:  Transformer hidden dimension.
        patch_size:  Side length of each cubic patch (chunk_size must be divisible).
                     patch_size=2 → 16³=4 096 tokens for a 32³ chunk (high quality).
                     patch_size=4 →  8³=  512 tokens for a 32³ chunk (faster).
        depth:       Number of DiT blocks.
        num_heads:   Attention heads (hidden_dim must be divisible).
        mlp_ratio:   MLP hidden dim multiplier inside each block.
        time_dim:    Sinusoidal time embedding dimension.
        chunk_size:  Spatial side length of the input voxel grid.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        patch_size: int = 2,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        time_dim: int = 256,
        chunk_size: int = 32,
    ):
        super().__init__()
        assert chunk_size % patch_size == 0, "chunk_size must be divisible by patch_size"
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.vocab_size = vocab_size
        self.mask_idx = vocab_size
        self.patch_size = patch_size
        self.chunk_size = chunk_size

        p = patch_size
        p3 = p ** 3
        g = chunk_size // p          # patches per axis
        n_patches = g ** 3

        self._p3 = p3
        self._grid = g

        # Per-voxel embedding: vocab_size block types + 1 mask token
        self.block_emb = nn.Embedding(vocab_size + 1, embed_dim)

        # Patch projection: p³ block embeddings + p³ condition bits → hidden_dim
        self.patch_proj = nn.Linear(p3 * (embed_dim + 1), hidden_dim)

        # Learned 3-D positional embeddings (one per patch location)
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, hidden_dim) * 0.02)

        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, time_dim)
            for _ in range(depth)
        ])

        # Output: each patch token → p³ per-voxel logits
        self.final = FinalLayer(hidden_dim, time_dim, p3 * vocab_size)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x_t: torch.Tensor,            # [B, D, H, W]  long
        condition_mask: torch.Tensor,  # [B, D, H, W]  bool
        t: torch.Tensor,              # [B]            float ∈ [0, 1]
    ) -> torch.Tensor:                # [B, vocab_size, D, H, W]
        B, D, H, W = x_t.shape
        p  = self.patch_size
        g  = self._grid
        p3 = self._p3

        # 1. Embed voxels ──────────────────────────────────────────────────────
        emb  = self.block_emb(x_t)                          # [B, D, H, W, embed_dim]
        cond = condition_mask.float().unsqueeze(-1)          # [B, D, H, W, 1]
        x    = torch.cat([emb, cond], dim=-1)               # [B, D, H, W, E+1]
        C    = x.shape[-1]

        # 2. Patchify ──────────────────────────────────────────────────────────
        # [B, D, H, W, C] → [B, g, p, g, p, g, p, C] → [B, g³, p³·C]
        x = x.reshape(B, g, p, g, p, g, p, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(B, g * g * g, p3 * C)

        # 3. Project + positional embed ───────────────────────────────────────
        x = self.patch_proj(x) + self.pos_emb              # [B, N, hidden_dim]

        # 4. Timestep embedding ────────────────────────────────────────────────
        t_emb = self.time_emb(t)                           # [B, time_dim]

        # 5. DiT blocks ────────────────────────────────────────────────────────
        for block in self.blocks:
            x = block(x, t_emb)

        # 6. Output head ───────────────────────────────────────────────────────
        x = self.final(x, t_emb)                           # [B, N, p³·V]

        # 7. Unpatchify ────────────────────────────────────────────────────────
        # [B, g³, p³·V] → [B, g, g, g, p, p, p, V] → [B, V, D, H, W]
        V = self.vocab_size
        x = x.reshape(B, g, g, g, p, p, p, V)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, V, D, H, W)

        return x
