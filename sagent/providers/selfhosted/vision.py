"""Qwen3.6's vision encoder: patch embedding, a ViT, and a patch merger.

The text backbone lives elsewhere; this module holds only the
vision tower and the remap that loads it from a Hugging Face checkpoint.

Typical usage example:

  vision = VisionModel(depth=27, hidden_size=1152, num_heads=16, ...)
  vision.load_state_dict(remap_hf_vision_state_dict(checkpoint), strict=True)
  embeddings = vision(pixel_values, grid_thw)       # (merged patches, out_hidden)
"""

from __future__ import annotations

from typing import override

import math

from torch import Tensor, nn
from torch.nn import functional as f
from treekle import from_plain

import torch


__all__ = [
    "VisionModel",
    "remap_hf_vision_state_dict",
]


class VisionRotaryEmbedding(nn.Module):
    """2-D rotary position frequencies for vision patches.

    Args:
      dim: The rotated width per axis.
      theta: The rotary base.

    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    @override
    def forward(self, seqlen: int, *, device: torch.device) -> Tensor:
        """Return the frequency table, ``(seqlen, dim // 2)``, on ``device``."""
        channels = torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.theta ** (channels / self.dim))
        seq = torch.arange(seqlen, dtype=torch.float32, device=device)
        return torch.outer(seq, inv_freq)


class VisionAttention(nn.Module):
    """Multi-head self-attention, each image attending only within itself.

    Args:
      dim: The token width.
      num_heads: The heads it splits into.

    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    @override
    def forward(
        self,
        x: Tensor,
        cu_seqlens: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        """Attend within each image.

        Args:
          x: Flattened patch tokens, ``(S, C)``.
          cu_seqlens: Cumulative patch counts that split ``x`` into images.
          position_embeddings: The rotary ``(cos, sin)``, ``(S, 1, D // 2)``.

        Returns:
          out: The attention output, ``(S, C)``.

        """
        length = x.shape[0]
        q, k, v = (
            self.qkv(x)
            .reshape(length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        q, k = _rotate(q, cos=cos, sin=sin), _rotate(k, cos=cos, sin=sin)
        lengths = from_plain((cu_seqlens[1:] - cu_seqlens[:-1]).tolist(), list[int])
        outs = [
            f.scaled_dot_product_attention(
                qc.transpose(0, 1).unsqueeze(0),
                kc.transpose(0, 1).unsqueeze(0),
                vc.transpose(0, 1).unsqueeze(0),
            )
            .squeeze(0)
            .transpose(0, 1)
            for qc, kc, vc in zip(
                q.split(lengths),
                k.split(lengths),
                v.split(lengths),
                strict=True,
            )
        ]
        return self.proj(torch.cat(outs, dim=0).reshape(length, -1).contiguous())


class VisionMLP(nn.Module):
    """Two-layer GELU MLP of a vision block."""

    def __init__(self, dim: int, intermediate: int) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(dim, intermediate, bias=True)
        self.linear_fc2 = nn.Linear(intermediate, dim, bias=True)
        self.act = nn.GELU(approximate="tanh")

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Return ``x`` through the MLP."""
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    """Pre-norm vision transformer block: attention, then the MLP."""

    def __init__(self, dim: int, num_heads: int, intermediate: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = VisionAttention(dim, num_heads)
        self.mlp = VisionMLP(dim, intermediate)

    @override
    def forward(
        self,
        x: Tensor,
        cu_seqlens: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        """Return ``x`` after attention and the MLP, each with its residual."""
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        return x + self.mlp(self.norm2(x))


class VisionPatchMerger(nn.Module):
    """Merges each square of patches into one wider token."""

    def __init__(self, hidden_size: int, merge_size: int, out_hidden_size: int) -> None:
        super().__init__()
        merged = hidden_size * merge_size * merge_size
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(merged, merged)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(merged, out_hidden_size)
        self._merged = merged

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Return each square of patches as one token, ``(S / merge**2, out)``."""
        x = self.norm(x).view(-1, self._merged)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionModel(nn.Module):
    """Qwen3.6's vision encoder: patch embedding, a ViT, and a patch merger.

    Args:
      depth: Transformer blocks.
      hidden_size: Token width inside the encoder.
      num_heads: Attention heads per block.
      intermediate_size: MLP width per block.
      in_channels: Pixel channels.
      patch_size: Patch edge, in pixels.
      temporal_patch_size: Frames per patch.
      spatial_merge_size: Patches a side the merger joins into one token.
      out_hidden_size: Width of the merged tokens, the text model's.
      num_position_embeddings: Learned positions, a square grid of them.
      initializer_range: Standard deviation of the initial weights.

    """

    def __init__(
        self,
        *,
        depth: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        in_channels: int,
        patch_size: int,
        temporal_patch_size: int,
        spatial_merge_size: int,
        out_hidden_size: int,
        num_position_embeddings: int,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        self.initializer_range = initializer_range
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        kernel = (temporal_patch_size, patch_size, patch_size)
        self.patch_embed = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=kernel,
            stride=kernel,
            bias=True,
        )
        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = math.isqrt(num_position_embeddings)
        self.rotary_pos_emb = VisionRotaryEmbedding(hidden_size // num_heads // 2)
        self.blocks = nn.ModuleList(
            [
                VisionBlock(hidden_size, num_heads, intermediate_size)
                for _ in range(depth)
            ],
        )
        self.merger = VisionPatchMerger(
            hidden_size,
            spatial_merge_size,
            out_hidden_size,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Draw every weight anew, as the reference initializes it."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv3d, nn.Embedding)):
                nn.init.normal_(module.weight, std=self.initializer_range)
            if isinstance(module, (nn.Linear, nn.Conv3d)) and module.bias is not None:
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()

    @override
    def forward(self, pixel_values: Tensor, grid_thw: Tensor) -> Tensor:
        """Encode images into merged patch embeddings.

        Args:
          pixel_values: Each patch's pixels, flattened, one row a patch.
          grid_thw: Each image's ``(T, H, W)`` patch grid, ``(N, 3)``.

        Returns:
          embeddings: The merged tokens, ``(patches / merge**2, out_hidden_size)``.

        Raises:
          ValueError: A grid's height or width is not a multiple of the merge.

        """
        grids = [from_plain(row, list[int]) for row in grid_thw.tolist()]
        merge = self.spatial_merge_size
        for frames, height, width in grids:
            # A grid the merger cannot tile would leave ``empty`` positions
            # unwritten, which attention then reads as garbage.
            if height % merge or width % merge:
                raise ValueError(
                    f"A grid's height and width must be multiples of"
                    f" spatial_merge_size={merge}; got (t={frames}, h={height},"
                    f" w={width}).",
                )
        x = self._patches(pixel_values) + self._positions(grids)
        rotary = self._rotary(grids).reshape(x.shape[0], -1)
        position_embeddings = (rotary.cos().unsqueeze(-2), rotary.sin().unsqueeze(-2))
        cu_seqlens = f.pad(
            torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2],
                grid_thw[:, 0],
            ).cumsum(dim=0, dtype=torch.int32),
            (1, 0),
            value=0,
        )
        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        return self.merger(x)

    def _patches(self, pixel_values: Tensor) -> Tensor:
        """Embed each patch's pixels with the patch convolution."""
        x = pixel_values.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        embedded = self.patch_embed(x.to(self.patch_embed.weight.dtype))
        return embedded.view(-1, self.hidden_size)

    def _rotary(self, grids: list[list[int]]) -> Tensor:
        """Return each patch's 2-D rotary frequencies, in the merger's order."""
        merge = self.spatial_merge_size
        device = self.pos_embed.weight.device
        table = self.rotary_pos_emb(max(max(h, w) for _, h, w in grids), device=device)
        ids: list[Tensor] = []
        for frames, height, width in grids:
            rows, cols = height // merge, width // merge
            inner = torch.arange(merge, device=device)
            row = (
                torch.arange(rows, device=device)[:, None, None, None] * merge
                + inner[None, None, :, None]
            ).expand(rows, cols, merge, merge)
            col = (
                torch.arange(cols, device=device)[None, :, None, None] * merge
                + inner[None, None, None, :]
            ).expand(rows, cols, merge, merge)
            coords = torch.stack((row.reshape(-1), col.reshape(-1)), dim=-1)
            ids.append(coords.repeat(frames, 1))
        return table[torch.cat(ids)].flatten(1)

    def _positions(self, grids: list[list[int]]) -> Tensor:
        """Interpolate the learned positions bilinearly onto each image's grid."""
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype
        side = self.num_grid_per_side
        merge = self.spatial_merge_size
        indices: list[list[int]] = [[] for _ in range(4)]
        weights: list[list[float]] = [[] for _ in range(4)]
        for _, height, width in grids:
            hi = torch.linspace(0, side - 1, height)
            wi = torch.linspace(0, side - 1, width)
            hf, wf = hi.int(), wi.int()
            hc, wc = (hf + 1).clip(max=side - 1), (wf + 1).clip(max=side - 1)
            dh, dw = hi - hf, wi - wf
            corners = (
                (hf * side, wf, (1 - dh), (1 - dw)),
                (hf * side, wc, (1 - dh), dw),
                (hc * side, wf, dh, (1 - dw)),
                (hc * side, wc, dh, dw),
            )
            for i, (base, offset, row_weight, col_weight) in enumerate(corners):
                indices[i] += from_plain(
                    (base[None].T + offset[None]).flatten().tolist(),
                    list[int],
                )
                weights[i] += from_plain(
                    (row_weight[None].T * col_weight[None]).flatten().tolist(),
                    list[float],
                )
        index = torch.tensor(indices, dtype=torch.long, device=device)
        weight = torch.tensor(weights, dtype=dtype, device=device)
        embedded = self.pos_embed(index) * weight[:, :, None]
        # Summed in this order, as the reference sums them: a reduction of
        # another order rounds differently.
        per_patch = embedded[0] + embedded[1] + embedded[2] + embedded[3]
        chunks = per_patch.split([h * w for _, h, w in grids])
        return torch.cat(
            [
                chunk.repeat(frames, 1)
                .view(frames, height // merge, merge, width // merge, merge, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
                for chunk, (frames, height, width) in zip(chunks, grids, strict=True)
            ],
        )


def remap_hf_vision_state_dict(hf_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return a Hugging Face checkpoint's vision weights, keyed for ``VisionModel``.

    Args:
      hf_sd: The checkpoint's state dict; weights of the text model are skipped.

    Returns:
      sd: The vision encoder's weights, ready for ``load_state_dict``.

    """
    out: dict[str, Tensor] = {}
    vp = "model.visual"
    out["patch_embed.weight"] = hf_sd[f"{vp}.patch_embed.proj.weight"]
    out["patch_embed.bias"] = hf_sd[f"{vp}.patch_embed.proj.bias"]
    out["pos_embed.weight"] = hf_sd[f"{vp}.pos_embed.weight"]
    for part in ("linear_fc1", "linear_fc2", "norm"):
        for kind in ("weight", "bias"):
            out[f"merger.{part}.{kind}"] = hf_sd[f"{vp}.merger.{part}.{kind}"]
    blocks = {
        int(key.split(".")[3]) for key in hf_sd if key.startswith(f"{vp}.blocks.")
    }
    parts = (
        "norm1",
        "norm2",
        "attn.qkv",
        "attn.proj",
        "mlp.linear_fc1",
        "mlp.linear_fc2",
    )
    for i in sorted(blocks):
        for part in parts:
            for kind in ("weight", "bias"):
                out[f"blocks.{i}.{part}.{kind}"] = hf_sd[
                    f"{vp}.blocks.{i}.{part}.{kind}"
                ]
    return out


# The pairing is Hugging Face's, which the checkpoint was trained with; the rotation
# runs in float32 and returns ``x``'s dtype.
def _rotate(x: Tensor, *, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x``'s half-split channel pairs (``i``, ``i + D/2``) by the table."""
    dtype = x.dtype
    x = x.float()
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    rotated = torch.cat(
        [first * cos - second * sin, second * cos + first * sin],
        dim=-1,
    )
    return rotated.to(dtype)
