"""Tests for Qwen3.6's vision encoder, on a tiny configuration."""

from __future__ import annotations

import math
import re

from torch import Tensor, nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

import pytest
import torch

from sagent.providers.selfhosted import vision
from sagent.providers.selfhosted.vision import (
    VisionModel,
    VisionRotaryEmbedding,
    remap_hf_vision_state_dict,
)


def _model(*, depth: int = 1, std: float = 0.02) -> VisionModel:
    return VisionModel(
        depth=depth,
        hidden_size=8,
        num_heads=2,
        intermediate_size=8,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=6,
        num_position_embeddings=16,
        initializer_range=std,
    )


def _pixels(grid: Tensor, *, seed: int = 0) -> Tensor:
    patches = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    return torch.randn(
        patches,
        3 * 1 * 2 * 2,
        generator=torch.Generator().manual_seed(seed),
    )


@pytest.mark.parametrize("std", [0.02, 0.07])
def test_weights_start_as_the_reference_draws_them(
    std: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drawn: list[float] = []
    normal = nn.init.normal_

    def record(
        tensor: Tensor,
        mean: float = 0.0,
        std: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        drawn.append(std)
        return normal(tensor, mean=mean, std=std, generator=generator)

    monkeypatch.setattr(nn.init, "normal_", record)
    model = _model(std=std)
    weighted = [
        m
        for m in model.modules()
        if isinstance(m, (nn.Linear, nn.Conv3d, nn.Embedding))
    ]
    # The last draws are the reference's; torch's own come first.
    assert drawn[-len(weighted) :] == [std] * len(weighted)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            assert module.bias is not None
            assert torch.count_nonzero(module.bias) == 0
        elif isinstance(module, nn.LayerNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert module.bias is not None
            assert torch.count_nonzero(module.bias) == 0
    # Drawn again, the weights change and the biases return to zero.
    linear, norm = model.merger.linear_fc1, model.merger.norm
    bias = linear.bias
    assert bias is not None
    before = linear.weight.clone()
    with torch.no_grad():
        bias.fill_(1.0)
        norm.weight.fill_(3.0)
    model.reset_parameters()
    assert not torch.equal(linear.weight, before)
    assert torch.count_nonzero(bias) == 0
    assert torch.equal(norm.weight, torch.ones(8))


def test_a_layer_without_a_bias_is_drawn_without_one() -> None:
    model = _model()
    model.merger.linear_fc1.register_parameter("bias", None)
    model.reset_parameters()
    assert model.merger.linear_fc1.bias is None


def test_the_rotary_table_is_each_position_times_each_frequency() -> None:
    table = VisionRotaryEmbedding(4, theta=100.0)(3, device=torch.device("cpu"))
    expected = torch.tensor([[p * 1.0, p / 10.0] for p in range(3)])
    torch.testing.assert_close(table, expected, rtol=0, atol=0)


def test_rotate_turns_each_half_split_pair_by_its_angle() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    quarter = torch.tensor([[math.pi / 2, 0.0]])
    rotated = vision._rotate(x, cos=quarter.cos(), sin=quarter.sin())
    # Pair (1, 3) turns a quarter; pair (2, 4) not at all.
    torch.testing.assert_close(
        rotated.float(),
        torch.tensor([[-3.0, 2.0, 1.0, 4.0]]),
        rtol=0,
        atol=1e-6,
    )
    assert rotated.dtype == torch.bfloat16


def test_the_encoder_merges_each_square_of_patches_into_one_token() -> None:
    model = _model(depth=2)
    grid = torch.tensor([[1, 4, 6], [2, 2, 2]])
    with torch.no_grad():
        out = model(_pixels(grid), grid)
    # 24 patches and 2 frames of 4 patches, four to a token.
    assert out.shape == (8, 6)
    assert out.dtype == torch.float32
    with torch.no_grad():
        half = model.to(torch.bfloat16)(_pixels(grid), grid)
    assert half.dtype == torch.bfloat16


def test_each_image_attends_only_within_itself() -> None:
    model = _model()
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    pixels = _pixels(grid)
    changed = pixels.clone()
    changed[4:] = _pixels(grid, seed=1)[4:]
    with torch.no_grad():
        first, second = model(pixels, grid), model(changed, grid)
    assert torch.equal(first[0], second[0])
    assert not torch.equal(first[1], second[1])


def test_a_grid_the_merger_cannot_tile_is_refused() -> None:
    grid = torch.tensor([[1, 5, 4]])
    with pytest.raises(
        ValueError,
        match=r"spatial_merge_size=2; got \(t=1, h=5, w=4\)",
    ):
        _model()(_pixels(grid), grid)
    with pytest.raises(ValueError, match="w=3"):
        _model()(_pixels(torch.tensor([[1, 2, 3]])), torch.tensor([[1, 2, 3]]))


def test_positions_on_the_learned_grid_are_its_rows_in_merge_order() -> None:
    model = _model()
    positions = model._positions([[1, 4, 4]])
    order = [
        (block_row * 2 + row) * 4 + block_col * 2 + col
        for block_row in range(2)
        for block_col in range(2)
        for row in range(2)
        for col in range(2)
    ]
    torch.testing.assert_close(positions, model.pos_embed.weight[order], rtol=0, atol=0)
    # A grid finer than the learned one blends its neighbors.
    finer = model._positions([[1, 8, 8]])
    assert finer.shape == (64, 8)
    torch.testing.assert_close(finer[0], model.pos_embed.weight[0], rtol=0, atol=0)


def test_the_rotary_frequencies_follow_each_patchs_row_and_column() -> None:
    model = _model()
    rotary = model._rotary([[2, 2, 4]])
    table = model.rotary_pos_emb(4, device=torch.device("cpu"))
    coords = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
    expected = torch.stack([torch.cat([table[r], table[c]]) for r, c in coords])
    # Each frame repeats the first's positions.
    torch.testing.assert_close(rotary, torch.cat([expected, expected]), rtol=0, atol=0)


def test_a_checkpoints_vision_weights_load_strictly() -> None:
    model = _model(depth=2)
    hf = {
        "model.visual."
        + name.replace("patch_embed.", "patch_embed.proj."): torch.randn_like(value)
        for name, value in model.state_dict().items()
    }
    hf["model.language_model.embed_tokens.weight"] = torch.randn(3, 3)
    model.load_state_dict(remap_hf_vision_state_dict(hf), strict=True)
    for name, value in model.state_dict().items():
        key = "model.visual." + name.replace("patch_embed.", "patch_embed.proj.")
        assert torch.equal(value, hf[key]), name


def _tiny_reference() -> Qwen3VLVisionModel:
    """Return the reference encoder, as Hugging Face ships it, in the tiny shape."""
    config = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=8,
        intermediate_size=8,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=6,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3VLVisionModel(config).eval()


@pytest.mark.parametrize(
    "grid",
    [
        [[1, 4, 6], [2, 2, 2]],
        # Finer than the learned 4x4 grid: positions are interpolated.
        [[1, 6, 8]],
    ],
)
def test_the_encoder_computes_what_the_reference_does(grid: list[list[int]]) -> None:
    reference = _tiny_reference()
    ours = _model(depth=2).eval()
    state = {f"model.visual.{k}": v for k, v in reference.state_dict().items()}
    ours.load_state_dict(remap_hf_vision_state_dict(state), strict=True)
    grid_thw = torch.tensor(grid)
    pixels = _pixels(grid_thw, seed=1)

    with torch.no_grad():
        want = reference(pixels, grid_thw).pooler_output
        got = ours(pixels, grid_thw)

    assert want is not None
    torch.testing.assert_close(got, want, rtol=0, atol=1e-8)


def test_the_default_draws_and_rotary_base_are_the_references() -> None:
    model = VisionModel(
        depth=1,
        hidden_size=8,
        num_heads=2,
        intermediate_size=8,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=6,
        num_position_embeddings=16,
    )

    assert model.initializer_range == Qwen3VLVisionConfig().initializer_range
    assert model.rotary_pos_emb.theta == 10_000.0


@pytest.mark.parametrize(("hidden_size", "num_heads"), [(9, 2), (6, 2)])
def test_heads_of_no_even_width_are_refused(hidden_size: int, num_heads: int) -> None:
    message = (
        f"hidden_size={hidden_size} must split into num_heads={num_heads}"
        " heads of an even width."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        VisionModel(
            depth=1,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=8,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=6,
            num_position_embeddings=16,
        )


def test_every_tensor_is_made_on_the_models_device() -> None:
    """On ``meta``, any tensor made on the CPU instead fails to mix with it."""
    model = _model(depth=1).to("meta")
    grid = torch.tensor([[2, 4, 6]])

    out = model(_pixels(grid).to("meta"), grid)

    assert (out.device.type, tuple(out.shape)) == ("meta", (12, 6))


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_the_encoder_on_cuda_computes_what_it_does_on_the_cpu() -> None:
    """``meta`` takes a CPU index for a lookup; CUDA does not."""
    model = _model(depth=1).eval()
    grid = torch.tensor([[2, 4, 6]])
    pixels = _pixels(grid)
    with torch.no_grad():
        want = model(pixels, grid)
        got = model.to("cuda")(pixels.to("cuda"), grid)

    torch.testing.assert_close(got.cpu(), want, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
