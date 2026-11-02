"""Tests for estimating a request's tokens locally."""

from __future__ import annotations

from types import SimpleNamespace

import io
import math

from PIL import Image
from treekle import json_dumps, mutable

import pytest

from sagent.model.capability import ModelCapability, ModelLimits
from sagent.model.types import Request
from sagent.providers.openai import token_count
from sagent.session.types import (
    AssistantMessage,
    Attachment,
    Thinking,
    TokenUsage,
    ToolCall,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.read import Read


def _png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(out, format="PNG")
    return out.getvalue()


def _model(model_id: str, *, max_edge_px: int = 0) -> ModelCapability:
    return ModelCapability(
        model_id=model_id,
        approx_chars_per_token=2.0,
        limits=ModelLimits(max_image_edge_px=max_edge_px),
    )


@pytest.fixture
def tokenizers(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Count one token per character, and record which encodings were asked for."""
    asked: list[str] = []

    def get_encoding(name: str) -> SimpleNamespace:
        asked.append(name)
        return SimpleNamespace(encode_ordinary=list)

    monkeypatch.setattr(
        token_count,
        "tiktoken",
        SimpleNamespace(get_encoding=get_encoding),
    )
    return asked


def test_estimate_reads_every_text_the_request_sends(tokenizers: list[str]) -> None:
    read = Read()
    request = Request(
        system=("sys",),
        tools=(read,),
        records=(
            UserMessage(content="user"),
            AssistantMessage(content="said"),
            AssistantMessage(),
            Thinking(summary="summary", content="raw"),
            Thinking(content="raw"),
            ToolCall(call_id="c", name="Read", arguments={"file_path": "a"}),
            UncategorizedToolResult(call_id="c", content="result"),
            # Bookkeeping says nothing to the model.
            TokenUsage(),
        ),
    )
    texts = [
        "sys",
        "Read",
        read.description,
        json_dumps(mutable(read.schema)),
        "user",
        "said",
        "",
        "summary",
        "raw",
        "Read",
        '{"file_path": "a"}',
        "result",
    ]
    estimated = token_count.estimate(
        request,
        capability=_model("gpt-5.6"),
        encoding="o200k_base",
    )
    assert estimated == len("\n".join(texts))
    assert tokenizers == ["o200k_base"]


def test_estimate_without_a_tokenizer_divides_by_characters_per_token(
    tokenizers: list[str],
) -> None:
    request = Request(records=(UserMessage(content="x" * 21),))
    assert token_count.estimate(request, capability=_model("kimi-3.0")) == 10
    assert tokenizers == []


def test_estimate_adds_each_image_shrunk_to_the_models_edge() -> None:
    capability = _model("gpt-4.1", max_edge_px=512)
    image = Attachment(mime_descriptor="image/png", data=_png(1024, 1024))
    pdf = Attachment(mime_descriptor="application/pdf", data=b"%PDF")
    request = Request(
        records=(
            UserMessage(attachments=(image, pdf)),
            # The model's own images are not sent back.
            AssistantMessage(attachments=(image,)),
            UncategorizedToolResult(call_id="c", attachments=(image,)),
        ),
    )
    # Shrunk to 512 px, each is one 512 px tile: 85 + 170.
    text = int(len("\n\n") / 2.0)
    assert token_count.estimate(request, capability=capability) == text + 2 * 255
    one = Request(records=(UserMessage(attachments=(image,)),))
    # At 1024 px, the 768 px short side is two tiles a side: 85 + 4 * 170.
    assert token_count.estimate(one, capability=_model("gpt-4.1")) == 765


@pytest.mark.parametrize(
    ("model_id", "width", "height", "tokens"),
    [
        # Patches: 64 x 64 is 4 patches, at 1.2 each.
        ("astra-6.0", 64, 64, 5),
        ("gpt-mini-4.1", 64, 64, 7),
        # Over its 2,500-patch budget, an image shrinks to 50 x 50 patches.
        ("gpt-5.4", 4096, 4096, 3000),
        ("gpt-5.5", 4096, 4096, 12_000),
        # Tiles: a base, and one rate per 512 px tile after the shrinks.
        ("gpt-5.1", 512, 512, 210),
        ("gpt-omni-mini-4.0", 512, 512, 8500),
        ("gpt-4.1", 4096, 1024, 85 + 170 * 4),
        # A short side over 768 px shrinks to it: 2000 x 1000 is 1536 x 768.
        ("gpt-4.1", 2000, 1000, 85 + 170 * 6),
        # Another vendor's model takes GPT-4o's estimate.
        ("kimi-3.0", 100, 100, 255),
    ],
)
def test_an_image_costs_what_the_models_published_formula_says(
    model_id: str,
    width: int,
    height: int,
    tokens: int,
) -> None:
    assert token_count.image_tokens(model_id, width, height) == tokens


@pytest.mark.parametrize(
    ("model_id", "tokens"),
    [
        # Four 32 px patches: 1.2 each, or 1.62 for GPT-4.1 mini.
        *(
            (model_id, 5)
            for model_id in (
                "astra-6.0",
                "sol-6.1",
                "sol-6.0",
                "luna-6.0",
                "sol-5.6",
                "gpt-5.6",
                "terra-5.6",
                "luna-5.6",
                "gpt-5.5",
                "gpt-5.4",
                "gpt-mini-5.4",
                "gpt-nano-5.4",
                "gpt-5.2",
            )
        ),
        ("gpt-mini-4.1", 7),
        # One 512 px tile.
        ("gpt-5.1", 70 + 140),
        ("gpt-omni-mini-4.0", 2833 + 5667),
        ("gpt-4.1", 85 + 170),
        ("gpt-omni-4.0", 85 + 170),
    ],
)
def test_each_model_bills_a_small_image_by_its_own_rule(
    model_id: str,
    tokens: int,
) -> None:
    assert token_count.image_tokens(model_id, 64, 64) == tokens


@pytest.mark.parametrize(
    ("model_id", "budget"),
    [
        ("gpt-5.5", 10_000),
        ("gpt-5.4", 2_500),
        ("gpt-mini-5.4", 2_500),
        ("gpt-nano-5.4", 2_500),
        ("gpt-5.2", 6_144),
        ("gpt-mini-4.1", 6_144),
    ],
)
def test_an_image_one_patch_over_its_budget_is_shrunk_into_it(
    model_id: str,
    budget: int,
) -> None:
    rate = 1.62 if model_id == "gpt-mini-4.1" else 1.2
    # 41 x 61, 73 x 137, and 5 x 1229 patches are each one over a budget.
    over = {2_500: (41, 61), 10_000: (73, 137), 6_144: (5, 1229)}[budget]
    width, height = (32 * n for n in over)

    assert token_count.image_tokens(model_id, width, height) < math.ceil(
        (budget + 1) * rate,
    )


def test_gpt_6_shrinks_no_image_into_a_budget() -> None:
    assert token_count.image_tokens("astra-6.0", 4096, 4096) == math.ceil(
        128 * 128 * 1.2,
    )


@pytest.mark.parametrize(
    ("width", "height", "budget", "fitted"),
    [
        # The guide's worked example: 1800 x 2400 into 1,536 patches.
        (1800, 2400, 1536, (1056, 1408)),
        (2400, 1800, 1536, (1408, 1056)),
        # Exactly the budget stays as it is; no budget shrinks nothing.
        (1600, 1600, 2_500, (1600, 1600)),
        (1599, 1599, 2_500, (1599, 1599)),
        (64, 64, 0, (64, 64)),
        (64, 64, 1, (32, 32)),
        # A side shrunk below one pixel keeps one.
        (1, 100_000, 1, (1, 1)),
        (100_000, 1, 1, (1, 1)),
    ],
)
def test_an_image_is_fit_to_its_patch_budget(
    width: int,
    height: int,
    budget: int,
    fitted: tuple[int, int],
) -> None:
    assert token_count._fit_patches(width, height, budget=budget) == fitted


@pytest.mark.parametrize(
    ("width", "height", "fitted"),
    [
        # The 768 px short-side shrink, of either side.
        (1537, 769, (1535, 768)),
        (1000, 2000, (768, 1536)),
        (768, 768, (768, 768)),
        # A side shrunk below one pixel keeps one.
        (1, 100_000, (1, 2048)),
        (100_000, 1, (2048, 1)),
    ],
)
def test_an_image_is_fit_to_its_tiles(
    width: int,
    height: int,
    fitted: tuple[int, int],
) -> None:
    assert token_count._fit_tiles(width, height) == fitted


@pytest.mark.parametrize(("width", "height"), [(513, 512), (512, 513)])
def test_a_pixel_past_a_tile_is_another_tile(width: int, height: int) -> None:
    assert token_count.image_tokens("gpt-5.1", width, height) == 70 + 140 * 2


def test_an_image_attached_is_billed_by_its_models_rule() -> None:
    image = Attachment(mime_descriptor="image/png", data=_png(64, 64))
    unreadable = Attachment(mime_descriptor="image/png", data=b"not a png")

    assert token_count.attachment_tokens(image, capability=_model("astra-6.0")) == 5
    assert token_count.attachment_tokens(unreadable, capability=_model("gpt-5.1")) == 0


def test_thinking_with_nothing_said_costs_nothing() -> None:
    request = Request(records=(UserMessage(content="x" * 4), Thinking()))

    assert token_count.estimate(request, capability=_model("kimi-3.0")) == 2


def test_a_tall_thin_image_keeps_at_least_one_pixel_a_side() -> None:
    assert token_count.image_tokens("gpt-4.1", 1, 100_000) == 85 + 170 * 4


def test_an_image_within_the_edge_is_billed_at_its_own_size() -> None:
    capability = _model("gpt-4.1", max_edge_px=4096)
    image = Attachment(mime_descriptor="image/png", data=_png(3000, 100))
    request = Request(records=(UserMessage(attachments=(image,)),))
    # 3000 x 100 fits 2048 px as 2048 x 68: four tiles by one.
    assert token_count.estimate(request, capability=capability) == 85 + 170 * 4


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
