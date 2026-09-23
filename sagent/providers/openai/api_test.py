"""Tests for ``providers.openai``: API-key dispatch + effort-model gating."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import asyncio
import os

from PIL import Image

import httpx2
import pytest
import tiktoken

from sagent.providers.openai.api import OpenAI
from sagent.types.cost import PriceCatalogProduct

import sagent.catalog.openai


if TYPE_CHECKING:
    from sagent.types.capability import (
        ThinkingEffort,
    )


@pytest.mark.network_openai
@pytest.mark.asyncio
async def test_every_catalog_row_is_a_model_the_vendor_serves() -> None:
    """Every catalog row must complete a successful Responses request."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")

    # Bounded concurrency: firing one connection per row made the slowest
    # response the whole test's fate, and `ReadTimeout` under that fan-out was
    # the observed failure rather than any catalog defect.
    model_ids = tuple(
        row.wire_model_id or model_id
        for model_id, row in sagent.catalog.openai.models().items()
    )
    gate = asyncio.Semaphore(8)

    async def _probe(client: httpx2.AsyncClient, model_id: str) -> bool:
        async with gate:
            return await _is_served(client, model_id=model_id, key=key)

    async with httpx2.AsyncClient(timeout=60.0) as client:
        verdicts = await asyncio.gather(
            *(_probe(client, model_id) for model_id in model_ids),
        )
    assert all(verdicts), "some catalog probes never completed successfully"


async def _is_served(client: httpx2.AsyncClient, *, model_id: str, key: str) -> bool:
    for attempt in range(3):
        try:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers={"authorization": f"Bearer {key}"},
                json={
                    "model": model_id,
                    "input": [{"role": "user", "content": "."}],
                    "max_output_tokens": 16,
                },
            )
        except httpx2.TransportError:
            await asyncio.sleep(2**attempt)
            continue
        if response.status_code == 200:
            return True
        if response.status_code >= 500 or response.status_code == 429:
            await asyncio.sleep(2**attempt)
            continue
        response.raise_for_status()
    return False


def test_openai_from_key_constructs() -> None:
    p = OpenAI.from_key("sk-test")
    assert p.api_key == "sk-test"
    assert p.base_url == "https://api.openai.com/v1"


def test_openai_from_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key not configured"):
        OpenAI.from_env()


def test_openai_from_env_reads_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    p = OpenAI.from_env()
    assert p.api_key == "sk-env"


def test_openai_default_model_known() -> None:
    p = OpenAI.from_key("k")
    m = p.model()
    assert m.tagged_model_id == p.catalog.resolve("default")[0].model_id


def test_openai_default_models_are_astra_and_luna() -> None:
    """Astra for reasoning, Luna for utility (50x/42x cheaper per token)."""
    assert OpenAI.catalog.resolve("default")[0].model_id == "astra-6"
    assert OpenAI.catalog.resolve("utility")[0].model_id == "luna-6"


def test_normalized_openai_id_uses_vendor_id_on_the_wire() -> None:
    model = OpenAI.from_key("k").model("astra-6")
    assert model._wire_model_id == "gpt-6-astra"


def test_openai_known_model_returns_backend() -> None:
    p = OpenAI.from_key("k")
    m = p.model("gpt-4o")
    assert m.capability.model_id == "gpt-4o"
    assert m.limits.max_request_tokens == 128_000


def test_openai_unknown_model_raises() -> None:
    p = OpenAI.from_key("k")
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("not-a-real-model")


@pytest.mark.parametrize(
    ("base_id", "full_tokens"),
    [
        ("sol-5.6", 1_050_000),
        ("gpt-5.6", 1_050_000),
        ("terra-5.6", 1_050_000),
        ("luna-5.6", 1_050_000),
        ("gpt-5.5", 1_000_000),
        ("gpt-5.5-pro", 1_050_000),
        ("gpt-5.4", 1_050_000),
        ("gpt-5.4-pro", 1_050_000),
    ],
)
def test_openai_two_tier_defaults_to_full_window(
    base_id: str,
    full_tokens: int,
) -> None:
    p = OpenAI.from_key("k")
    full = p.model(base_id)
    small = p.model(f"{base_id}+272k")
    assert full.limits.max_request_tokens == full_tokens
    assert small.limits.max_request_tokens == 272_000
    assert small.tagged_model_id.endswith("+272k")
    assert (
        full.capability.prices[PriceCatalogProduct()]
        == small.capability.prices[PriceCatalogProduct()]
    )
    assert full.limits.max_request_bytes == small.limits.max_request_bytes


def test_openai_gpt_6_astra_profile() -> None:
    """Measured against the live API 2026-09-04.

    ``reasoning.effort`` 400s on ``none``/``minimal`` and takes low..max;
    ``service_tier`` takes auto/default/flex/priority (``batch`` 400s);
    the untagged window is 1.05M and ``+272k`` selects the smaller cap.
    """
    m = OpenAI.from_key("k").model("astra-6")
    assert m.limits.max_request_tokens == 1_050_000
    assert m.limits.max_response_tokens == 128_000
    assert m.capability.prices[PriceCatalogProduct()].request == 10.0
    assert m.capability.prices[PriceCatalogProduct()].response == 50.0
    assert m.capability.prices[PriceCatalogProduct()].cache_write == 12.5
    assert m.capability.prices[PriceCatalogProduct()].cache_read == 1.0
    assert m.capability.service_tier == frozenset(
        {"auto", "default", "flex", "priority"},
    )


def test_openai_gpt_6_astra_cannot_be_asked_not_to_think() -> None:
    """Astra withholds ``none``.

    The API rejects both ``none`` and ``minimal`` outright, so offering
    them would let a caller select a value the wire refuses.
    """
    m = OpenAI.from_key("k").model("astra-6")
    assert m.capability.thinking_effort == frozenset(
        {"low", "medium", "high", "xhigh", "max"},
    )
    with pytest.raises(ValueError, match="thinking_effort"):
        m.settings.thinking_effort = "none"


def test_openai_gpt_6_keeps_the_native_effort_ladder() -> None:
    """GPT-6 spells ``xhigh``/``max`` literally, as 5.6 does.

    Falling back to the pre-5.6 ladder would silently downgrade ``max``
    to ``high`` and lose the top rung the model actually serves.
    """
    assert (
        sagent.catalog.openai.reasoning_effort(
            "max",
            model_id="gpt-6-astra",
        )
        == "max"
    )
    assert (
        sagent.catalog.openai.reasoning_effort(
            "xhigh",
            model_id="gpt-6-astra",
        )
        == "xhigh"
    )


@pytest.mark.parametrize("effort", ["none", "min"])
def test_openai_gpt_6_effort_floor_never_emits_a_rejected_value(
    effort: ThinkingEffort,
) -> None:
    """Astra's floor is ``low``: the API 400s on ``none`` and ``minimal``.

    The capability withholds both levels, but this function is public and
    reachable without that check, so returning ``none`` here would leave
    the guarantee resting on call order rather than on the mapping.
    """
    assert (
        sagent.catalog.openai.reasoning_effort(
            effort,
            model_id="gpt-6-astra",
        )
        == "low"
    )


def test_openai_default_model_opts_into_full_window() -> None:
    # The untagged default is the full window.
    p = OpenAI.from_key("k")
    m = p.model()
    assert m.tagged_model_id == "astra-6"
    assert m.limits.max_request_tokens == 1_050_000


@pytest.mark.parametrize(
    ("model_id", "request_price", "response_price", "cache_write_price"),
    [
        ("sol-5.6", 4.0, 20.0, 5.0),
        ("gpt-5.6", 4.0, 20.0, 5.0),
        ("gpt-5.6-terra", 2.0, 12.0, 2.5),
        ("luna-5.6", 0.2, 1.2, 0.25),
    ],
)
def test_openai_gpt_56_profiles(
    model_id: str,
    request_price: float,
    response_price: float,
    cache_write_price: float,
) -> None:
    m = OpenAI.from_key("k").model(model_id)
    assert m.limits.max_request_tokens == 1_050_000
    assert m.limits.max_response_tokens == 128_000
    assert m.capability.prices[PriceCatalogProduct()].request == request_price
    assert m.capability.prices[PriceCatalogProduct()].response == response_price
    assert m.capability.prices[PriceCatalogProduct()].cache_write == cache_write_price
    assert m.capability.prices[PriceCatalogProduct()].cache_read == request_price / 10
    # The >272K surcharge is a second catalog row, not a multiplier field:
    # 2x on every input pool, 1.5x on the response.
    tier = m.capability.prices[PriceCatalogProduct(min_request_tokens=272_000)]
    assert tier.request == request_price * 2.0
    assert tier.cache_write == cache_write_price * 2.0
    assert tier.cache_read == request_price / 10 * 2.0
    assert tier.response == response_price * 1.5
    assert m.limits.max_image_edge_px == 0
    assert m.limits.max_image_bytes == 0
    assert m.limits.max_request_bytes == 512 * 1024 * 1024


def _png(width: int, height: int) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (width, height)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize(
    ("width", "height", "tokens"),
    [(1, 1, 1), (1024, 1024, 1024), (33, 65, 6)],
)
def test_openai_gpt_56_image_tokens_use_32px_patches(
    width: int,
    height: int,
    tokens: int,
) -> None:
    model = OpenAI.from_key("k").model("sol-5.6")
    assert model.approx_image_tokens(_png(width, height)) == tokens


@pytest.mark.parametrize(
    ("width", "height", "tokens"),
    [(1, 1, 2), (96, 96, 11), (512, 512, 308), (100, 200, 34), (2048, 1024, 2458)],
)
def test_openai_gpt_6_image_tokens_scale_the_patch_grid(
    width: int,
    height: int,
    tokens: int,
) -> None:
    """Astra bills ``floor(1.2 * patches) + 1``, not 5.6's bare patch count.

    Every expectation is a server-reported ``usage.input_tokens`` delta
    measured on 2026-09-04, with the text-only envelope differenced out.
    Reusing the 5.6 formula under-counts each of these by ~20%.
    """
    model = OpenAI.from_key("k").model("gpt-6-astra")
    assert model.approx_image_tokens(_png(width, height)) == tokens


@pytest.mark.anyio
@pytest.mark.compute_large_fixture
async def test_openai_gpt_56_uses_o200k_tokenizer() -> None:
    model = OpenAI.from_key("k").model("sol-5.6")
    text = "GPT-5.6 token counting: 東京 and function_call(arg=42)"
    expected = len(tiktoken.get_encoding("o200k_base").encode(text))
    assert await model.actual_text_tokens(text) == expected


@pytest.mark.anyio
@pytest.mark.compute_large_fixture
async def test_openai_gpt_6_uses_o200k_tokenizer() -> None:
    """Tiktoken's registry predates ``gpt-6``, so the fallback must catch it.

    Without the generation mapping the model silently degrades to the
    chars/4 heuristic. Server counts tracked ``o200k_base`` within 1.2%
    over 2.9k chars of mixed code, prose, and CJK (measured 2026-09-04).
    """
    model = OpenAI.from_key("k").model("gpt-6-astra")
    text = "GPT-6 token counting: 東京 and function_call(arg=42)"
    expected = len(tiktoken.get_encoding("o200k_base").encode(text))
    assert await model.actual_text_tokens(text) == expected


@pytest.mark.parametrize("base_id", ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"])
def test_openai_no_cliff_model_defaults_to_full_window(base_id: str) -> None:
    p = OpenAI.from_key("k")
    model = p.model(base_id)
    assert model.limits.max_request_tokens == 1_047_576
    assert set(model.capability.context) == {""}


@pytest.mark.parametrize(
    "model_id",
    ["gpt-5.4-mini+1m", "gpt-5.4-nano+1m", "gpt-5.3-codex+1m", "gpt-5.2+1m"],
)
def test_openai_400k_model_has_no_plus1m(model_id: str) -> None:
    # 400K-window models have no long-context mode; ``+1m`` must not resolve.
    p = OpenAI.from_key("k")
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model(model_id)


def test_openai_utility_model_default() -> None:
    p = OpenAI.from_key("k")
    m = p.model("utility")
    assert m.capability.model_id == "luna-6"


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("o3-mini", True),
        ("gpt-5.5", True),
        ("gpt-5.4-mini", True),
        ("gpt-4o", False),
        ("gpt-4.1", False),
    ],
)
def test_openai_effort_gating(model_id: str, expected: bool) -> None:
    """Effort support is catalog data, not an id-prefix guess.

    The prefix predicate it replaced answered ``True`` for any unknown
    ``o4-*`` id; the catalog answers only for models it actually declares
    a reasoning-effort table for.
    """
    m = OpenAI.from_key("k").model(model_id)
    assert (m.capability.thinking_effort != frozenset({"none"})) is expected


def test_openai_pricing_attached_to_model() -> None:
    p = OpenAI.from_key("k")
    m = p.model("gpt-5.5")
    price = m.capability.prices[PriceCatalogProduct()]
    assert price.request > 0
    assert price.response > 0


def test_openai_valid_service_tiers() -> None:
    p = OpenAI.from_key("k")
    m = p.model("gpt-5.5")
    assert m.capability.service_tier == frozenset(
        {"auto", "default", "flex", "priority"},
    )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
