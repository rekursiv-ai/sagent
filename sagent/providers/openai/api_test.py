"""Tests for ``providers.openai``: API-key dispatch + effort-model gating."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import cast

import asyncio
import os

from PIL import Image

import httpx2
import pytest
import tiktoken

from sagent.catalog import openai as openai_catalog
from sagent.lib.custom_json import DictCodec, StrCodec
from sagent.providers.openai.api import OpenAI
from sagent.types.capability import (
    ModelSettings,
    ServiceTier,
    ThinkingEffort,
)
from sagent.types.cost import PriceCatalogProduct
from sagent.types.model import ModelRequest
from sagent.types.runtime import UserMessage
from sagent.types.tools import Tool


@pytest.mark.network_openai
@pytest.mark.asyncio
async def test_every_catalog_row_is_a_model_the_vendor_serves() -> None:
    """A catalog row must name a model the API will actually accept.

    Retirement is only visible from a real call. ``GET /v1/models`` lists
    what the KEY is entitled to see, which is a different set: ``gpt-5.6``
    is absent from that listing yet serves ``POST /v1/responses``
    normally, and trusting the listing removed a live model from this
    catalog. Only ``model_not_found`` from a real request proves a row is
    dead, so that is what this asserts.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")

    # Only `model_not_found` is evidence. A timeout or 5xx says the vendor was
    # slow, which is not a claim about the catalog -- returning `True` there
    # reported a live model as retired, and returning `False` would hide a real
    # one. Unresolved is its own verdict, counted but never failing.
    async def _is_dead(client: httpx2.AsyncClient, model_id: str) -> bool | None:
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
                return False
            if response.status_code >= 500 or response.status_code == 429:
                await asyncio.sleep(2**attempt)
                continue
            error = DictCodec.coerce(DictCodec.coerce(response.json()).get("error"))
            return StrCodec.coerce(error.get("code")) == "model_not_found"
        return None

    # Bounded concurrency: firing one connection per row made the slowest
    # response the whole test's fate, and `ReadTimeout` under that fan-out was
    # the observed failure rather than any catalog defect.
    model_ids = tuple(openai_catalog.models())
    gate = asyncio.Semaphore(8)

    async def _probe(client: httpx2.AsyncClient, model_id: str) -> bool | None:
        async with gate:
            return await _is_dead(client, model_id)

    async with httpx2.AsyncClient(timeout=60.0) as client:
        verdicts = await asyncio.gather(
            *(_probe(client, model_id) for model_id in model_ids)
        )
    dead = [
        model_id
        for model_id, is_dead in zip(model_ids, verdicts, strict=True)
        if is_dead is True
    ]
    assert not dead, f"catalog names models the API does not serve: {dead}"
    # Positive control: every row unresolved means the probe measured nothing,
    # which must not read as a pass.
    assert any(verdict is not None for verdict in verdicts), (
        "every catalog probe failed to reach the API; nothing was verified"
    )


def test_openai_from_key_constructs() -> None:
    p = OpenAI.from_key("sk-test")
    assert isinstance(p, OpenAI)
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
    m = p.model()  # picks DEFAULT_MODEL.
    assert m.tagged_model_id == OpenAI.DEFAULT_MODEL


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
        ("gpt-5.6-sol", 1_050_000),
        ("gpt-5.6", 1_050_000),
        ("gpt-5.6-terra", 1_050_000),
        ("gpt-5.6-luna", 1_050_000),
        ("gpt-5.5", 1_000_000),
        ("gpt-5.5-pro", 1_050_000),
        ("gpt-5.4", 1_050_000),
        ("gpt-5.4-pro", 1_050_000),
    ],
)
def test_openai_two_tier_default_caps_at_272k(base_id: str, full_tokens: int) -> None:
    p = OpenAI.from_key("k")
    base = p.model(base_id)
    full = p.model(f"{base_id}+1m")
    assert base.limits.max_request_tokens == 272_000
    assert full.limits.max_request_tokens == full_tokens
    assert full.tagged_model_id == f"{base_id}+1m"
    # ``+1m`` only widens the window; pricing and other limits track the base.
    assert (
        full.capability.prices[PriceCatalogProduct()]
        == base.capability.prices[PriceCatalogProduct()]
    )
    assert full.limits.max_request_bytes == base.limits.max_request_bytes


def test_openai_gpt_6_astra_profile() -> None:
    """Measured against the live API 2026-09-04.

    ``reasoning.effort`` 400s on ``none``/``minimal`` and takes low..max;
    ``service_tier`` takes auto/default/flex/priority (``batch`` 400s);
    the window is 272K untagged with a 1.05M ``+1m`` variant.
    """
    m = OpenAI.from_key("k").model("gpt-6-astra")
    assert m.limits.max_request_tokens == 272_000
    assert m.limits.max_response_tokens == 128_000
    assert m.capability.prices[PriceCatalogProduct()].request == 10.0
    assert m.capability.prices[PriceCatalogProduct()].response == 50.0
    assert m.capability.prices[PriceCatalogProduct()].cache_write == 12.5
    assert m.capability.prices[PriceCatalogProduct()].cache_read == 1.0
    assert m.capability.service_tier == frozenset(
        {"auto", "default", "flex", "priority"}
    )
    assert OpenAI.from_key("k").model("gpt-6-astra+1m").limits.max_request_tokens == (
        1_050_000
    )


def test_openai_gpt_6_astra_cannot_be_asked_not_to_think() -> None:
    """Astra is the one OpenAI row that withholds ``none``.

    The API rejects both ``none`` and ``minimal`` outright, so offering
    them would let a caller select a value the wire refuses.
    """
    m = OpenAI.from_key("k").model("gpt-6-astra")
    assert m.capability.thinking_effort == frozenset(
        {"low", "medium", "high", "xhigh", "max"}
    )
    with pytest.raises(ValueError, match="thinking_effort"):
        m.settings.thinking_effort = "none"


def test_openai_gpt_6_keeps_the_native_effort_ladder() -> None:
    """GPT-6 spells ``xhigh``/``max`` literally, as 5.6 does.

    Falling back to the pre-5.6 ladder would silently downgrade ``max``
    to ``high`` and lose the top rung the model actually serves.
    """
    assert openai_catalog.reasoning_effort("max", model_id="gpt-6-astra") == "max"
    assert openai_catalog.reasoning_effort("xhigh", model_id="gpt-6-astra") == "xhigh"
    # Chat Completions tops out at ``xhigh``; ``max`` is Responses-only.
    assert (
        openai_catalog.reasoning_effort("max", model_id="gpt-6-astra", chat=True)
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
    assert openai_catalog.reasoning_effort(effort, model_id="gpt-6-astra") == "low"


def test_openai_default_model_opts_into_full_window() -> None:
    # API-key default is the ``+1m`` variant: full window out of the box.
    p = OpenAI.from_key("k")
    m = p.model()
    assert m.tagged_model_id == "gpt-5.6-sol+1m"
    assert m.limits.max_request_tokens == 1_050_000


@pytest.mark.parametrize(
    ("model_id", "request_price", "response_price", "cache_write_price"),
    [
        ("gpt-5.6-sol", 4.0, 20.0, 5.0),
        ("gpt-5.6", 4.0, 20.0, 5.0),
        ("gpt-5.6-terra", 2.0, 12.0, 2.5),
        ("gpt-5.6-luna", 0.2, 1.2, 0.25),
    ],
)
def test_openai_gpt_56_profiles(
    model_id: str,
    request_price: float,
    response_price: float,
    cache_write_price: float,
) -> None:
    m = OpenAI.from_key("k").model(model_id)
    assert m.limits.max_request_tokens == 272_000
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
    model = OpenAI.from_key("k").model("gpt-5.6-sol")
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
    model = OpenAI.from_key("k").model("gpt-5.6-sol")
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
def test_openai_no_cliff_model_plus1m_is_alias(base_id: str) -> None:
    # gpt-4.1 has a single flat price (no 272K tier), so ``+1m`` is an alias:
    # both ids resolve to the same full window.
    p = OpenAI.from_key("k")
    assert (
        p.model(f"{base_id}+1m").limits.max_request_tokens
        == p.model(base_id).limits.max_request_tokens
        == 1_047_576
    )


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
    m = p.utility_model()
    assert m.capability.model_id == OpenAI.DEFAULT_UTILITY_MODEL


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
        {"auto", "default", "flex", "priority"}
    )


@pytest.mark.parametrize("tier", ["default", "flex", "priority"])
def test_openai_build_body_emits_service_tier(tier: ServiceTier) -> None:
    m = OpenAI.from_key("k").model("gpt-5.5")
    m._settings = ModelSettings(capability=m.capability, service_tier=tier)
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")]),
        stream=False,
    )
    assert body["service_tier"] == tier


def test_openai_build_body_omits_the_default_tier() -> None:
    """``auto`` is "let the vendor pick"; sending it pins nothing."""
    m = OpenAI.from_key("k").model("gpt-5.5")
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")]),
        stream=False,
    )
    assert "service_tier" not in body


def test_openai_build_body_omits_effort_when_none() -> None:
    """``none`` means "send no knob", not "send the wire's off value"."""
    m = OpenAI.from_key("k").model("gpt-5.5")
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")]),
        stream=False,
    )
    assert "reasoning_effort" not in body


@pytest.mark.parametrize(
    ("effort", "wire_effort"),
    [
        ("min", "minimal"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "high"),
        ("max", "high"),
    ],
)
def test_openai_pre_56_maps_effort_to_wire_vocabulary(
    effort: ThinkingEffort,
    wire_effort: str,
) -> None:
    """The pre-5.6 wire takes only ``minimal``/``low``/``medium``/``high``."""
    m = OpenAI.from_key("k").model("gpt-5.5")
    m._settings = ModelSettings(capability=m.capability, thinking_effort=effort)
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")]),
        stream=False,
    )
    assert body["reasoning_effort"] == wire_effort


@pytest.mark.parametrize(
    ("effort", "wire_effort"),
    [
        ("min", "none"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        ("max", "xhigh"),
    ],
)
def test_openai_gpt_56_maps_effort_for_chat_completions(
    effort: ThinkingEffort,
    wire_effort: str,
) -> None:
    m = OpenAI.from_key("k").model("gpt-5.6-sol")
    m._settings = ModelSettings(capability=m.capability, thinking_effort=effort)
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")]),
        stream=False,
    )
    assert body["reasoning_effort"] == wire_effort


@pytest.mark.parametrize("effort", ["none", "low", "max"])
def test_openai_gpt_56_chat_tools_force_none_effort(effort: ThinkingEffort) -> None:
    tool = cast(
        Tool,
        SimpleNamespace(
            name="List",
            description="List files",
            directive_schema={"type": "object", "properties": {}},
        ),
    )
    model = OpenAI.from_key("k").model("gpt-5.6-sol")
    model._settings = ModelSettings(capability=model.capability, thinking_effort=effort)
    body = model._build_body(
        ModelRequest(messages=[UserMessage(text="x")], tools=[tool]),
        stream=False,
    )
    assert body["reasoning_effort"] == "none"


@pytest.mark.parametrize(("effort", "wire_effort"), [("low", "low"), ("max", "xhigh")])
def test_openai_gpt_6_chat_tools_reach_the_wire(
    effort: ThinkingEffort, wire_effort: str
) -> None:
    """Let the provider decide whether Astra accepts Chat Completions tools."""
    tool = cast(
        Tool,
        SimpleNamespace(
            name="List",
            description="List files",
            directive_schema={"type": "object", "properties": {}},
        ),
    )
    model = OpenAI.from_key("k").model("gpt-6-astra")
    model._settings = ModelSettings(capability=model.capability, thinking_effort=effort)
    body = model._build_body(
        ModelRequest(messages=[UserMessage(text="x")], tools=[tool]),
        stream=False,
    )
    assert body["reasoning_effort"] == wire_effort
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "List",
                "description": "List files",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


@pytest.mark.asyncio
async def test_openai_gpt_6_chat_tools_attempt_surfaces_provider_error() -> None:
    """Send Astra tools and preserve the provider's rejection text."""
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(400, text="tools temporarily unsupported")

    tool = cast(
        Tool,
        SimpleNamespace(
            name="List",
            description="List files",
            directive_schema={"type": "object", "properties": {}},
        ),
    )
    model = OpenAI.from_key("k").model("gpt-6-astra")
    model._client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    try:
        with pytest.raises(
            httpx2.HTTPStatusError, match="tools temporarily unsupported"
        ):
            await model.stream(
                ModelRequest(messages=[UserMessage(text="x")], tools=[tool])
            )
    finally:
        await model.close()
    assert len(requests) == 1


def test_openai_gpt_6_without_tools_still_builds_a_body() -> None:
    """Astra builds a normal Chat Completions body without tools."""
    model = OpenAI.from_key("k").model("gpt-6-astra")
    body = model._build_body(
        ModelRequest(messages=[UserMessage(text="x")]),
        stream=False,
    )
    assert body["model"] == "gpt-6-astra"
    assert body["reasoning_effort"] == "low"


def test_openai_reasoning_model_uses_max_completion_tokens() -> None:
    # gpt-5 / o-series reject ``max_tokens`` (400 unsupported_parameter);
    # they require ``max_completion_tokens``.
    p = OpenAI.from_key("k")
    m = p.model("gpt-5.5")
    body = m._build_body(
        ModelRequest(messages=[UserMessage(text="x")], max_response_tokens=42),
        stream=False,
    )
    assert body["max_completion_tokens"] == 42
    assert "max_tokens" not in body


def test_catalog_efforts_are_all_buildable() -> None:
    """Every effort the catalog offers must reach the wire.

    The catalog stores the wire value as data; a parallel mapper keyed off a
    different vocabulary made ``min`` pass validation and then raise at
    body-build time.
    """
    m = OpenAI.from_key("k").model("gpt-5.6-sol")
    for effort in m.capability.thinking_effort - {"none"}:
        m._settings = ModelSettings(capability=m.capability, thinking_effort=effort)
        body = m._build_body(
            ModelRequest(messages=[UserMessage(text="x")]),
            stream=False,
        )
        assert body["reasoning_effort"] == openai_catalog.reasoning_effort(
            effort, model_id="gpt-5.6-sol", chat=True
        )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
