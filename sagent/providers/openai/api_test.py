"""OpenAI API-key dispatch and catalog-to-backend wiring."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO

import asyncio
import os

from PIL import Image

import httpx2
import pytest
import tiktoken

from sagent.catalog import openai
from sagent.catalog.loader import models
from sagent.catalog.openai import api
from sagent.catalog.table import ModelCatalog, ModelTable
from sagent.providers.openai.api import OpenAI
from sagent.tools import tool
from sagent.types.capability import (
    ModelCapability,
    ModelLimits,
    ThinkingCapability,
)
from sagent.types.cost import PriceCatalog, PriceKey, TokenPrice
from sagent.types.model import ModelRequest
from sagent.types.runtime import UserMessage


@pytest.fixture(autouse=True)
def synthetic_catalog(monkeypatch: pytest.MonkeyPatch) -> ModelTable:
    """Keep dispatch tests independent of live catalog facts."""
    row = ModelCapability(
        model_id="example-1.0",
        wire_model_id="wire-example",
        context={
            "": ModelLimits(max_request_tokens=600_000, max_response_tokens=50_000),
            "+272k": ModelLimits(
                max_request_tokens=272_000,
                max_response_tokens=20_000,
            ),
        },
        thinking=ThinkingCapability(
            effort=frozenset({"low", "high"}),
            budget=frozenset({"none", "auto"}),
            output=frozenset({"none", "text"}),
        ),
        service_tier=frozenset({"auto", "priority"}),
        prices=PriceCatalog(
            {
                PriceKey("auto"): TokenPrice(
                    request=3.0,
                    response=7.0,
                    cache_write=2.0,
                    cache_write_1h=4.0,
                    cache_read=0.5,
                ),
            },
        ),
    )
    utility = replace(
        row,
        model_id="utility-example-1.0",
        wire_model_id="wire-utility",
        thinking=ThinkingCapability(),
    )
    table = ModelTable(
        rows=(
            row,
            utility,
            *(
                replace(row, model_id=name, wire_model_id=wire)
                for name, wire in (
                    ("sol-5.6", "gpt-5.6-sol"),
                    ("astra-6.0", "gpt-6-astra"),
                )
            ),
        ),
        roles={"default": "example", "utility": "utility-example"},
    )

    def lookup(provider: str) -> ModelTable:
        assert provider == "openai"
        return table

    monkeypatch.setattr(openai, "models", lookup)
    monkeypatch.setattr(OpenAI, "catalog", ModelCatalog(models=table, transport=api()))
    return table


@pytest.mark.network_openai
@pytest.mark.asyncio
async def test_every_catalog_row_is_a_model_the_vendor_serves() -> None:
    """Every catalog row must complete a successful Responses request."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    model_ids = tuple(
        row.wire_model_id or name for name, row in models("openai").items()
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


def test_openai_from_key_constructs() -> None:
    provider = OpenAI.from_key("sk-test")
    assert provider.api_key == "sk-test"
    assert provider.base_url == "https://api.openai.com/v1"


@pytest.mark.asyncio
async def test_openai_sdk_connect_timeout_outlasts_a_network_stall() -> None:
    provider = OpenAI.from_key("k")
    sdk = await provider.get_sdk()
    timeout = sdk.timeout
    assert isinstance(timeout, httpx2.Timeout)
    assert (timeout.connect, timeout.read) == (30.0, 600.0)
    await provider.close_sdk()
    assert sdk.is_closed()


def test_openai_from_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key not configured"):
        OpenAI.from_env()


def test_openai_from_env_reads_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert OpenAI.from_env().api_key == "sk-env"


@pytest.mark.parametrize("role", ["default", "utility"])
def test_roles_select_catalog_rows(role: str, synthetic_catalog: ModelTable) -> None:
    model = OpenAI.from_key("k").model(role)
    assert model.capability == synthetic_catalog[role] & api()
    assert model.limits == synthetic_catalog[role].context[""]


def test_omitted_model_selects_the_default(synthetic_catalog: ModelTable) -> None:
    model = OpenAI.from_key("k").model()
    assert model.capability.model_id == synthetic_catalog["default"].model_id


@pytest.mark.parametrize("name", ["example-1.0", "wire-example", "example"])
def test_known_names_dispatch_to_the_same_backend(name: str) -> None:
    model = OpenAI.from_key("k").model(name)
    assert model.capability.model_id == "example-1.0"
    assert model._wire_model_id == "wire-example"


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        OpenAI.from_key("k").model("not-a-real-model")


def test_context_selection_preserves_prices(synthetic_catalog: ModelTable) -> None:
    provider = OpenAI.from_key("k")
    full = provider.model()
    small = provider.model("example+272k")
    assert full.limits == synthetic_catalog["example"].context[""]
    assert small.limits == synthetic_catalog["example"].context["+272k"]
    assert small.tagged_model_id == "example-1.0+272k"
    assert small.capability.prices is synthetic_catalog["example"].prices
    assert full.capability.prices is small.capability.prices


def test_unoffered_context_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"has no \+1m context"):
        OpenAI.from_key("k").model("example+1m")


def test_unoffered_effort_is_rejected() -> None:
    model = OpenAI.from_key("k").model()
    assert model.settings.thinking_effort == "low"
    with pytest.raises(ValueError, match="thinking_effort"):
        model.settings.thinking_effort = "none"


def test_a_cache_key_reaches_the_api_as_prompt_cache_key() -> None:
    model = OpenAI.from_key("k").model()
    request = ModelRequest(messages=[UserMessage(text="hi")], cache_key="run-1")

    assert model._build_kwargs(request).get("prompt_cache_key") == "run-1"
    assert model._headers(request) == {}
    assert "prompt_cache_key" not in model._build_kwargs(
        ModelRequest(messages=[UserMessage(text="hi")]),
    )


def _look(topic: str) -> str:
    """Look a topic up."""
    return topic


def _peek(topic: str) -> str:
    """Peek at a topic."""
    return topic


def _offering(*offers: tuple[str, ...] | None) -> ModelRequest:
    """Return a request of Look and Peek, one user turn per tool offer."""
    return ModelRequest(
        messages=[UserMessage(text="hi", offered_tools=offer) for offer in offers],
        tools=[tool(_look, name="Look"), tool(_peek, name="Peek")],
    )


def test_a_tool_gate_reaches_the_api_as_allowed_tools_with_every_schema() -> None:
    """Every schema ships so the prefix caches; the gate says which may be called."""
    model = OpenAI.from_key("k").model()
    peek_only = {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "function", "name": "Peek"}],
    }
    gated = model._build_kwargs(_offering(("Peek",)))
    assert [t.get("name") for t in gated.get("tools", [])] == ["Look", "Peek"]
    assert gated.get("tool_choice") == peek_only
    assert model._build_kwargs(_offering(())).get("tool_choice") == "none"
    assert "tool_choice" not in model._build_kwargs(_offering(None))
    # The latest offer is the gate in force; one naming every tool lifts it.
    lifted = model._build_kwargs(_offering(("Peek",), ("Look", "Peek")))
    assert "tool_choice" not in lifted
    assert (
        model._build_kwargs(_offering(("Peek",), None)).get("tool_choice") == peek_only
    )
    assert model._build_kwargs(_offering(("Nope",))).get("tool_choice") == "none"


def test_catalog_controls_effort_and_tier_selection() -> None:
    provider = OpenAI.from_key("k")
    reasoning = provider.model("example")
    utility = provider.model("utility")
    assert reasoning.capability.thinking.effort == {"low", "high"}
    assert utility.capability.thinking.effort == {"none"}
    reasoning.settings.service_tier = "priority"
    assert reasoning._effective_service_tier() == "priority"
    with pytest.raises(ValueError, match="service_tier"):
        reasoning.settings.service_tier = "flex"


@pytest.mark.parametrize(
    ("model_id", "width", "height", "tokens"),
    [
        ("sol-5.6", 1, 1, 2),
        ("sol-5.6", 1024, 1024, 1229),
        ("sol-5.6", 33, 65, 8),
        ("gpt-6-astra", 96, 96, 11),
        ("gpt-6-astra", 512, 512, 308),
        ("gpt-6-astra", 100, 200, 34),
        ("gpt-6-astra", 2048, 1024, 2458),
    ],
)
def test_image_token_estimation_reaches_the_model_formula(
    model_id: str,
    width: int,
    height: int,
    tokens: int,
) -> None:
    buffer = BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    model = OpenAI.from_key("k").model(model_id)
    assert model.approx_image_tokens(buffer.getvalue()) == tokens


@pytest.mark.anyio
@pytest.mark.compute_large_fixture
@pytest.mark.parametrize("name", ["sol-5.6", "gpt-6-astra"])
async def test_text_counting_reaches_the_model_tokenizer(name: str) -> None:
    model = OpenAI.from_key("k").model(name)
    text = "Token counting: 東京 and function_call(arg=42)"
    assert await model.actual_text_tokens(text) == len(
        tiktoken.get_encoding("o200k_base").encode(text),
    )


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
            await asyncio.sleep(1 << attempt)
            continue
        if response.status_code == 200:
            return True
        if response.status_code >= 500 or response.status_code == 429:
            await asyncio.sleep(1 << attempt)
            continue
        response.raise_for_status()
    return False


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
