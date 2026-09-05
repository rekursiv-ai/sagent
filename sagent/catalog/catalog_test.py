"""Invariants every vendor catalog must satisfy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import ModuleType
from typing import cast

import pytest

from sagent.catalog import (
    anthropic,
    dashscope,
    google,
    llamacpp,
    minimax,
    moonshot,
    openai,
)
from sagent.types.capability import ModelCapability, ModelSettings
from sagent.types.cost import PriceCatalogProduct, TokenCount


_VENDORS = (anthropic, dashscope, google, llamacpp, minimax, moonshot, openai)

_ROWS = [
    pytest.param(row, id=f"{module.__name__.rsplit('.', 1)[-1]}:{model_id}")
    for module in _VENDORS
    for model_id, row in module.models().items()
]


type _Case = tuple[Mapping[str, ModelCapability], ModelCapability]


def _transports() -> tuple[list[_Case], list[str]]:
    cases: list[_Case] = []
    ids: list[str] = []
    for module in _VENDORS:
        vendor = module.__name__.rsplit(".", 1)[-1]
        for name in ("api", "cli", "subscription", "chat"):
            if not hasattr(module, name):
                continue
            factory = cast(Callable[[], ModelCapability], getattr(module, name))
            cases.append((module.models(), factory()))
            ids.append(f"{vendor}:{name}")
    return cases, ids


_TRANSPORTS, _TRANSPORT_IDS = _transports()


@pytest.mark.parametrize("module", _VENDORS, ids=lambda m: m.__name__)
def test_every_vendor_serves_at_least_one_model(module: ModuleType) -> None:
    assert module.models()


@pytest.mark.parametrize("module", _VENDORS, ids=lambda m: m.__name__)
def test_models_is_a_function_not_a_table(module: ModuleType) -> None:
    assert isinstance(module.models(), Mapping)
    assert not hasattr(module, "MODELS")


@pytest.mark.parametrize("row", _ROWS)
def test_a_row_keys_itself_by_its_wire_id(row: ModelCapability) -> None:
    assert row.model_id


@pytest.mark.parametrize("row", _ROWS)
def test_a_row_offers_a_default_context(row: ModelCapability) -> None:
    assert "" in row.context


@pytest.mark.parametrize("row", _ROWS)
def test_every_context_declares_both_windows(row: ModelCapability) -> None:
    for tag, limits in row.context.items():
        assert limits.max_request_tokens > 0, tag
        assert limits.max_response_tokens > 0, tag


@pytest.mark.parametrize("row", _ROWS)
def test_every_row_is_priced(row: ModelCapability) -> None:
    assert len(row.prices) > 0
    product = PriceCatalogProduct(service_tier=ModelSettings().service_tier)
    assert row.prices[product] * TokenCount(request=1_000) is not None


@pytest.mark.parametrize("row", _ROWS)
def test_no_axis_is_empty(row: ModelCapability) -> None:
    assert row.thinking_effort
    assert row.thinking_budget
    assert row.thinking_output
    assert row.service_tier
    assert row.manage_context_server_side


@pytest.mark.parametrize("row", _ROWS)
def test_only_reasoning_only_models_reject_none(row: ModelCapability) -> None:
    """Keep vendor-enforced reasoning mandatory, and optional elsewhere."""
    reasoning_only = row.model_id.endswith("-thinking-2507") or row.model_id in {
        "gpt-6-astra",
        "gpt-5.5-pro",
        "gpt-5.4-pro",
        "o1",
        "o3-mini",
    }
    assert ("none" not in row.thinking_effort) == reasoning_only


@pytest.mark.parametrize("row", _ROWS)
def test_a_priced_tier_is_an_offered_tier(row: ModelCapability) -> None:
    for product in row.prices:
        assert product.service_tier in row.service_tier, product


@pytest.mark.parametrize(("models", "transport"), _TRANSPORTS, ids=_TRANSPORT_IDS)
def test_a_transport_only_removes(
    models: Mapping[str, ModelCapability], transport: ModelCapability
) -> None:
    for row in models.values():
        met = row & transport
        assert met.thinking_effort <= row.thinking_effort
        assert met.thinking_budget <= row.thinking_budget
        assert met.thinking_output <= row.thinking_output
        assert met.service_tier <= row.service_tier
        assert met.manage_context_server_side <= row.manage_context_server_side


@pytest.mark.parametrize(("models", "transport"), _TRANSPORTS, ids=_TRANSPORT_IDS)
def test_a_transport_preserves_windows_and_prices(
    models: Mapping[str, ModelCapability], transport: ModelCapability
) -> None:
    for row in models.values():
        met = row & transport
        assert met.context == row.context
        assert met.prices == row.prices
        assert met.model_id == row.model_id


@pytest.mark.parametrize(("models", "transport"), _TRANSPORTS, ids=_TRANSPORT_IDS)
def test_a_narrowed_row_leaves_every_axis_selectable(
    models: Mapping[str, ModelCapability], transport: ModelCapability
) -> None:
    for row in models.values():
        met = row & transport
        assert met.thinking_effort, row.model_id
        assert met.thinking_budget, row.model_id
        assert met.thinking_output, row.model_id
        assert met.service_tier, row.model_id
        assert met.manage_context_server_side, row.model_id


@pytest.mark.parametrize(("models", "transport"), _TRANSPORTS, ids=_TRANSPORT_IDS)
def test_a_transport_never_advertises_what_no_row_can_reach(
    models: Mapping[str, ModelCapability], transport: ModelCapability
) -> None:
    """A transport axis wider than EVERY row is a knob nobody can select.

    ``&`` takes the narrow side, and an omitted axis defaults narrow, so a
    row that simply forgets to declare one silently deletes the transport's
    offer. That is how ``flex``/``priority`` became unreachable on every
    OpenAI model and prompt caching switched itself off on every Anthropic
    one -- both green, because nothing compared the two sides.
    """
    for name in ("service_tier", "cache_ttl_sec"):
        offered = getattr(transport, name)
        best = max(
            (getattr(row & transport, name) for row in models.values()),
            key=lambda v: len(v) if isinstance(v, frozenset) else v,
        )
        assert best == offered, (
            f"{name}: transport offers {offered!r} but the widest row"
            f" reaches only {best!r}"
        )


# Base input/output USD per Mtok, transcribed from each vendor's pricing page
# on 2026-09-04. Only rows whose price this repo actually bills against are
# listed; the point is to catch a REPRICE, which no structural invariant sees.
# Sources:
#   https://developers.openai.com/api/docs/pricing
#   https://docs.anthropic.com/en/docs/about-claude/pricing
#   https://ai.google.dev/gemini-api/docs/pricing
_PUBLISHED_PRICES = {
    "gpt-6-astra": (10.0, 50.0),
    "gpt-5.6-sol": (4.0, 20.0),
    "gpt-5.6": (4.0, 20.0),
    "gpt-5.6-terra": (2.0, 12.0),
    "gpt-5.6-luna": (0.2, 1.2),
    "gpt-5.5": (5.0, 30.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-fable-5-1": (10.0, 50.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-2.5-pro": (1.25, 10.0),
}


@pytest.mark.parametrize(("model_id", "published"), _PUBLISHED_PRICES.items())
def test_a_row_bills_the_vendors_published_rate(
    model_id: str, published: tuple[float, float]
) -> None:
    """Catch a reprice, which every structural invariant here passes through.

    The GPT-5.6 family was repriced after its rows landed (Sol $5/$30 ->
    $4/$20, Luna $1/$6 -> $0.20/$1.20) and Sonnet 5's introductory $2/$10
    became permanent. Every other test in this file stayed green: they
    assert a row IS priced, never that the number is right.
    """
    rows = {mid: row for module in _VENDORS for mid, row in module.models().items()}
    price = rows[model_id].prices[PriceCatalogProduct()]
    assert (price.request, price.response) == published


def test_a_cache_rate_is_a_multiple_of_the_rate_it_rides() -> None:
    """Anthropic cache rates key off BASE INPUT, so they ride every modifier.

    Fast mode left them unmultiplied, billing a cached fast request at
    half its real rate. Fable 5.1 bills cache hits at 0.025x rather than
    the 0.1x every other model uses.
    """
    rows = anthropic.models()
    opus = rows["claude-opus-5"].prices
    standard = opus[PriceCatalogProduct()]
    fast = opus[PriceCatalogProduct(service_tier="priority")]
    assert (fast.request, fast.response) == (10.0, 50.0)
    assert fast.cache_write == fast.request * 1.25
    assert fast.cache_read == fast.request * 0.1
    assert fast.cache_write == standard.cache_write * 2.0
    fable = rows["claude-fable-5-1"].prices[PriceCatalogProduct()]
    assert fable.cache_read == 0.25


def test_no_catalog_declares_a_latency_tag() -> None:
    for module in _VENDORS:
        for model_id in module.models():
            assert "+fast" not in model_id


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
