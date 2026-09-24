"""Invariants every vendor catalog must satisfy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

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


if TYPE_CHECKING:
    from types import ModuleType


_VENDORS = (anthropic, dashscope, google, llamacpp, minimax, moonshot, openai)


def _models(module: ModuleType) -> Mapping[str, ModelCapability]:
    """Return a vendor's typed model catalog."""
    return cast(Callable[[], Mapping[str, ModelCapability]], module.models)()


_ROWS = [
    pytest.param(row, id=f"{module.__name__.rsplit('.', 1)[-1]}:{model_id}")
    for module in _VENDORS
    for model_id, row in _models(module).items()
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
            cases.append((_models(module), factory()))
            ids.append(f"{vendor}:{name}")
    return cases, ids


_TRANSPORTS, _TRANSPORT_IDS = _transports()


def _module_id(module: ModuleType) -> str:
    """Return a vendor module's qualified name for pytest."""
    return module.__name__


@pytest.mark.parametrize("module", _VENDORS, ids=_module_id)
def test_every_vendor_serves_at_least_one_model(module: ModuleType) -> None:
    assert _models(module)


@pytest.mark.parametrize("module", _VENDORS, ids=_module_id)
def test_models_is_a_function_not_a_table(module: ModuleType) -> None:
    assert isinstance(_models(module), Mapping)
    assert not hasattr(module, "MODELS")


@pytest.mark.parametrize("module", _VENDORS, ids=_module_id)
def test_every_catalog_names_its_default_and_utility_rows(module: ModuleType) -> None:
    models = _models(module)
    assert models["default"].model_id in models
    assert models["utility"].model_id in models


def test_openai_metadata_does_not_leak_from_new_models_to_legacy_rows() -> None:
    models = openai.models()
    assert models["astra-6"].approx_chars_per_token == 3.71
    assert models["gpt-4"].knowledge_cutoff is None


@pytest.mark.parametrize("row", _ROWS)
def test_a_row_has_a_catalog_id(row: ModelCapability) -> None:
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
def test_untagged_context_is_the_largest_window(row: ModelCapability) -> None:
    assert row.context[""].max_request_tokens == max(
        limits.max_request_tokens for limits in row.context.values()
    )


@pytest.mark.parametrize("row", _ROWS)
def test_every_row_is_priced(row: ModelCapability) -> None:
    assert len(row.prices) > 0
    product = PriceCatalogProduct(service_tier=ModelSettings().service_tier)
    assert row.prices[product] * TokenCount(request=1_000) is not None


@pytest.mark.parametrize("row", _ROWS)
def test_no_axis_is_empty(row: ModelCapability) -> None:
    assert row.thinking.effort
    assert row.thinking.budget
    assert row.thinking.output
    assert row.service_tier
    assert row.manage_context_server_side


@pytest.mark.parametrize("row", _ROWS)
def test_only_reasoning_only_models_reject_none(row: ModelCapability) -> None:
    """Keep vendor-enforced reasoning mandatory, and optional elsewhere."""
    reasoning_only = row.model_id.endswith("-thinking-2507") or row.model_id in {
        "astra-6",
        "gpt-5.5-pro",
        "gpt-5.4-pro",
        "o1",
        "o3-mini",
    }
    assert ("none" not in row.thinking.effort) == reasoning_only


@pytest.mark.parametrize("row", _ROWS)
def test_a_priced_tier_is_an_offered_tier(row: ModelCapability) -> None:
    for product in row.prices:
        assert product.service_tier in row.service_tier, product


@pytest.mark.parametrize(("models", "transport"), _TRANSPORTS, ids=_TRANSPORT_IDS)
def test_a_transport_only_removes(
    models: Mapping[str, ModelCapability],
    transport: ModelCapability,
) -> None:
    for row in models.values():
        met = row & transport
        assert met.thinking.effort <= row.thinking.effort
        assert met.thinking.budget <= row.thinking.budget
        assert met.thinking.output <= row.thinking.output
        assert met.service_tier <= row.service_tier
        assert met.manage_context_server_side <= row.manage_context_server_side


@pytest.mark.parametrize(("models", "transport"), _TRANSPORTS, ids=_TRANSPORT_IDS)
def test_a_transport_preserves_windows_and_prices(
    models: Mapping[str, ModelCapability],
    transport: ModelCapability,
) -> None:
    for row in models.values():
        met = row & transport
        assert met.context == row.context
        assert met.prices == row.prices
        assert met.model_id == row.model_id
        assert met.knowledge_cutoff == row.knowledge_cutoff
        assert met.approx_chars_per_token == row.approx_chars_per_token


@pytest.mark.parametrize(
    ("model_id", "cutoff"),
    [
        ("fable-5.1", "June 2026"),
        ("fable-5", "January 2026"),
        ("opus-5.5", "June 2026"),
        ("opus-5", "May 2026"),
        ("opus-4.8", "January 2026"),
        ("opus-4.7", "January 2026"),
        ("opus-4.6", "May 2025"),
        ("opus-4.5", "May 2025"),
        ("sonnet-5", "January 2026"),
        ("sonnet-4.6", "August 2025"),
        ("sonnet-4.5", "January 2025"),
        ("haiku-4.5", "February 2025"),
    ],
)
def test_every_anthropic_row_states_its_published_cutoff(
    model_id: str,
    cutoff: str,
) -> None:
    """Reliable knowledge cutoff, per each model's overview page (2026-09-24)."""
    assert anthropic.models()[model_id].knowledge_cutoff == cutoff


def test_a_tier_is_offered_exactly_when_it_is_priced() -> None:
    """``service_tier`` is derived from ``prices``, so the two cannot drift."""
    rows = anthropic.models()
    for row in rows.values():
        priced = {"auto", "default", *(p.service_tier for p in row.prices)}
        assert row.service_tier == priced, row.model_id
    fast = {m for m, row in rows.items() if "priority" in row.service_tier}
    assert fast == {"default", "opus-5.5", "opus-5", "opus-4.8"}


def test_pre_4_6_models_have_no_1m_window() -> None:
    """Sonnet 4.5 and Opus 4.5 are 200k / 64k models; a 1M row would 400."""
    for model_id in ("opus-4.5", "sonnet-4.5", "haiku-4.5"):
        row = anthropic.models()[model_id]
        assert set(row.context) == {""}, model_id
        assert row.context[""].max_request_tokens == 200_000, model_id
        assert row.context[""].max_response_tokens == 64_000, model_id


def test_published_knowledge_cutoffs_are_catalog_data() -> None:
    assert anthropic.models()["opus-5.5"].knowledge_cutoff == "June 2026"
    assert anthropic.models()["fable-5.1"].knowledge_cutoff == "June 2026"
    assert openai.models()["astra-6"].knowledge_cutoff == "April 30, 2026"
    assert openai.models()["sol-6"].knowledge_cutoff == "April 20, 2026"
    assert openai.models()["luna-6"].knowledge_cutoff == "May 18, 2026"


def test_anthropic_tokenizer_ratio_is_catalog_data() -> None:
    assert anthropic.models()["opus-5.5"].approx_chars_per_token == 2.38
    assert anthropic.models()["opus-4.6"].approx_chars_per_token == 3.12


@pytest.mark.parametrize(("models", "transport"), _TRANSPORTS, ids=_TRANSPORT_IDS)
def test_a_narrowed_row_leaves_every_axis_selectable(
    models: Mapping[str, ModelCapability],
    transport: ModelCapability,
) -> None:
    for row in models.values():
        met = row & transport
        assert met.thinking.effort, row.model_id
        assert met.thinking.budget, row.model_id
        assert met.thinking.output, row.model_id
        assert met.service_tier, row.model_id
        assert met.manage_context_server_side, row.model_id


@pytest.mark.parametrize(("models", "transport"), _TRANSPORTS, ids=_TRANSPORT_IDS)
def test_a_transport_never_advertises_what_no_row_can_reach(
    models: Mapping[str, ModelCapability],
    transport: ModelCapability,
) -> None:
    """A transport axis wider than EVERY row is a knob nobody can select.

    ``&`` takes the narrow side, and an omitted axis defaults narrow, so a
    row that simply forgets to declare one silently deletes the transport's
    offer. That is how ``flex``/``priority`` became unreachable on every
    OpenAI model and prompt caching switched itself off on every Anthropic
    one -- both green, because nothing compared the two sides.
    """
    for name in ("service_tier", "cache_ttl_sec"):
        offered = cast(frozenset[str] | bool | None, getattr(transport, name))
        best = max(
            (
                cast(frozenset[str] | bool | None, getattr(row & transport, name))
                for row in models.values()
            ),
            key=lambda v: len(v) if isinstance(v, frozenset) else int(bool(v)),
        )
        assert best == offered, (
            f"{name}: transport offers {offered!r} but the widest row"
            f" reaches only {best!r}"
        )


# Base input/output USD per Mtok, transcribed from each vendor's pricing page
# on 2026-09-22. Only rows whose price this repo actually bills against are
# listed; the point is to catch a REPRICE, which no structural invariant sees.
# Sources:
#   https://developers.openai.com/api/docs/pricing
#   https://docs.anthropic.com/en/docs/about-claude/pricing
#   https://ai.google.dev/gemini-api/docs/pricing
_PUBLISHED_PRICES = {
    "astra-6": (10.0, 50.0),
    "sol-6": (2.0, 10.0),
    "luna-6": (0.1, 0.5),
    "sol-5.6": (4.0, 20.0),
    "gpt-5.6": (4.0, 20.0),
    "terra-5.6": (2.0, 12.0),
    "luna-5.6": (0.2, 1.2),
    "gpt-5.5": (5.0, 30.0),
    "opus-5": (5.0, 25.0),
    "opus-5.5": (4.0, 20.0),
    "fable-5.1": (10.0, 50.0),
    "sonnet-5": (2.0, 10.0),
    "haiku-4.5": (1.0, 5.0),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-2.5-pro": (1.25, 10.0),
}


@pytest.mark.parametrize(("model_id", "published"), _PUBLISHED_PRICES.items())
def test_a_row_bills_the_vendors_published_rate(
    model_id: str,
    published: tuple[float, float],
) -> None:
    """Catch a reprice, which every structural invariant here passes through.

    The GPT-5.6 family was repriced after its rows landed (Sol $5/$30 ->
    $4/$20, Luna $1/$6 -> $0.20/$1.20) and Sonnet 5's introductory $2/$10
    became permanent. Every other test in this file stayed green: they
    assert a row IS priced, never that the number is right.
    """
    rows = {mid: row for module in _VENDORS for mid, row in _models(module).items()}
    price = rows[model_id].prices[PriceCatalogProduct()]
    assert (price.request, price.response) == published


def test_a_cache_rate_is_a_multiple_of_the_rate_it_rides() -> None:
    """Anthropic cache rates key off BASE INPUT, so they ride every modifier.

    Fast mode left them unmultiplied, billing a cached fast request at
    half its real rate. Fable 5.1 bills cache hits at 0.025x rather than
    the 0.1x every other model uses.
    """
    rows = anthropic.models()
    opus = rows["opus-5"].prices
    standard = opus[PriceCatalogProduct()]
    fast = opus[PriceCatalogProduct(service_tier="priority")]
    assert (fast.request, fast.response) == (10.0, 50.0)
    assert fast.cache_write == fast.request * 1.25
    assert fast.cache_read == fast.request * 0.1
    assert fast.cache_write == standard.cache_write * 2.0
    fable = rows["fable-5.1"].prices[PriceCatalogProduct()]
    assert fable.cache_read == 0.25


@pytest.mark.parametrize(
    ("model_id", "cache_read"),
    [
        ("fable-5.1", 0.25),
        ("fable-5", 1.0),
        ("opus-5.5", 0.2),
        ("opus-5", 0.5),
        ("sonnet-5", 0.2),
        ("sonnet-4.6", 0.3),
        ("haiku-4.5", 0.1),
    ],
)
def test_anthropic_cache_hits_bill_the_published_rate(
    model_id: str,
    cache_read: float,
) -> None:
    """Fable 5.1 is 0.025x, Opus 5.5 0.05x, every other model 0.1x input."""
    price = anthropic.models()[model_id].prices[PriceCatalogProduct()]
    assert price.cache_read == pytest.approx(cache_read)


def test_anthropic_context_betas_are_catalog_data() -> None:
    rows = anthropic.models()
    assert rows["opus-4.8"].context[""].request_betas == frozenset(
        {"context-1m-2025-08-07"},
    )
    assert not rows["opus-4.8"].context["+200k"].request_betas
    assert not rows["opus-5.5"].context[""].request_betas


@pytest.mark.parametrize(
    ("catalog_id", "wire_id"),
    [
        ("astra-6", "gpt-6-astra"),
        ("sol-6", "gpt-6-sol"),
        ("luna-6", "gpt-6-luna"),
        ("terra-5.6", "gpt-5.6-terra"),
        ("fable-5.1", "claude-fable-5-1"),
        ("opus-5.5", "claude-opus-5-5"),
        ("sonnet-5", "claude-sonnet-5"),
        ("haiku-4.5", "claude-haiku-4-5"),
        ("opus-4.8", "claude-opus-4-8"),
        ("opus-4.7", "claude-opus-4-7"),
        ("opus-4.6", "claude-opus-4-6"),
        ("opus-4.5", "claude-opus-4-5"),
        ("sonnet-4.6", "claude-sonnet-4-6"),
        ("sonnet-4.5", "claude-sonnet-4-5"),
    ],
)
def test_simple_catalog_ids_are_distinct_from_vendor_wire_ids(
    catalog_id: str,
    wire_id: str,
) -> None:
    rows = {**openai.models(), **anthropic.models()}
    row = rows[catalog_id]
    assert row.model_id == catalog_id
    assert row.wire_model_id == wire_id
    assert wire_id not in rows


def test_no_catalog_declares_a_latency_tag() -> None:
    for module in _VENDORS:
        for model_id in _models(module):
            assert "+fast" not in model_id


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
