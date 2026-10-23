"""OpenAI transport, subscription projection, and wire behavior."""

from collections.abc import Iterator
from dataclasses import replace
from datetime import date

import pytest

from sagent.catalog import loader, openai
from sagent.catalog.table import ModelTable
from sagent.types.capability import (
    ModelCapability,
    ModelLimits,
    ThinkingCapability,
    ThinkingEffort,
)
from sagent.types.cost import PriceCatalog, PriceKey, ServiceTier, TokenPrice


_TRANSPORT_THINKING = ThinkingCapability(
    effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
    budget=frozenset({"none", "auto", "fixed"}),
    output=frozenset({"none", "text", "redacted"}),
)


@pytest.fixture(autouse=True)
def synthetic_catalog(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModelTable]:
    """Resolve behavioral model keys without reading provider assets."""
    table = ModelTable(
        rows=tuple(
            ModelCapability(model_id=name, wire_model_id=f"wire-{name}")
            for name in (
                "astra-6.0",
                "sol-6.1",
                "sol-5.6",
                "luna-5.6",
                "gpt-5.5",
                "gpt-5.1",
                "gpt-4.0",
                "gpt-omni-4.0",
                "o-mini-3.0",
                "gpt-pro-5.5",
                "gpt-turbo-4.0",
            )
        ),
    )

    def lookup(provider: str) -> ModelTable:
        assert provider == "openai"
        return table

    monkeypatch.setattr(openai, "models", lookup)
    openai.subscription_models.cache_clear()
    yield table
    openai.subscription_models.cache_clear()


@pytest.mark.parametrize(
    ("reported", "tier"),
    [
        (None, "auto"),
        ("auto", "auto"),
        ("default", "auto"),
        ("flex", "flex"),
        ("priority", "priority"),
        ("fast", "priority"),
        ("ultrafast", "ultrafast"),
        ("scale", "auto"),
    ],
)
def test_openai_bills_the_tier_it_reports_serving(
    reported: str | None,
    tier: ServiceTier,
) -> None:
    assert openai.served_tier(reported) == tier


def test_an_unknown_reported_tier_is_unknown_not_an_error() -> None:
    assert openai.served_tier("hyperfast") is None


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_reasoning_effort_sends_the_selected_level_verbatim(
    effort: ThinkingEffort,
) -> None:
    assert openai.reasoning_effort(effort) == effort


@pytest.mark.parametrize("effort", ["none", "min"])
def test_reasoning_effort_has_no_spelling_for_sending_no_knob(
    effort: ThinkingEffort,
) -> None:
    with pytest.raises(ValueError, match=rf"^effort {effort!r} is not sent"):
        openai.reasoning_effort(effort)


def test_subscription_offers_only_auto_and_priority_over_account_auth() -> None:
    assert openai.subscription() == ModelCapability(
        thinking=_TRANSPORT_THINKING,
        service_tier=frozenset({"auto", "priority"}),
        account_auth=True,
    )


def test_compatible_offers_only_the_standard_tier_without_server_context() -> None:
    assert openai.compatible() == ModelCapability(
        thinking=_TRANSPORT_THINKING,
        service_tier=frozenset({"auto"}),
        manage_context_server_side=frozenset({False}),
    )


def test_api_offers_every_tier_over_key_auth() -> None:
    assert openai.api() == ModelCapability(
        thinking=_TRANSPORT_THINKING,
        service_tier=frozenset({"auto", "default", "flex", "priority", "ultrafast"}),
    )


@pytest.mark.parametrize(
    ("prompt_tokens", "response", "expected_request", "expected_response"),
    [
        (10_000, 1_000, 10_000, 1_000),
        (272_000, 32_000, 272_000, 32_000),
        (900_000, 100_000, 272_000, 32_000),
        (10_000, 100_000, 10_000, 32_000),
        (900_000, 1_000, 272_000, 1_000),
    ],
)
def test_subscription_clamps_windows_and_preserves_other_facts(
    monkeypatch: pytest.MonkeyPatch,
    prompt_tokens: int,
    response: int,
    expected_request: int,
    expected_response: int,
) -> None:
    limits = ModelLimits(
        max_request_tokens=prompt_tokens,
        max_response_tokens=response,
        max_request_bytes=123_456,
        max_image_edge_px=789,
        max_image_bytes=12_345,
    )
    source = ModelCapability(
        model_id="example-1.0",
        wire_model_id="example-wire",
        knowledge_cutoff="fixture cutoff",
        approx_chars_per_token=2.5,
        effort_as_level=True,
        thinking=_TRANSPORT_THINKING,
        context={"": limits, "+272k": replace(limits, max_request_tokens=42)},
        prices=PriceCatalog(
            {
                PriceKey("auto"): TokenPrice(
                    request=1.0,
                    response=2.0,
                    cache_write=3.0,
                    cache_write_1h=4.0,
                    cache_read=5.0,
                ),
            },
        ),
    )
    other = replace(source, model_id="other-1.0", wire_model_id="other-wire")
    table = ModelTable(
        rows=(source, other),
        roles={"default": "example", "utility": "other"},
    )

    def lookup(provider: str) -> ModelTable:
        assert provider == "openai"
        return table

    monkeypatch.setattr(openai, "models", lookup)

    projected = openai.subscription_models()

    assert projected.rows == tuple(
        replace(
            row,
            context={
                "": replace(
                    limits,
                    max_request_tokens=expected_request,
                    max_response_tokens=expected_response,
                ),
            },
        )
        for row in table.rows
    )
    assert projected.roles == table.roles
    assert projected["default"].prices is source.prices
    assert table["default"] is source
    assert source.context[""].max_request_tokens == prompt_tokens
    assert "+272k" in source.context
    assert openai.subscription_models() is projected


@pytest.mark.parametrize(
    ("model_id", "width", "height", "tokens"),
    [
        ("astra-6.0", 1024, 1024, 1229),
        ("wire-astra-6.0", 4096, 512, 2458),
        ("sol-5.6", 33, 65, 8),
        ("gpt-5.5", 8000, 8000, 12_000),
        ("gpt-omni-4.0", 33, 65, 255),
        ("gpt-5.1", 1024, 1024, 70 + 140 * 4),
        ("o-mini-3.0", 33, 65, 0),
        ("gpt-4.0", 33, 65, 0),
        ("gpt-pro-5.5", 33, 65, 85 + 170),
        ("gpt-turbo-4.0", 33, 65, 85 + 170),
        ("unlisted-vendor", 33, 65, 85 + 170),
    ],
)
def test_image_tokens_follow_the_published_formula(
    model_id: str,
    width: int,
    height: int,
    tokens: int,
) -> None:
    assert openai.image_tokens(model_id, width, height) == tokens


@pytest.mark.parametrize(
    ("model_id", "encoding"),
    [
        ("astra-6.0", "o200k_base"),
        ("wire-gpt-omni-4.0", "o200k_base"),
        ("gpt-4.0", "cl100k_base"),
        ("gpt-turbo-4.0", "cl100k_base"),
        ("unknown", None),
    ],
)
def test_tokenizer_selects_encoding_from_the_resolved_row(
    model_id: str,
    encoding: str | None,
) -> None:
    assert openai.tokenizer(model_id) == encoding


@pytest.mark.parametrize(
    ("model_id", "all_turns"),
    [
        ("wire-sol-6.1", True),
        ("luna-5.6", True),
        ("gpt-5.5", False),
        ("unknown", False),
    ],
)
def test_reasoning_context_uses_the_resolved_behavior_key(
    model_id: str,
    all_turns: bool,
) -> None:
    assert openai.keeps_reasoning_across_turns(model_id) is all_turns


# ---- published facts in ``assets/openai.yaml`` ------------------------------
# ``loader.models`` is the real catalog; the autouse fixture patches only the
# name ``openai.models`` the behavior functions above read.

_TODAY = date(2026, 10, 22)


def test_every_per_model_table_names_a_catalog_row() -> None:
    """A renamed row would silently fall back to the default image/turn rules."""
    names = set(loader.models("openai"))
    assert openai._VISION.keys() <= names
    assert names >= openai._TEXT_ONLY
    assert names >= openai._ALL_TURNS
    assert names >= openai._CL100K


# (input, cached input, cache write, output) from the standard / flex / fast
# tabs of the pricing page, 2026-09-24.
@pytest.mark.parametrize(
    ("model_id", "tier", "published"),
    [
        ("astra-6.0", "auto", (10.0, 1.0, 12.5, 50.0)),
        ("astra-6.0", "flex", (5.0, 0.5, 6.25, 25.0)),
        ("astra-6.0", "priority", (20.0, 2.0, 25.0, 100.0)),
        ("astra-6.0", "ultrafast", (60.0, 6.0, 75.0, 300.0)),
        ("sol-6.1", "auto", (2.0, 0.1, 2.5, 10.0)),
        ("sol-6.1", "flex", (1.0, 0.05, 1.25, 5.0)),
        ("luna-6.0", "flex", (0.05, 0.005, 0.0625, 0.25)),
        ("luna-6.0", "priority", (0.2, 0.02, 0.25, 1.0)),
        ("sol-5.6", "auto", (4.0, 0.4, 5.0, 20.0)),
        ("luna-5.6", "auto", (0.2, 0.02, 0.25, 1.2)),
        ("terra-5.6", "flex", (1.0, 0.1, 1.25, 6.0)),
        ("gpt-5.5", "auto", (5.0, 0.5, 0.0, 30.0)),
        ("gpt-5.4", "flex", (1.25, 0.13, 0.0, 7.5)),
        ("gpt-omni-4.0", "priority", (4.25, 2.125, 0.0, 17.0)),
        ("gpt-5.2", "priority", (3.5, 0.35, 0.0, 28.0)),
    ],
)
def test_tiers_bill_the_published_rate(
    model_id: str,
    tier: ServiceTier,
    published: tuple[float, float, float, float],
) -> None:
    price = loader.models("openai")[model_id].prices[PriceKey(tier)]
    assert (price.request, price.cache_read, price.cache_write, price.response) == (
        published
    )


@pytest.mark.parametrize(
    ("model_id", "tier", "published"),
    [
        ("luna-6.0", "auto", (0.2, 0.02, 0.25, 0.75)),
        ("luna-6.0", "priority", (0.4, 0.04, 0.5, 1.5)),
        ("luna-5.6", "auto", (0.4, 0.04, 0.5, 1.8)),
        ("astra-6.0", "ultrafast", (120.0, 12.0, 150.0, 450.0)),
        # Published, not 2x the rounded $0.13.
        ("gpt-5.4", "flex", (2.5, 0.25, 0.0, 11.25)),
    ],
)
def test_prompts_over_272k_bill_the_published_long_context_card(
    model_id: str,
    tier: ServiceTier,
    published: tuple[float, float, float, float],
) -> None:
    prices = loader.models("openai")[model_id].prices
    short = prices.rate(service_tier=tier, prompt_tokens=272_000, at=_TODAY)
    long = prices.rate(service_tier=tier, prompt_tokens=272_001, at=_TODAY)
    assert short is prices[PriceKey(tier)]
    assert (long.request, long.cache_read, long.cache_write, long.response) == (
        published
    )


@pytest.mark.parametrize(
    ("model_id", "tier"),
    [("gpt-5.5", "priority"), ("gpt-5.4", "priority"), ("gpt-pro-5.5", "flex")],
)
def test_a_tier_the_table_leaves_unbanded_has_no_long_card(
    model_id: str,
    tier: ServiceTier,
) -> None:
    """The pricing table prints "-" for these long columns."""
    assert PriceKey(tier, 272_000) not in loader.models("openai")[model_id].prices


@pytest.mark.parametrize(
    ("model_id", "offered"),
    [
        ("astra-6.0", {"auto", "default", "flex", "priority", "ultrafast"}),
        ("sol-6.1", {"auto", "default", "flex", "priority"}),
        ("gpt-5.5", {"auto", "default", "flex", "priority"}),
        ("gpt-pro-5.5", {"auto", "default", "flex"}),
        ("gpt-4.1", {"auto", "default", "priority"}),
        ("o-1.0", {"auto", "default"}),
    ],
)
def test_only_the_published_tiers_are_offered(
    model_id: str,
    offered: set[ServiceTier],
) -> None:
    """A model absent from a tier's pricing tab cannot be billed at it."""
    assert loader.models("openai")[model_id].service_tier == offered


@pytest.mark.parametrize(
    ("model_id", "default_tokens"),
    [
        ("astra-6.0", 922_000),
        ("sol-6.1", 922_000),
        ("sol-5.6", 922_000),
        ("luna-5.6", 922_000),
        ("gpt-5.5", 1_050_000),
        ("gpt-pro-5.5", 1_050_000),
        ("gpt-5.4", 1_050_000),
    ],
)
def test_long_context_rows_default_to_the_full_window(
    model_id: str,
    default_tokens: int,
) -> None:
    context = loader.models("openai")[model_id].context
    assert context.keys() == {"", "+272k"}
    assert context[""].max_request_tokens == default_tokens
    assert context["+272k"].max_request_tokens == 272_000
    assert context[""].max_response_tokens == 128_000
    assert context[""].max_request_bytes == 512 * 1024 * 1024


@pytest.mark.parametrize(
    ("model_id", "window", "edge_px"),
    [
        ("astra-6.0", (922_000, 128_000), 65_535),
        ("gpt-5.2", (400_000, 128_000), 2048),
        ("gpt-mini-5.4", (272_000, 128_000), 2048),
        ("gpt-4.1", (1_047_576, 32_768), 2048),
        ("gpt-omni-4.0", (128_000, 16_384), 2048),
        ("gpt-4.0", (8_192, 8_192), 0),
    ],
)
def test_rows_carry_the_published_limits(
    model_id: str,
    window: tuple[int, int],
    edge_px: int,
) -> None:
    limits = loader.models("openai")[model_id].context[""]
    assert (limits.max_request_tokens, limits.max_response_tokens) == window
    assert limits.max_image_edge_px == edge_px


@pytest.mark.parametrize(
    ("model_id", "efforts"),
    [
        # Astra and 6.1 Sol 400 on ``none``.
        ("astra-6.0", {"low", "medium", "high", "xhigh", "max"}),
        ("sol-6.1", {"low", "medium", "high", "xhigh", "max"}),
        ("sol-6.0", {"none", "low", "medium", "high", "xhigh", "max"}),
        ("gpt-5.5", {"none", "low", "medium", "high", "xhigh"}),
        ("gpt-pro-5.5", {"medium", "high", "xhigh"}),
        ("gpt-5.1", {"none", "low", "medium", "high"}),
        ("o-mini-3.0", {"low", "medium", "high"}),
        ("gpt-omni-4.0", {"none"}),
        ("gpt-4.1", {"none"}),
    ],
)
def test_rows_offer_the_published_reasoning_efforts(
    model_id: str,
    efforts: set[ThinkingEffort],
) -> None:
    assert loader.models("openai")[model_id].thinking.effort == efforts


def test_no_row_offers_min() -> None:
    """No model page lists ``minimal``, so no row may let a caller select it."""
    assert all(
        "min" not in row.thinking.effort for row in loader.models("openai").values()
    )


@pytest.mark.parametrize(
    ("model_id", "wire_id", "cutoff"),
    [
        ("astra-6.0", "gpt-6-astra", "April 30, 2026"),
        ("sol-6.1", "gpt-6.1-sol", "April 30, 2026"),
        ("sol-6.0", "gpt-6-sol", "April 20, 2026"),
        ("luna-6.0", "gpt-6-luna", "May 18, 2026"),
        ("terra-5.6", "gpt-5.6-terra", "February 16, 2026"),
        ("gpt-pro-5.5", "gpt-5.5-pro", "December 1, 2025"),
        ("o-mini-3.0", "o3-mini", "October 1, 2023"),
        ("gpt-omni-4.0", "gpt-4o", "October 1, 2023"),
        ("gpt-4.0", "gpt-4", "December 1, 2023"),
    ],
)
def test_rows_name_their_wire_id_and_cutoff(
    model_id: str,
    wire_id: str,
    cutoff: str,
) -> None:
    table = loader.models("openai")
    assert table[model_id].wire_model_id == wire_id
    assert table[model_id].knowledge_cutoff == cutoff
    assert table.exact(wire_id) is table[model_id]


def test_roles_name_the_selected_models() -> None:
    table = loader.models("openai")
    assert {role: table[role].model_id for role in table.roles} == {
        "best": "astra-6.0",
        "default": "sol-6.1",
        "everyday": "sol-6.1",
        "utility": "luna-6.0",
    }
    assert table["astra-6.0"].approx_chars_per_token == 3.71


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
