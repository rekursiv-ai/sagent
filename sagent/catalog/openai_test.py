"""OpenAI catalog rows, transports, and wire mappings."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from sagent.catalog import openai
from sagent.types.capability import (
    ModelCapability,
    ModelLimits,
    ThinkingCapability,
    ThinkingEffort,
)
from sagent.types.cost import PriceKey, ServiceTier, TokenPrice


_TODAY = date(2026, 9, 24)

_TRANSPORT_THINKING = ThinkingCapability(
    effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
    budget=frozenset({"none", "auto", "fixed"}),
    output=frozenset({"none", "text", "redacted"}),
)


@pytest.fixture(autouse=True)
def rebuild_catalog() -> None:
    """Build rows per test; a cached catalog hides edits to its row helpers."""
    openai.models.cache_clear()
    openai.subscription_models.cache_clear()


def test_openai_metadata_does_not_leak_from_new_models_to_legacy_rows() -> None:
    models = openai.models()
    assert models["astra-6.0"].approx_chars_per_token == 3.71
    assert models["gpt-4"].knowledge_cutoff == "December 1, 2023"


# (tier, input, cached input, cache write, output), from the standard / flex /
# fast tabs of https://developers.openai.com/api/docs/pricing, 2026-09-24.
@pytest.mark.parametrize(
    ("model_id", "tier", "published"),
    [
        ("astra-6.0", "auto", (10.0, 1.0, 12.5, 50.0)),
        ("astra-6.0", "flex", (5.0, 0.5, 6.25, 25.0)),
        ("astra-6.0", "priority", (20.0, 2.0, 25.0, 100.0)),
        ("luna-6.0", "flex", (0.05, 0.005, 0.0625, 0.25)),
        ("luna-6.0", "priority", (0.2, 0.02, 0.25, 1.0)),
        ("terra-5.6", "flex", (1.0, 0.1, 1.25, 6.0)),
        ("gpt-5.4", "flex", (1.25, 0.13, 0.0, 7.5)),
        ("gpt-4o", "priority", (4.25, 2.125, 0.0, 17.0)),
        ("gpt-5.2", "priority", (3.5, 0.35, 0.0, 28.0)),
    ],
)
def test_openai_tiers_bill_the_published_rate(
    model_id: str,
    tier: ServiceTier,
    published: tuple[float, float, float, float],
) -> None:
    price = openai.models()[model_id].prices[PriceKey(tier)]
    assert (price.request, price.cache_read, price.cache_write, price.response) == (
        published
    )


@pytest.mark.parametrize(
    ("model_id", "offered"),
    [
        ("astra-6.0", {"auto", "default", "flex", "priority"}),
        ("gpt-5.5-pro", {"auto", "default", "flex"}),
        ("gpt-4.1", {"auto", "default", "priority"}),
        ("o1", {"auto", "default"}),
    ],
)
def test_openai_offers_only_the_published_tiers(
    model_id: str,
    offered: set[str],
) -> None:
    """A model absent from a tier's pricing tab cannot be billed at it."""
    assert openai.models()[model_id].service_tier == offered


def test_openai_long_prompts_bill_the_published_long_context_rate() -> None:
    """GPT-6 publishes its >272K column; 2x input / 1.5x output reproduces it."""
    prices = openai.models()["luna-6.0"].prices
    short = prices.rate(service_tier="auto", prompt_tokens=272_000, at=_TODAY)
    long = prices.rate(service_tier="auto", prompt_tokens=272_001, at=_TODAY)
    assert (short.request, short.response) == (0.1, 0.5)
    assert (long.request, long.cache_read, long.cache_write, long.response) == (
        pytest.approx(0.2),
        pytest.approx(0.02),
        pytest.approx(0.25),
        pytest.approx(0.75),
    )
    fast_long = prices.rate(service_tier="priority", prompt_tokens=300_000, at=_TODAY)
    assert (fast_long.request, fast_long.response) == (
        pytest.approx(0.4),
        pytest.approx(1.5),
    )


@pytest.mark.parametrize(
    ("reported", "tier"),
    [
        (None, "auto"),
        ("auto", "auto"),
        ("default", "auto"),
        ("flex", "flex"),
        ("priority", "priority"),
        ("fast", "priority"),
    ],
)
def test_openai_bills_the_tier_it_reports_serving(
    reported: str | None,
    tier: ServiceTier,
) -> None:
    assert openai.served_tier(reported) == tier


def test_openai_rejects_a_tier_it_cannot_price() -> None:
    with pytest.raises(ValueError, match=r"^unpriced OpenAI service_tier 'scale'$"):
        _ = openai.served_tier("scale")


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_reasoning_effort_sends_the_selected_level_verbatim(
    effort: ThinkingEffort,
) -> None:
    assert openai.reasoning_effort(effort) == effort


@pytest.mark.parametrize("effort", ["none", "min"])
def test_reasoning_effort_has_no_spelling_for_sending_no_knob(
    effort: ThinkingEffort,
) -> None:
    """``none`` omits ``reasoning`` and no row offers ``min``; neither is a level."""
    with pytest.raises(ValueError, match=rf"^effort {effort!r} is not sent"):
        _ = openai.reasoning_effort(effort)


def test_no_openai_row_offers_min() -> None:
    """No model page lists ``minimal``, so no row may let a caller select it."""
    assert all("min" not in row.thinking.effort for row in openai.models().values())


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


def test_only_long_context_rows_price_a_band_above_272k() -> None:
    models = openai.models()
    assert PriceKey("auto", 272_000) in models["luna-6.0"].prices
    assert PriceKey("auto", 272_000) not in models["gpt-4"].prices


def test_cards_carry_every_published_column() -> None:
    price = openai.models()["astra-6.0"].prices[PriceKey("auto")]
    assert price == TokenPrice(
        request=10.0,
        response=50.0,
        cache_write=12.5,
        cache_write_1h=0.0,
        cache_read=1.0,
    )


def test_api_offers_every_tier_over_key_auth() -> None:
    assert openai.api() == ModelCapability(
        thinking=_TRANSPORT_THINKING,
        service_tier=frozenset({"auto", "default", "flex", "priority"}),
    )


def test_subscription_models_clamp_to_the_chatgpt_backend() -> None:
    row = openai.subscription_models()["astra-6.0"]
    assert set(row.context) == {""}
    assert row.context[""].max_request_tokens == 272_000
    assert row.context[""].max_response_tokens == 32_000


def test_default_limits_serve_the_published_input_cap_and_a_272k_cap() -> None:
    """Serve the published "Maximum input tokens: 922,000" under a 1,050,000 window."""
    limits = ModelLimits(
        max_request_tokens=922_000,
        max_response_tokens=128_000,
        max_request_bytes=512 * 1024 * 1024,
        max_image_edge_px=65_535,
    )
    assert openai.models()["astra-6.0"].context == {
        "": limits,
        "+272k": replace(limits, max_request_tokens=272_000),
    }


def test_tiled_image_limits_cap_the_body_and_image_edge() -> None:
    assert openai.models()["gpt-4o"].context == {
        "": ModelLimits(
            max_request_tokens=128_000,
            max_response_tokens=16_384,
            max_request_bytes=512 * 1024 * 1024,
            max_image_edge_px=2048,
        ),
    }


def test_sol_6_1_bills_its_published_card_and_is_the_default() -> None:
    """Cached input is 5% of input on 6.1 Sol, half of every other GPT-6 card."""
    sol = openai.models()["sol-6.1"]
    assert sol.wire_model_id == "gpt-6.1-sol"
    assert sol.knowledge_cutoff == "April 30, 2026"
    assert sol.prices[PriceKey("auto")] == TokenPrice(
        request=2.0,
        response=10.0,
        cache_write=2.5,
        cache_write_1h=0.0,
        cache_read=0.1,
    )
    assert sol.prices[PriceKey("flex")].cache_read == 0.05
    assert sol.prices[PriceKey("priority")].cache_read == 0.2
    assert sol.thinking.effort == frozenset({"low", "medium", "high", "xhigh", "max"})
    assert openai.models()["default"] is sol


@pytest.mark.parametrize(
    ("model_id", "tier"),
    [("gpt-5.5", "priority"), ("gpt-5.4", "priority"), ("gpt-5.5-pro", "flex")],
)
def test_a_tier_the_table_leaves_unbanded_has_no_long_card(
    model_id: str,
    tier: ServiceTier,
) -> None:
    """The pricing table prints "-" for these long columns."""
    assert PriceKey(tier, 272_000) not in openai.models()[model_id].prices


def test_a_published_long_card_overrides_the_ratio() -> None:
    """gpt-5.4 flex long cached is $0.25, not 2x the rounded $0.13."""
    long = openai.models()["gpt-5.4"].prices[PriceKey("flex", 272_000)]
    assert (long.request, long.cache_read, long.response) == (2.5, 0.25, 11.25)


@pytest.mark.parametrize(
    ("model_id", "width", "height", "tokens"),
    [
        # images-vision worked examples for gpt-6-astra.
        ("gpt-6-astra", 1024, 1024, 1229),
        ("gpt-6-astra", 4096, 512, 2458),
        ("sol-5.6", 33, 65, 8),
        # 8000x8000 shrinks into gpt-5.5's 10,000-patch budget.
        ("gpt-5.5", 8000, 8000, 12_000),
        ("gpt-4o", 33, 65, 255),
        ("gpt-5.1", 1024, 1024, 70 + 140 * 4),
        # Text-only rows bill no image.
        ("o3-mini", 33, 65, 0),
        ("gpt-4", 33, 65, 0),
        # Image input but no published formula: the 512px-tile estimate.
        ("gpt-5.5-pro", 33, 65, 85 + 170),
        ("gpt-4-turbo", 33, 65, 85 + 170),
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
        ("gpt-6-astra", "o200k_base"),
        ("gpt-4o", "o200k_base"),
        ("gpt-4", "cl100k_base"),
        ("gpt-4-turbo", "cl100k_base"),
        ("kimi-k3", None),
    ],
)
def test_tokenizer_is_catalog_data(model_id: str, encoding: str | None) -> None:
    assert openai.tokenizer(model_id) == encoding


@pytest.mark.parametrize(
    ("model_id", "all_turns"),
    [("gpt-6.1-sol", True), ("gpt-5.6-luna", True), ("gpt-5.5", False)],
)
def test_reasoning_context_spans_turns_only_on_5_6_and_later(
    model_id: str,
    all_turns: bool,
) -> None:
    assert openai.keeps_reasoning_across_turns(model_id) is all_turns


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
