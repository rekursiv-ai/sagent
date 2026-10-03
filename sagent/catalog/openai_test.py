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
    assert models["astra-6"].approx_chars_per_token == 3.71
    assert models["gpt-4"].knowledge_cutoff is None


# (tier, input, cached input, cache write, output), from the standard / flex /
# fast tabs of https://developers.openai.com/api/docs/pricing, 2026-09-24.
@pytest.mark.parametrize(
    ("model_id", "tier", "published"),
    [
        ("astra-6", "auto", (10.0, 1.0, 12.5, 50.0)),
        ("astra-6", "flex", (5.0, 0.5, 6.25, 25.0)),
        ("astra-6", "priority", (20.0, 2.0, 25.0, 100.0)),
        ("luna-6", "flex", (0.05, 0.005, 0.0625, 0.25)),
        ("luna-6", "priority", (0.2, 0.02, 0.25, 1.0)),
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
        ("astra-6", {"auto", "default", "flex", "priority"}),
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
    prices = openai.models()["luna-6"].prices
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


@pytest.mark.parametrize(
    ("effort", "model_id", "wire"),
    [
        ("min", "terra-5.6", "none"),
        ("min", "gpt-5.6-terra", "none"),
        ("min", "unlisted-5.6", "none"),
        ("min", "luna-6", "low"),
        ("none", "astra-6", "low"),
        ("none", "gpt-6-astra", "low"),
        ("none", "default", "low"),
        ("none", "sol-6", "none"),
        ("xhigh", "astra-6", "xhigh"),
        ("max", "unlisted-model", "max"),
    ],
)
def test_reasoning_effort_maps_to_the_responses_vocabulary(
    effort: ThinkingEffort,
    model_id: str,
    wire: str,
) -> None:
    """A wire id resolves to its catalog row before the model rules apply."""
    assert openai.reasoning_effort(effort, model_id=model_id) == wire


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
    assert PriceKey("auto", 272_000) in models["luna-6"].prices
    assert PriceKey("auto", 272_000) not in models["gpt-4"].prices


def test_cards_carry_every_published_column() -> None:
    price = openai.models()["astra-6"].prices[PriceKey("auto")]
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
    row = openai.subscription_models()["astra-6"]
    assert set(row.context) == {""}
    assert row.context[""].max_request_tokens == 272_000
    assert row.context[""].max_response_tokens == 32_000


def test_default_limits_serve_a_patched_1m_window_and_a_272k_cap() -> None:
    limits = ModelLimits(
        max_request_tokens=1_050_000,
        max_response_tokens=128_000,
        max_request_bytes=512 * 1024 * 1024,
    )
    assert openai.models()["astra-6"].context == {
        "": limits,
        "+272k": replace(limits, max_request_tokens=272_000),
    }


def test_tiled_image_limits_cap_the_body_and_image_edge() -> None:
    assert openai.models()["gpt-4"].context == {
        "": ModelLimits(
            max_request_tokens=8_192,
            max_response_tokens=8_192,
            max_request_bytes=20 * 1024 * 1024,
            max_image_edge_px=2048,
            max_image_bytes=20 * 1024 * 1024,
        ),
    }


def test_sol_6_1_is_priced_by_a_stand_in_copy_of_the_sol_6_card() -> None:
    sol = openai.models()
    assert sol["sol-6.1"].wire_model_id == "gpt-6.1-sol"
    assert sol["sol-6.1"].prices == sol["sol-6"].prices
    assert sol["default"].model_id.startswith("astra-")


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
