"""OpenAI transport, subscription projection, and wire behavior."""

from collections.abc import Iterator
from dataclasses import replace

import pytest

from sagent.catalog import openai
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
                "sol-5.6",
                "gpt-5.5",
                "gpt-5.1",
                "gpt-omni-4.0",
                "o-mini-3.0",
                "gpt-4.0",
                "gpt-pro-5.5",
                "gpt-turbo-4.0",
                "sol-6.1",
                "luna-5.6",
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


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
