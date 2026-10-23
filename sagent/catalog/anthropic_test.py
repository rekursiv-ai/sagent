"""Anthropic transport restrictions and usage accounting."""

from datetime import date

import dataclasses

import pytest

from sagent.catalog import anthropic
from sagent.catalog.loader import models
from sagent.types.capability import ThinkingBudget
from sagent.types.cost import PriceKey, TokenCount


# Published facts, transcribed from the pages ``assets/anthropic.yaml`` cites
# (cutoffs and prices 2026-10-21; thinking and fast mode measured against the
# live API). Structural checks pass through a mistyped number; these do not.


@pytest.mark.parametrize(
    ("model_id", "cutoff"),
    [
        ("fable-5.1", "June 2026"),
        ("fable-5.0", "January 2026"),
        ("mythos-5.1", "June 2026"),
        ("mythos-5.0", "January 2026"),
        ("opus-5.5", "June 2026"),
        ("opus-5.0", "May 2026"),
        ("opus-4.8", "January 2026"),
        ("opus-4.7", "January 2026"),
        ("opus-4.6", "May 2025"),
        ("opus-4.5", "May 2025"),
        ("sonnet-5.5", "June 2026"),
        ("sonnet-5.0", "January 2026"),
        ("sonnet-4.6", "August 2025"),
        ("sonnet-4.5", "January 2025"),
        ("haiku-4.5", "February 2025"),
    ],
)
def test_every_row_states_its_published_cutoff(model_id: str, cutoff: str) -> None:
    assert models("anthropic")[model_id].knowledge_cutoff == cutoff


@pytest.mark.parametrize(
    ("model_id", "rates"),
    [
        ("fable-5.1", (10.0, 50.0, 12.5, 20.0, 0.25)),
        ("fable-5.0", (10.0, 50.0, 12.5, 20.0, 1.0)),
        ("opus-5.5", (4.0, 20.0, 5.0, 8.0, 0.2)),
        ("opus-5.0", (5.0, 25.0, 6.25, 10.0, 0.5)),
        ("sonnet-5.5", (2.0, 10.0, 2.5, 4.0, 0.2)),
        ("sonnet-5.0", (2.0, 10.0, 2.5, 4.0, 0.2)),
        ("sonnet-4.6", (3.0, 15.0, 3.75, 6.0, 0.3)),
        ("haiku-5.5", (0.1, 0.5, 0.125, 0.2, 0.01)),
        ("haiku-4.5", (1.0, 5.0, 1.25, 2.0, 0.1)),
    ],
)
def test_a_row_bills_the_published_standard_card(
    model_id: str,
    rates: tuple[float, float, float, float, float],
) -> None:
    card = models("anthropic")[model_id].prices[PriceKey("auto")]
    assert (
        card.request,
        card.response,
        card.cache_write,
        card.cache_write_1h,
        card.cache_read,
    ) == rates


def test_fast_mode_cache_rates_ride_the_fast_input_rate() -> None:
    """Cache multipliers apply on top of fast pricing; unscaled ones billed half."""
    fast = models("anthropic")["opus-5.0"].prices[PriceKey("priority")]
    assert (fast.request, fast.response) == (10.0, 50.0)
    assert (fast.cache_write, fast.cache_write_1h, fast.cache_read) == (12.5, 20.0, 1.0)


def test_only_the_fast_mode_opus_models_offer_priority() -> None:
    """4.7 lost fast mode on 2026-07-24 and 4.6 never had it; both bill standard."""
    table = models("anthropic")
    fast = {name for name, row in table.items() if "priority" in row.service_tier}
    assert fast == {"opus-5.5", "opus-5.0", "opus-4.8"}


def test_dated_and_banded_cards_bill_their_published_rates() -> None:
    table = models("anthropic")
    sonnet = table["sonnet-5.5"].prices
    assert (
        sonnet.rate(
            service_tier="auto",
            prompt_tokens=1,
            at=date(2026, 10, 6),
        ).cache_read
        == 0.2
    )
    assert (
        sonnet.rate(
            service_tier="auto",
            prompt_tokens=1,
            at=date(2026, 10, 7),
        ).cache_read
        == 0.1
    )
    haiku = table["haiku-5.5"].prices
    at = date(2026, 10, 22)
    assert haiku.rate(service_tier="auto", prompt_tokens=100_000, at=at).request == 0.1
    assert haiku.rate(service_tier="auto", prompt_tokens=100_001, at=at).request == 0.5


def test_roles_name_the_newest_of_each_tier() -> None:
    table = models("anthropic")
    assert {role: table[role].model_id for role in table.roles} == {
        "best": "fable-5.1",
        "default": "opus-5.5",
        "everyday": "sonnet-5.5",
        "utility": "haiku-5.5",
    }


@pytest.mark.parametrize("model_id", ["fable-5.1", "opus-5.5", "sonnet-5.0"])
def test_1m_models_serve_1m_and_offer_200k(model_id: str) -> None:
    context = models("anthropic")[model_id].context
    assert context.keys() == {"", "+200k"}
    assert context[""].max_request_tokens == 1_000_000
    assert context["+200k"].max_request_tokens == 200_000
    assert context[""].max_response_tokens == 128_000


@pytest.mark.parametrize("model_id", ["opus-4.5", "sonnet-4.5", "haiku-4.5"])
def test_pre_4_6_models_have_only_a_200k_window(model_id: str) -> None:
    """A 1M row on these 400s; they are 200k / 64k models."""
    context = models("anthropic")[model_id].context
    assert context.keys() == {""}
    assert context[""].max_request_tokens == 200_000
    assert context[""].max_response_tokens == 64_000


@pytest.mark.parametrize(
    ("model_id", "budget"),
    [
        # ``enabled`` 400s from 4.7 on: adaptive only.
        ("fable-5.1", {"none", "auto"}),
        ("opus-4.8", {"none", "auto"}),
        ("opus-4.6", {"none", "auto", "fixed"}),
        # The 4.5 generation 400s on ``adaptive``.
        ("opus-4.5", {"none", "fixed"}),
        ("sonnet-4.5", {"none", "fixed"}),
        ("haiku-4.5", {"none", "fixed"}),
    ],
)
def test_each_generation_offers_its_measured_budget_modes(
    model_id: str,
    budget: set[ThinkingBudget],
) -> None:
    assert models("anthropic")[model_id].thinking.budget == budget


def test_xhigh_effort_is_a_4_7_and_later_level() -> None:
    table = models("anthropic")
    assert table["fable-5.1"].thinking.effort == {
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }
    assert "xhigh" not in table["opus-4.6"].thinking.effort


@pytest.mark.parametrize(
    ("model_id", "edge_px", "chars_per_token"),
    [
        ("opus-5.5", 2576, 2.38),
        ("opus-4.7", 2576, 2.38),
        ("opus-4.6", 1568, 3.12),
        ("sonnet-4.5", 1568, 3.12),
    ],
)
def test_image_limits_and_tokenizer_follow_the_generation(
    model_id: str,
    edge_px: int,
    chars_per_token: float,
) -> None:
    row = models("anthropic")[model_id]
    assert row.approx_chars_per_token == chars_per_token
    for limits in row.context.values():
        assert limits.max_image_edge_px == edge_px
        # 10 MB base64-encoded on the wire is 7.5 MB raw.
        assert limits.max_image_bytes == 7_500_000
        assert limits.max_request_bytes == 32 * 1024 * 1024
        assert not limits.request_betas


@pytest.mark.parametrize(
    ("model_id", "wire_id"),
    [
        ("fable-5.1", "claude-fable-5-1"),
        ("mythos-5.1", "claude-mythos-5-1"),
        ("opus-5.5", "claude-opus-5-5"),
        ("opus-5.0", "claude-opus-5"),
        ("sonnet-5.0", "claude-sonnet-5"),
        ("haiku-4.5", "claude-haiku-4-5"),
    ],
)
def test_catalog_names_map_to_vendor_wire_ids(model_id: str, wire_id: str) -> None:
    table = models("anthropic")
    assert table[model_id].wire_model_id == wire_id
    assert table[wire_id] is table[model_id]


def test_the_api_offers_every_knob_and_both_cache_lifetimes() -> None:
    api = anthropic.api()
    assert api.thinking.effort == {
        "none",
        "min",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }
    assert api.thinking.budget == {"none", "auto", "fixed"}
    assert api.thinking.output == {"none", "text", "redacted"}
    assert api.cache_ttl_sec == {300.0, 3600.0}
    assert api.manage_context_server_side == {False, True}
    assert api.retries_internally
    assert api.service_tier == {"auto", "default", "priority"}
    assert not api.account_auth


def test_a_subscription_is_the_api_billed_to_the_account() -> None:
    assert anthropic.subscription() == dataclasses.replace(
        anthropic.api(),
        account_auth=True,
    )


def test_the_cli_offers_no_effort_and_manages_its_own_context() -> None:
    cli = anthropic.cli()
    assert cli.thinking.effort == {"none"}
    assert cli.thinking.budget == {"none", "auto", "fixed"}
    assert cli.thinking.output == {"none", "text"}
    assert cli.manage_context_server_side == {True}


def test_a_split_usage_block_bills_each_cache_lifetime() -> None:
    usage = {
        "input_tokens": 11,
        "output_tokens": 7,
        "cache_creation_input_tokens": 50,
        "cache_creation": {"ephemeral_1h_input_tokens": 20},
        "cache_read_input_tokens": 3,
    }
    assert anthropic.usage_tokens(usage, cache_ttl_sec=300.0) == TokenCount(
        request=11,
        response=7,
        cache_write=30,
        cache_write_1h=20,
        cache_read=3,
    )


@pytest.mark.parametrize(
    ("cache_ttl_sec", "written_1h"),
    [(300.0, 0), (3599.0, 0), (3600.0, 50)],
)
def test_an_unsplit_write_bills_at_the_lifetime_asked_for(
    cache_ttl_sec: float,
    written_1h: int,
) -> None:
    usage = {"cache_creation_input_tokens": 50}
    tokens = anthropic.usage_tokens(usage, cache_ttl_sec=cache_ttl_sec)
    assert (tokens.cache_write, tokens.cache_write_1h) == (50 - written_1h, written_1h)


def test_an_empty_usage_block_counts_nothing() -> None:
    assert anthropic.usage_tokens({}, cache_ttl_sec=3600.0) == TokenCount()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
