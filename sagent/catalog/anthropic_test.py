"""Anthropic catalog rows, checked against the published docs (2026-10-21)."""

from __future__ import annotations

from datetime import date

import dataclasses

import pytest

from sagent.catalog import anthropic
from sagent.types.cost import PriceKey, TokenCount, TokenPrice


def test_sonnet_5_5_bills_its_published_card() -> None:
    """https://platform.claude.com/docs/en/models/sonnet-5-5/overview."""
    assert anthropic.models()["sonnet-5.5"].prices[PriceKey("auto")] == TokenPrice(
        request=2.0,
        response=10.0,
        cache_write=2.5,
        cache_write_1h=4.0,
        cache_read=0.2,
    )


def test_sonnet_5_5_cache_reads_halve_from_2026_10_07() -> None:
    """https://www.anthropic.com/claude-haiku-5-5, "Further updates"."""
    prices = anthropic.models()["sonnet-5.5"].prices
    before = prices.rate(service_tier="auto", prompt_tokens=1, at=date(2026, 10, 6))
    after = prices.rate(service_tier="auto", prompt_tokens=1, at=date(2026, 10, 7))
    assert (before.cache_read, after.cache_read) == (0.2, 0.1)
    assert dataclasses.replace(after, cache_read=0.2) == before


@pytest.mark.parametrize(
    ("prompt_tokens", "card"),
    [
        (
            100_000,
            TokenPrice(
                request=0.1,
                response=0.5,
                cache_write=0.125,
                cache_write_1h=0.2,
                cache_read=0.01,
            ),
        ),
        (
            100_001,
            TokenPrice(
                request=0.5,
                response=2.5,
                cache_write=0.625,
                cache_write_1h=1.0,
                cache_read=0.05,
            ),
        ),
    ],
)
def test_haiku_5_5_bills_by_prompt_length(prompt_tokens: int, card: TokenPrice) -> None:
    """https://platform.claude.com/docs/en/about-claude/pricing, long context."""
    prices = anthropic.models()["haiku-5.5"].prices
    at = date(2026, 10, 21)
    assert prices.rate(service_tier="auto", prompt_tokens=prompt_tokens, at=at) == card


def test_haiku_5_5_is_the_newest_haiku() -> None:
    """https://platform.claude.com/docs/en/models/haiku-5-5/overview."""
    rows = anthropic.models()
    row = rows["haiku"]
    assert (row.model_id, row.wire_model_id) == ("haiku-5.5", "claude-haiku-5-5")
    assert row.knowledge_cutoff == "June 2026"
    assert "medium" in row.thinking.effort
    assert "priority" not in row.service_tier
    limits = row.context[""]
    assert (limits.max_request_tokens, limits.max_response_tokens) == (
        1_000_000,
        128_000,
    )


@pytest.mark.parametrize(
    ("role", "model_id"),
    [
        ("best", "fable-5.1"),
        ("default", "opus-5.5"),
        ("everyday", "sonnet-5.5"),
        ("utility", "haiku-5.5"),
    ],
)
def test_roles_climb_one_model_line_each(role: str, model_id: str) -> None:
    """https://platform.claude.com/docs/en/about-claude/models/choosing-a-model."""
    assert anthropic.models()[role].model_id == model_id


def test_sonnet_5_5_is_the_newest_sonnet() -> None:
    rows = anthropic.models()
    assert rows["sonnet-5.5"].wire_model_id == "claude-sonnet-5-5"
    assert rows["sonnet"].model_id == "sonnet-5.5"
    limits = rows["sonnet-5.5"].context[""]
    assert (limits.max_request_tokens, limits.max_response_tokens) == (
        1_000_000,
        128_000,
    )


@pytest.mark.parametrize(
    ("model_id", "cutoff"),
    [
        ("sonnet-5.5", "June 2026"),
        ("mythos-5.1", "June 2026"),
        ("mythos-5.0", "January 2026"),
    ],
)
def test_new_rows_state_their_published_cutoff(model_id: str, cutoff: str) -> None:
    assert anthropic.models()[model_id].knowledge_cutoff == cutoff


@pytest.mark.parametrize(
    ("mythos", "fable", "wire"),
    [
        ("mythos-5.1", "fable-5.1", "claude-mythos-5-1"),
        ("mythos-5.0", "fable-5.0", "claude-mythos-5"),
    ],
)
def test_mythos_shares_fables_specs_and_prices(
    mythos: str,
    fable: str,
    wire: str,
) -> None:
    """Each Mythos overview page: "It shares Claude Fable ...'s specifications"."""
    rows = anthropic.models()
    assert rows[mythos].wire_model_id == wire
    assert rows[mythos].prices == rows[fable].prices
    assert rows[mythos].context == rows[fable].context
    assert rows[mythos].thinking == rows[fable].thinking
    assert "priority" not in rows[mythos].service_tier


def test_mythos_family_resolves_to_its_newest_row() -> None:
    assert anthropic.models()["mythos"].model_id == "mythos-5.1"


def test_no_row_needs_a_context_beta() -> None:
    """For every 1M model, "1M is the default: you don't need a beta header"."""
    for model_id, row in anthropic.models().items():
        for tag, limits in row.context.items():
            assert not limits.request_betas, (model_id, tag)


def test_image_cap_leaves_room_for_base64_under_10_mb() -> None:
    """The vision page caps 10 MB base64-encoded; the cap applies to raw bytes."""
    for row in anthropic.models().values():
        raw = row.context[""].max_image_bytes
        assert raw == 7_500_000
        assert -(-raw // 3) * 4 <= 10_000_000


@pytest.mark.parametrize(
    "model_id",
    [
        "fable-5.1",
        "fable-5.0",
        "mythos-5.1",
        "mythos-5.0",
        "opus-5.5",
        "opus-5.0",
        "opus-4.8",
        "opus-4.7",
        "sonnet-5.5",
        "sonnet-5.0",
        "haiku-5.5",
    ],
)
def test_omitted_display_models_still_offer_summarized_text(model_id: str) -> None:
    """``display`` defaults to omitted here; ``summarized`` opts back in."""
    output = anthropic.models()[model_id].thinking.output
    assert output == frozenset({"none", "text", "redacted"})


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
