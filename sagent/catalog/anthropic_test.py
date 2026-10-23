"""Anthropic transport restrictions and usage accounting."""

import dataclasses

import pytest

from sagent.catalog import anthropic
from sagent.types.cost import TokenCount


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
