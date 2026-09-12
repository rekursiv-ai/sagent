"""Tests for ``agent.cache_waste``: per-turn prompt-cache miss detection."""

from __future__ import annotations

from sagent.agent.cache_waste import (
    CacheMiss,
    detect_cache_miss,
    summarize_cache_waste,
)
from sagent.types.cost import TokenCost, TokenCount


def _spend(tokens: TokenCount) -> TokenCost:
    """Flat $1/token-bucket-unit stub: request costs 10x cache_read."""
    return TokenCost(
        request=tokens.request * 10.0,
        cache_read=tokens.cache_read * 1.0,
    )


def test_detect_cache_miss_none_on_first_turn() -> None:
    """No prior prefix (all-zero ``previous``) means nothing could be missed."""
    assert (
        detect_cache_miss(
            TokenCount(),
            TokenCount(request=5_000),
            idle_sec=1.0,
            cache_ttl_sec=300.0,
            model_changed=False,
            spend=_spend,
        )
        is None
    )


def test_detect_cache_miss_none_on_model_change() -> None:
    """Token counts aren't comparable across a tokenizer change."""
    assert (
        detect_cache_miss(
            TokenCount(cache_read=50_000),
            TokenCount(request=50_000),
            idle_sec=1.0,
            cache_ttl_sec=300.0,
            model_changed=True,
            spend=_spend,
        )
        is None
    )


def test_detect_cache_miss_none_below_noise_floor() -> None:
    """A delta at or below the breakpoint-granularity noise floor is ignored."""
    assert (
        detect_cache_miss(
            TokenCount(cache_read=10_000),
            TokenCount(cache_read=9_500, request=500),
            idle_sec=1.0,
            cache_ttl_sec=300.0,
            model_changed=False,
            spend=_spend,
        )
        is None
    )


def test_detect_cache_miss_ttl_expired() -> None:
    """An idle gap past the cache TTL is an expected miss, not a mutation."""
    miss = detect_cache_miss(
        TokenCount(cache_read=10_000),
        TokenCount(request=10_000),
        idle_sec=600.0,
        cache_ttl_sec=300.0,
        model_changed=False,
        spend=_spend,
    )
    assert miss is not None
    assert miss.cause == "ttl_expired"
    assert miss.missed_tokens == 10_000
    assert miss.idle_sec == 600.0


def test_detect_cache_miss_prefix_mutated() -> None:
    """A miss within the TTL window points at a mutated prefix -- #361's bug."""
    miss = detect_cache_miss(
        TokenCount(cache_read=10_000),
        TokenCount(request=10_000),
        idle_sec=5.0,
        cache_ttl_sec=300.0,
        model_changed=False,
        spend=_spend,
    )
    assert miss is not None
    assert miss.cause == "prefix_mutated"
    assert miss.missed_tokens == 10_000


def test_detect_cache_miss_prices_the_delta() -> None:
    """Wasted cost is (fresh rate - cache rate) x missed tokens, per ``spend``."""
    miss = detect_cache_miss(
        TokenCount(cache_read=10_000),
        TokenCount(request=10_000),
        idle_sec=5.0,
        cache_ttl_sec=300.0,
        model_changed=False,
        spend=_spend,
    )
    assert miss is not None
    # _spend: 10_000 tokens as request = 100_000; as cache_read = 10_000.
    assert miss.wasted_cost_usd == 90_000.0


def test_detect_cache_miss_counts_prior_cache_write_and_read_in_prefix() -> None:
    """The prior prefix is request + cache_write + cache_read, not request alone.

    A response whose prior turn wrote a fresh cache entry (``cache_write``)
    should have that whole prefix servable as ``cache_read`` next turn; only
    counting ``request`` would under-detect misses on a just-warmed cache.
    """
    miss = detect_cache_miss(
        TokenCount(request=1_000, cache_write=8_000, cache_read=1_000),
        TokenCount(request=10_000),
        idle_sec=5.0,
        cache_ttl_sec=300.0,
        model_changed=False,
        spend=_spend,
    )
    assert miss is not None
    assert miss.missed_tokens == 10_000


def test_summarize_cache_waste_empty() -> None:
    totals = summarize_cache_waste([])
    assert totals.missed_tokens == 0
    assert totals.wasted_cost_usd == 0.0
    assert totals.miss_count == 0


def test_summarize_cache_waste_aggregates_by_cause() -> None:
    misses = [
        CacheMiss(
            missed_tokens=1_000,
            wasted_cost_usd=9.0,
            idle_sec=1.0,
            cause="prefix_mutated",
        ),
        CacheMiss(
            missed_tokens=2_000,
            wasted_cost_usd=18.0,
            idle_sec=400.0,
            cause="ttl_expired",
        ),
        CacheMiss(
            missed_tokens=3_000,
            wasted_cost_usd=27.0,
            idle_sec=2.0,
            cause="prefix_mutated",
        ),
    ]
    totals = summarize_cache_waste(misses)
    assert totals.missed_tokens == 6_000
    assert totals.wasted_cost_usd == 54.0
    assert totals.miss_count == 3
    assert totals.ttl_expired_count == 1
    assert totals.prefix_mutated_count == 2


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
