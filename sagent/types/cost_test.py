"""Tests for token counts, prices, and the cost of one request."""

from __future__ import annotations

from dataclasses import MISSING, fields
from datetime import date

import operator

import pytest

from sagent.types.cost import (
    PriceCatalog,
    PriceKey,
    TokenCost,
    TokenCount,
    TokenPrice,
    TokenStats,
)


_TODAY = date(2026, 9, 24)


def _price(request: float, *, response: float = 0.0) -> TokenPrice:
    return TokenPrice(
        request=request,
        response=response,
        cache_write=0.0,
        cache_write_1h=0.0,
        cache_read=0.0,
    )


def _catalog() -> PriceCatalog:
    return PriceCatalog(
        {
            PriceKey("auto"): _price(2.0),
            PriceKey("auto", 200_000): _price(4.0),
            PriceKey("priority"): _price(6.0),
        },
    )


def _as_stats(value: TokenStats[int]) -> TokenStats[int]:
    return value


def test_a_count_defaults_to_zero() -> None:
    assert TokenCount().total == 0


def test_total_sums_all_five_meters() -> None:
    tokens = TokenCount(
        request=1,
        response=2,
        cache_write=4,
        cache_write_1h=8,
        cache_read=16,
    )
    assert tokens.total == 31


def test_add_is_per_meter() -> None:
    a = TokenCount(
        request=100,
        response=10,
        cache_write=5,
        cache_write_1h=3,
        cache_read=1,
    )
    b = TokenCount(
        request=40,
        response=3,
        cache_write=2,
        cache_write_1h=1,
        cache_read=1,
    )
    assert a + b == TokenCount(
        request=140,
        response=13,
        cache_write=7,
        cache_write_1h=4,
        cache_read=2,
    )


def test_sub_is_per_meter() -> None:
    a = TokenCount(
        request=100,
        response=10,
        cache_write=5,
        cache_write_1h=3,
        cache_read=9,
    )
    b = TokenCount(
        request=40,
        response=3,
        cache_write=2,
        cache_write_1h=1,
        cache_read=4,
    )
    assert a - b == TokenCount(
        request=60,
        response=7,
        cache_write=3,
        cache_write_1h=2,
        cache_read=5,
    )


def test_a_count_never_goes_negative_but_a_cost_may() -> None:
    """A restored total can fall below a snapshot; a count delta clamps."""
    assert TokenCount(request=1) - TokenCount(request=5, cache_read=2) == TokenCount()
    assert (TokenCost(request=1.0) - TokenCost(request=5.0)).request == -4.0


def test_non_matching_stats_defer_arithmetic() -> None:
    other = TokenStats[int](
        request=1,
        response=0,
        cache_write=0,
        cache_write_1h=0,
        cache_read=0,
    )
    stats = _as_stats(TokenCount(request=1))
    with pytest.raises(TypeError, match="unsupported operand"):
        operator.add(stats, other)
    with pytest.raises(TypeError, match="unsupported operand"):
        operator.sub(stats, other)


def test_prompt_counts_every_input_pool_but_not_the_response() -> None:
    tokens = TokenCount(
        request=1,
        response=100,
        cache_write=2,
        cache_write_1h=4,
        cache_read=8,
    )
    assert tokens.prompt == 15


def test_price_times_count_is_cost() -> None:
    price = TokenPrice(
        request=5.0,
        response=25.0,
        cache_write=6.25,
        cache_write_1h=10.0,
        cache_read=0.5,
    )
    cost = price * TokenCount(
        request=1_000_000,
        response=200_000,
        cache_write=1_000_000,
        cache_write_1h=1_000_000,
        cache_read=1_000_000,
    )
    assert cost == TokenCost(
        request=5.0,
        response=5.0,
        cache_write=6.25,
        cache_write_1h=10.0,
        cache_read=0.5,
    )


def test_the_unit_scales_the_rates() -> None:
    price = TokenPrice(
        request=1.0,
        response=0.0,
        cache_write=0.0,
        cache_write_1h=0.0,
        cache_read=0.0,
        tokens_per_unit=1_000,
    )
    assert (price * TokenCount(request=2_000)).request == 2.0


def test_multiplication_commutes() -> None:
    tokens = TokenCount(request=1_000_000)
    assert _price(5.0) * tokens == tokens * _price(5.0)


def test_a_price_must_state_every_rate() -> None:
    """An unstated rate would bill $0, so no rate has a default."""
    rates = [f for f in fields(TokenPrice) if f.name != "tokens_per_unit"]
    assert {f.name for f in rates} == {
        "request",
        "response",
        "cache_write",
        "cache_write_1h",
        "cache_read",
    }
    assert all(f.default is MISSING and f.default_factory is MISSING for f in rates)


@pytest.mark.parametrize(
    ("prompt_tokens", "expected"),
    [(0, 2.0), (200_000, 2.0), (200_001, 4.0), (1_000_000, 4.0)],
)
def test_a_band_applies_to_prompts_strictly_larger(
    prompt_tokens: int,
    expected: float,
) -> None:
    """Vendors publish "prompts > 200k": exactly 200k is still the base band."""
    rate = _catalog().rate(service_tier="auto", prompt_tokens=prompt_tokens, at=_TODAY)
    assert rate.request == expected


def test_default_bills_the_standard_rate() -> None:
    rate = _catalog().rate(service_tier="default", prompt_tokens=10, at=_TODAY)
    assert rate.request == 2.0


def test_an_unpriced_tier_raises_rather_than_billing_standard() -> None:
    with pytest.raises(KeyError) as excinfo:
        _catalog().rate(service_tier="flex", prompt_tokens=10, at=_TODAY)
    assert excinfo.value.args == (PriceKey("flex", 10, _TODAY),)


def test_a_tier_with_only_a_base_band_prices_every_size() -> None:
    rate = _catalog().rate(service_tier="priority", prompt_tokens=10**9, at=_TODAY)
    assert rate.request == 6.0


def test_the_latest_effective_price_wins() -> None:
    catalog = PriceCatalog(
        {
            PriceKey("auto"): _price(1.0),
            PriceKey("auto", effective_date=date(2027, 1, 1)): _price(2.0),
        },
    )
    before = catalog.rate(service_tier="auto", prompt_tokens=1, at=date(2026, 12, 31))
    on = catalog.rate(service_tier="auto", prompt_tokens=1, at=date(2027, 1, 1))
    assert (before.request, on.request) == (1.0, 2.0)


def test_a_request_before_any_price_raises() -> None:
    catalog = PriceCatalog(
        {PriceKey("auto", effective_date=date(2027, 1, 1)): _price(1.0)},
    )
    with pytest.raises(KeyError):
        catalog.rate(service_tier="auto", prompt_tokens=1, at=_TODAY)


def test_an_empty_catalog_raises_rather_than_billing_zero() -> None:
    with pytest.raises(KeyError):
        PriceCatalog().rate(service_tier="auto", prompt_tokens=1, at=_TODAY)


def test_cost_sizes_the_band_from_the_whole_prompt() -> None:
    """A mostly-cached 300k prompt is still a long prompt."""
    cost = _catalog().cost(
        TokenCount(request=10_000, cache_read=290_000),
        service_tier="auto",
        at=_TODAY,
    )
    assert cost.request == pytest.approx(0.04)


def test_service_tiers_are_the_priced_tiers() -> None:
    assert _catalog().service_tiers == {"auto", "default", "priority"}
    assert PriceCatalog({PriceKey("flex"): _price(1.0)}).service_tiers == {"flex"}
    assert PriceCatalog().service_tiers == frozenset()


def test_indexing_is_exact_and_ordered() -> None:
    catalog = _catalog()
    assert catalog[PriceKey("auto", 200_000)].request == 4.0
    with pytest.raises(KeyError):
        _ = catalog[PriceKey("auto", 199_999)]
    assert len(catalog) == 3
    assert list(catalog) == sorted(catalog)
    assert repr(catalog).startswith("PriceCatalog({")


def test_count_is_default_constructible() -> None:
    assert TokenCount().total == 0


def test_sub_deltas_cumulative_snapshots() -> None:
    a = TokenCount(request=100, response=10)
    b = TokenCount(request=40, response=3)
    assert a - b == TokenCount(request=60, response=7)


def test_prompt_counts_every_input_pool() -> None:
    tokens = TokenCount(
        request=1,
        response=100,
        cache_write=2,
        cache_write_1h=4,
        cache_read=8,
    )
    assert tokens.prompt == 15


def test_a_priced_tier_prices_every_band() -> None:
    """A tier with only a base band bills every prompt size at it."""
    rate = _catalog().rate(service_tier="priority", prompt_tokens=10**9, at=_TODAY)
    assert rate.request == 6.0


def test_indexing_is_exact() -> None:
    catalog = _catalog()
    assert catalog[PriceKey("auto", 200_000)].request == 4.0
    with pytest.raises(KeyError):
        _ = catalog[PriceKey("auto", 199_999)]
    assert len(catalog) == 3
    assert list(catalog) == sorted(catalog)
    assert repr(catalog).startswith("PriceCatalog({")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
