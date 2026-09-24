"""Tests for the token count / price / cost calculus."""

from __future__ import annotations

from datetime import date

import pytest

from sagent.types.cost import (
    PriceCatalog,
    PriceKey,
    TokenCost,
    TokenCount,
    TokenPrice,
)


_TODAY = date(2026, 9, 24)


def _price(
    request: float,
    *,
    response: float = 0.0,
    cache_read: float = 0.0,
) -> TokenPrice:
    return TokenPrice(
        request=request,
        response=response,
        cache_write=0.0,
        cache_write_1h=0.0,
        cache_read=cache_read,
    )


def test_count_is_default_constructible() -> None:
    assert TokenCount().total == 0


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


def test_multiplication_commutes() -> None:
    price = _price(5.0)
    tokens = TokenCount(request=1_000_000)
    assert price * tokens == tokens * price


def test_a_price_must_state_every_rate() -> None:
    """An unstated rate would bill $0; construction refuses it."""
    with pytest.raises(TypeError):
        _ = TokenPrice(request=5.0)  # ty: ignore[missing-argument] -- the missing rates are the point.  # pyright: ignore[reportCallIssue] -- same.


def _catalog() -> PriceCatalog:
    return PriceCatalog(
        {
            PriceKey("auto"): _price(2.0),
            PriceKey("auto", 200_000): _price(4.0),
            PriceKey("priority"): _price(6.0),
        },
    )


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
    """Flex once billed at the standard rate -- 2x -- through this fallback."""
    with pytest.raises(KeyError):
        _ = _catalog().rate(service_tier="flex", prompt_tokens=10, at=_TODAY)


def test_a_priced_tier_prices_every_band() -> None:
    """A tier with only a base band bills every prompt size at it."""
    rate = _catalog().rate(service_tier="priority", prompt_tokens=10**9, at=_TODAY)
    assert rate.request == 6.0


def test_the_latest_effective_price_wins() -> None:
    catalog = PriceCatalog(
        {
            PriceKey("auto"): _price(1.0),
            PriceKey("auto", effective_date=date(2027, 1, 1)): _price(2.0),
        },
    )
    assert (
        catalog.rate(
            service_tier="auto",
            prompt_tokens=1,
            at=date(2026, 12, 31),
        ).request
        == 1.0
    )
    assert (
        catalog.rate(service_tier="auto", prompt_tokens=1, at=date(2027, 1, 1)).request
        == 2.0
    )


def test_a_request_before_any_price_raises() -> None:
    catalog = PriceCatalog(
        {PriceKey("auto", effective_date=date(2027, 1, 1)): _price(1.0)},
    )
    with pytest.raises(KeyError):
        _ = catalog.rate(service_tier="auto", prompt_tokens=1, at=_TODAY)


def test_an_empty_catalog_raises_rather_than_billing_zero() -> None:
    with pytest.raises(KeyError):
        _ = PriceCatalog().rate(service_tier="auto", prompt_tokens=1, at=_TODAY)


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
    assert PriceCatalog().service_tiers == frozenset()


def test_indexing_is_exact() -> None:
    catalog = _catalog()
    assert catalog[PriceKey("auto", 200_000)].request == 4.0
    with pytest.raises(KeyError):
        _ = catalog[PriceKey("auto", 199_999)]
    assert len(catalog) == 3
    assert list(catalog) == sorted(catalog)


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
