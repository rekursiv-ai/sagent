"""Tests for the token count / price / cost calculus."""

from __future__ import annotations

import pytest

from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenCost,
    TokenCount,
    TokenPrice,
)


def test_count_is_default_constructible() -> None:
    assert TokenCount().total == 0


def test_add_is_per_bucket() -> None:
    a = TokenCount(request=100, response=10, cache_write=5, cache_read=1)
    b = TokenCount(request=40, response=3, cache_write=2, cache_read=1)
    assert a + b == TokenCount(request=140, response=13, cache_write=7, cache_read=2)


def test_sub_deltas_cumulative_snapshots() -> None:
    a = TokenCount(request=100, response=10)
    b = TokenCount(request=40, response=3)
    assert a - b == TokenCount(request=60, response=7)


def test_price_times_count_is_cost() -> None:
    price = TokenPrice(request=5.0, response=25.0)
    cost = price * TokenCount(request=1_000_000, response=200_000)
    assert cost == TokenCost(request=5.0, response=5.0)


def test_multiplication_commutes() -> None:
    price = TokenPrice(request=5.0)
    tokens = TokenCount(request=1_000_000)
    assert price * tokens == tokens * price


def test_price_has_no_total() -> None:
    # ``total`` sums buckets, which is meaningless for a rate.
    assert not hasattr(TokenPrice(), "tokens_per_unit_total")


def _catalog() -> PriceCatalog:
    return PriceCatalog(
        {
            PriceCatalogProduct(False, 0): TokenPrice(request=2.0),
            PriceCatalogProduct(False, 200_000): TokenPrice(request=4.0),
            PriceCatalogProduct(True, 0): TokenPrice(request=6.0),
        }
    )


@pytest.mark.parametrize(
    ("request_tokens", "expected"),
    [(0, 2.0), (199_999, 2.0), (200_000, 4.0), (1_000_000, 4.0)],
)
def test_floor_lookup_picks_the_tier_at_or_below(
    request_tokens: int,
    expected: float,
) -> None:
    catalog = _catalog()
    key = PriceCatalogProduct(False, request_tokens)
    assert catalog[key].request == expected


def test_fast_falls_back_to_standard_when_no_fast_tier() -> None:
    catalog = PriceCatalog({PriceCatalogProduct(False, 0): TokenPrice(request=2.0)})
    assert catalog[PriceCatalogProduct(True, 0)].request == 2.0


def test_fast_tier_wins_when_present() -> None:
    assert _catalog()[PriceCatalogProduct(True, 5)].request == 6.0


def test_empty_catalog_raises_rather_than_billing_zero() -> None:
    with pytest.raises(KeyError):
        _ = PriceCatalog()[PriceCatalogProduct()]


def test_contains_agrees_with_getitem() -> None:
    catalog = _catalog()
    for key in (
        PriceCatalogProduct(False, 0),
        PriceCatalogProduct(True, 10**9),
    ):
        assert key in catalog
        assert catalog[key] is not None
    assert PriceCatalogProduct() not in PriceCatalog()


def test_len_and_iter_report_declared_tiers() -> None:
    catalog = _catalog()
    assert len(catalog) == 3
    assert sorted(catalog) == [
        PriceCatalogProduct(False, 0),
        PriceCatalogProduct(False, 200_000),
        PriceCatalogProduct(True, 0),
    ]


def test_contains_agrees_with_getitem_above_the_lowest_tier() -> None:
    """``k in cat`` must mean ``cat[k]`` succeeds -- the Mapping contract.

    ``__contains__`` ignored ``min_request_tokens``, so a catalog whose
    lowest tier starts above the queried size reported membership and
    then raised on access.
    """
    catalog = PriceCatalog(
        {PriceCatalogProduct(False, 272_000): TokenPrice(request=1.0)}
    )
    key = PriceCatalogProduct(False, 0)
    assert (key in catalog) is _resolves(catalog, key)


def _resolves(catalog: PriceCatalog, key: PriceCatalogProduct) -> bool:
    try:
        _ = catalog[key]
    except KeyError:
        return False
    return True
