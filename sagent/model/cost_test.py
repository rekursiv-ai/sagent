"""Tests for token counts, prices, the sizes prices name, and a request's cost."""

from __future__ import annotations

from datetime import date

import itertools
import math
import operator

import pytest

from sagent.model.context import ContextSize, ContextSizeName
from sagent.model.cost import (
    PriceCatalog,
    PriceCondition,
    TokenCost,
    TokenCount,
    TokenPrice,
    TokenStats,
)


def _as_stats(value: TokenStats[int]) -> TokenStats[int]:
    return value


def _banded(*tops: int, window: int = 1_000_000) -> PriceCatalog:
    """Return ``auto`` prices rising after each of ``tops``, within ``window``."""
    edges: tuple[int | float, ...] = (0, *(top + 1 for top in tops), math.inf)
    return PriceCatalog(
        prices={
            PriceCondition(token_size_range=(int(lo), hi)): TokenPrice(request=float(i))
            for i, (lo, hi) in enumerate(itertools.pairwise(edges))
        },
        window=window,
    )


def _size(name: ContextSizeName, tokens: int, shown: str) -> ContextSize:
    return ContextSize(context_name=name, tokens=tokens, display_name=shown)


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
    price = TokenPrice(request=1.0, tokens_per_unit=1_000)
    assert (price * TokenCount(request=2_000)).request == 2.0


def test_multiplication_commutes() -> None:
    tokens = TokenCount(request=1_000_000)
    assert TokenPrice(request=5.0) * tokens == tokens * TokenPrice(request=5.0)


def test_a_price_defaults_to_free() -> None:
    assert TokenPrice() * TokenCount(request=1, response=1, cache_read=1) == TokenCost()


def test_a_catalog_is_default_constructible_and_hashable() -> None:
    assert PriceCondition() == PriceCondition(service_tier="auto")
    assert hash(PriceCatalog()) == hash(PriceCatalog())


# ---- the sizes a catalog names ----------------------------------------------


@pytest.mark.parametrize(
    ("tops", "sizes"),
    [
        ((), [_size("fullcontext", 1_000_000, "1m")]),
        (
            (272_000,),
            [
                _size("smallcontext", 272_000, "272k"),
                _size("fullcontext", 1_000_000, "1m"),
            ],
        ),
        (
            (32_000, 256_000),
            [
                _size("smallcontext", 32_000, "32k"),
                _size("midcontext", 256_000, "256k"),
                _size("fullcontext", 1_000_000, "1m"),
            ],
        ),
        ((2_000_000,), [_size("fullcontext", 1_000_000, "1m")]),
    ],
    ids=["flat", "one-band", "two-bands", "band-past-the-window"],
)
def test_small_and_mid_are_the_band_tops_and_large_the_window(
    tops: tuple[int, ...],
    sizes: list[ContextSize],
) -> None:
    assert _banded(*tops).sizes("auto") == tuple(sizes)


@pytest.mark.parametrize(
    ("spec", "size"),
    [
        ("smallcontext", _size("smallcontext", 32_000, "32k")),
        ("midcontext", _size("midcontext", 256_000, "256k")),
        ("fullcontext", _size("fullcontext", 1_000_000, "1m")),
        (32_000, _size("smallcontext", 32_000, "32k")),
        (100_000, _size("midcontext", 100_000, "100k")),
        (258_123, _size("fullcontext", 258_123, "258123")),
        (2_000_000, _size("fullcontext", 1_000_000, "1m")),
    ],
)
def test_a_size_resolves_by_name_or_as_a_count_held_to_the_window(
    spec: ContextSizeName | int,
    size: ContextSize,
) -> None:
    assert _banded(32_000, 256_000)["auto", spec] == size


def test_a_name_no_band_reaches_is_the_window() -> None:
    assert _banded()["auto", "smallcontext"] == _size("fullcontext", 1_000_000, "1m")


def test_a_count_below_one_is_no_size() -> None:
    with pytest.raises(KeyError):
        _ = _banded()["auto", 0]


def test_default_names_and_bills_as_auto() -> None:
    catalog = _banded(272_000)
    assert catalog["default", "smallcontext"] == catalog["auto", "smallcontext"]
    assert catalog["default", 1, date.min] == catalog["auto", 1, date.min]
    assert catalog.service_tiers == {"auto", "default"}


# ---- what a prompt costs ----------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "rate"),
    [(0, 0.0), (272_000, 0.0), (272_001, 1.0), (999_999, 1.0)],
)
def test_a_prompt_pays_its_bands_price(prompt: int, rate: float) -> None:
    assert _banded(272_000)["auto", prompt, date.min].request == rate


def test_a_named_size_is_priced_at_its_tokens() -> None:
    assert _banded(272_000)["auto", "smallcontext", date.min].request == 0.0


def test_the_latest_price_in_effect_applies() -> None:
    catalog = PriceCatalog(
        prices={
            PriceCondition(): TokenPrice(request=2.0),
            PriceCondition(effective_date=date(2026, 10, 7)): TokenPrice(request=1.0),
        },
    )
    assert catalog["auto", 1, date(2026, 10, 6)].request == 2.0
    assert catalog["auto", 1, date(2026, 10, 7)].request == 1.0


def test_a_tier_priced_nowhere_raises() -> None:
    with pytest.raises(KeyError):
        _ = _banded()["priority", 1, date.min]


def test_an_empty_catalog_raises_rather_than_billing_zero() -> None:
    with pytest.raises(KeyError):
        _ = PriceCatalog()["auto", 1, date.min]


def test_a_request_before_any_price_raises() -> None:
    catalog = PriceCatalog(
        prices={PriceCondition(effective_date=date(2027, 1, 1)): TokenPrice()},
    )
    with pytest.raises(KeyError):
        _ = catalog["auto", 1, date(2026, 12, 31)]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
