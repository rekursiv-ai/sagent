"""Tests for what a model offers, where it starts, and how its choices move."""

from __future__ import annotations

from dataclasses import replace

import itertools
import math

import pytest

from sagent.model.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
    literal_members,
)
from sagent.model.context import ContextSize, ContextSizeName
from sagent.model.cost import PriceCatalog, PriceCondition, TokenPrice


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


def _row(prices: PriceCatalog | None = None) -> ModelCapability:
    catalog = prices if prices is not None else _banded(272_000)
    return ModelCapability(
        model_id="m",
        limits=ModelLimits(max_request_tokens=catalog.window),
        prices=catalog,
        thinking_effort=frozenset({"none", "low", "high"}),
        thinking_budget=frozenset({"none", "auto"}),
        thinking_output=frozenset({"none", "text"}),
        service_tier=frozenset({"auto", "default", "priority"}),
        cache_ttl_sec=frozenset({300.0, 3600.0}),
        manage_context_server_side=frozenset({False}),
        default=ModelSettings(
            thinking_effort="low",
            cache_ttl_sec=300.0,
            context_size=catalog["auto", "fullcontext"],
        ),
    )


# ---- default construction ---------------------------------------------------


def test_every_class_is_default_constructible_and_hashable() -> None:
    capability = ModelCapability()
    assert capability.default == ModelSettings()
    assert hash(capability) == hash(ModelCapability())


# ---- what a model offers ----------------------------------------------------


def test_check_passes_what_is_offered() -> None:
    row = _row()
    row.check(row.default)
    row.check(replace(row.default, thinking_effort="high", service_tier="priority"))


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        (
            replace(_row().default, thinking_effort="max"),
            "thinking_effort='max' is not offered by m",
        ),
        (replace(_row().default, thinking_budget="fixed"), "thinking_budget='fixed'"),
        (
            replace(_row().default, thinking_output="redacted"),
            "thinking_output='redacted'",
        ),
        (
            replace(_row().default, service_tier="flex"),
            "allowed: 'auto', 'default', 'priority'",
        ),
        (replace(_row().default, cache_ttl_sec=0.0), "cache_ttl_sec=0.0"),
        (
            replace(_row().default, manage_context_server_side=True),
            "manage_context_server_side=True",
        ),
    ],
)
def test_check_names_what_is_not_offered_and_what_is(
    settings: ModelSettings,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _row().check(settings)


# ---- moving choices onto another model --------------------------------------


def test_fit_keeps_each_choice_offered_and_defaults_the_rest() -> None:
    row = _row()
    other = replace(
        row,
        thinking_effort=frozenset({"none", "high"}),
        cache_ttl_sec=frozenset({3600.0}),
        default=replace(row.default, thinking_effort="none", cache_ttl_sec=3600.0),
    )
    chosen = replace(row.default, thinking_effort="low", thinking_budget="auto")
    fitted = other.fit(chosen, was=row)
    assert (fitted.thinking_effort, fitted.thinking_budget, fitted.cache_ttl_sec) == (
        "none",
        "auto",
        3600.0,
    )


def test_fit_keeps_a_named_size_named_on_the_new_model() -> None:
    wide, narrow = _row(_banded(272_000)), _row(_banded(100_000, window=200_000))
    small = replace(wide.default, context_size=wide.prices["auto", "smallcontext"])
    assert narrow.fit(small, was=wide).context_size == _size(
        "smallcontext",
        100_000,
        "100k",
    )


def test_fit_keeps_a_counted_size_a_count_within_the_window() -> None:
    wide, narrow = _row(_banded(272_000)), _row(_banded(window=200_000))
    counted = replace(wide.default, context_size=wide.prices["auto", 300_000])
    assert narrow.fit(counted, was=wide).context_size == _size(
        "fullcontext",
        200_000,
        "200k",
    )
    counted = replace(wide.default, context_size=wide.prices["auto", 150_000])
    assert narrow.fit(counted, was=wide).context_size == _size(
        "fullcontext",
        150_000,
        "150k",
    )


def test_fit_to_a_tier_sizes_the_context_at_that_tier() -> None:
    prices = PriceCatalog(
        prices={
            PriceCondition(token_size_range=(0, 272_001)): TokenPrice(request=1.0),
            PriceCondition(token_size_range=(272_001, math.inf)): TokenPrice(
                request=2.0,
            ),
            PriceCondition(service_tier="priority"): TokenPrice(request=3.0),
        },
        window=1_000_000,
    )
    row = _row(prices)
    small = replace(row.default, context_size=prices["auto", "smallcontext"])
    fast = row.fit(small, was=row, service_tier="priority")
    assert (fast.service_tier, fast.context_size) == (
        "priority",
        _size("fullcontext", 1_000_000, "1m"),
    )


def test_fit_falls_back_to_the_default_tier_when_one_is_not_offered() -> None:
    row = _row()
    assert row.fit(row.default, was=row, service_tier="flex").service_tier == "auto"


# ---- names ------------------------------------------------------------------


def test_literal_members_reads_the_size_names_in_order() -> None:
    assert literal_members(ContextSizeName) == (
        "smallcontext",
        "midcontext",
        "fullcontext",
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
