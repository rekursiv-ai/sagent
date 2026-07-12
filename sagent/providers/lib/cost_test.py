"""Tests for ``providers.lib.cost``: shared cost computation + model profiles."""

from __future__ import annotations

import pytest

from sagent.providers.lib.cost import ModelProfile, compute_cost
from sagent.types.model import Pricing


def test_compute_cost_zero_pricing_returns_zero() -> None:
    in_c, out_c, total = compute_cost(Pricing(), 1_000, 2_000)
    assert in_c == 0.0
    assert out_c == 0.0
    assert total == 0.0


def test_compute_cost_uses_request_and_response_rates() -> None:
    pricing = Pricing(request=3.0, response=15.0)
    in_c, out_c, total = compute_cost(pricing, 1_000_000, 2_000_000)
    assert in_c == pytest.approx(3.0)
    assert out_c == pytest.approx(30.0)
    assert total == pytest.approx(33.0)


def test_compute_cost_includes_cache_write_and_read() -> None:
    pricing = Pricing(request=1.0, response=2.0, cache_write=4.0, cache_read=0.5)
    in_c, out_c, total = compute_cost(
        pricing,
        input_tokens=100_000,
        output_tokens=200_000,
        cache_creation=50_000,
        cache_read=400_000,
    )
    # input = 100k*1 + 50k*4 + 400k*0.5 = 100k + 200k + 200k = 500k / 1M = 0.5
    assert in_c == pytest.approx(0.5)
    assert out_c == pytest.approx(0.4)
    assert total == pytest.approx(0.9)


def test_compute_cost_total_is_sum() -> None:
    pricing = Pricing(request=2.0, response=5.0, cache_read=1.0)
    in_c, out_c, total = compute_cost(pricing, 500_000, 100_000, cache_read=200_000)
    assert total == pytest.approx(in_c + out_c)


def test_compute_cost_fast_applies_only_to_request_and_response() -> None:
    # Fast mode surcharges request/response (Anthropic docs list only
    # Input/Output fast rates); cache write/read stay at standard rates.
    pricing = Pricing(
        request=5.0,
        response=25.0,
        cache_write=6.25,
        cache_read=0.5,
        fast_request=10.0,
        fast_response=50.0,
    )
    in_c, out_c, _ = compute_cost(
        pricing,
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation=1_000_000,
        cache_read=1_000_000,
        fast=True,
    )
    # input = 1M*10 (fast) + 1M*6.25 (std cache_write) + 1M*0.5 (std cache_read)
    assert in_c == pytest.approx(10.0 + 6.25 + 0.5)
    assert out_c == pytest.approx(50.0)


def test_compute_cost_fast_without_fast_rates_falls_back_to_standard() -> None:
    # ``fast=True`` on a ``Pricing`` with no fast rates (default 0.0) must bill
    # at standard rates, not collapse request/response cost to $0.
    pricing = Pricing(request=5.0, response=25.0)
    in_c, out_c, total = compute_cost(pricing, 1_000_000, 1_000_000, fast=True)
    assert (in_c, out_c, total) == (5.0, 25.0, 30.0)


def test_compute_cost_applies_long_context_tier_to_every_input_pool() -> None:
    pricing = Pricing(
        request=5.0,
        response=30.0,
        cache_write=6.25,
        cache_read=0.5,
        long_context_threshold=272_000,
        long_context_input_multiplier=2.0,
        long_context_output_multiplier=1.5,
    )
    in_c, out_c, total = compute_cost(
        pricing,
        input_tokens=100_001,
        output_tokens=1_000,
        cache_creation=100_000,
        cache_read=72_000,
    )
    assert in_c == pytest.approx(
        (100_001 * 10.0 + 100_000 * 12.5 + 72_000 * 1.0) / 1_000_000
    )
    assert out_c == pytest.approx(1_000 * 45.0 / 1_000_000)
    assert total == pytest.approx(in_c + out_c)


def test_compute_cost_keeps_base_tier_at_272k_boundary() -> None:
    pricing = Pricing(
        request=5.0,
        response=30.0,
        long_context_threshold=272_000,
        long_context_input_multiplier=2.0,
        long_context_output_multiplier=1.5,
    )
    in_c, out_c, _ = compute_cost(
        pricing,
        input_tokens=272_000,
        output_tokens=1_000,
    )
    assert in_c == pytest.approx(272_000 * 5.0 / 1_000_000)
    assert out_c == pytest.approx(1_000 * 30.0 / 1_000_000)


def test_model_profile_defaults() -> None:
    p = ModelProfile(max_request_tokens=1000, max_response_tokens=500)
    assert p.max_request_tokens == 1000
    assert p.max_response_tokens == 500
    assert p.supports_thinking is True
    assert p.chars_per_token == 4.0
    assert p.pricing == Pricing()


def test_model_profile_custom_pricing_chars() -> None:
    pricing = Pricing(request=1.5)
    p = ModelProfile(
        max_request_tokens=2000,
        max_response_tokens=1000,
        pricing=pricing,
        chars_per_token=2.83,
        supports_thinking=False,
    )
    assert p.pricing is pricing
    assert p.chars_per_token == 2.83
    assert p.supports_thinking is False


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
