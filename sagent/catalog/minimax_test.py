"""MiniMax transport restrictions and the pay-as-you-go rows (2026-10-03)."""

from dataclasses import replace
from datetime import date

import pytest

from sagent.catalog import minimax
from sagent.catalog.loader import models
from sagent.catalog.openai import compatible
from sagent.types.cost import PriceKey, ServiceTier


def test_api_adds_priority_to_the_compatible_transport() -> None:
    transport = compatible()
    assert minimax.api() == replace(
        transport,
        service_tier=frozenset({"auto", "priority"}),
    )


@pytest.mark.parametrize(
    ("model_id", "published"),
    [
        ("minimax-2.7", (0.3, 1.2, 0.06, 0.375)),
        ("minimax-highspeed-2.7", (0.6, 2.4, 0.06, 0.375)),
        ("minimax-2.5", (0.3, 1.2, 0.03, 0.375)),
        ("minimax-highspeed-2.5", (0.6, 2.4, 0.03, 0.375)),
        ("minimax-2.1", (0.3, 1.2, 0.03, 0.375)),
        ("minimax-highspeed-2.1", (0.6, 2.4, 0.03, 0.375)),
        ("minimax-2.0", (0.3, 1.2, 0.03, 0.375)),
    ],
)
def test_m2_rows_bill_the_published_card(
    model_id: str,
    published: tuple[float, float, float, float],
) -> None:
    price = models("minimax")[model_id].prices[PriceKey("auto")]
    assert (price.request, price.response, price.cache_read, price.cache_write) == (
        published
    )


@pytest.mark.parametrize(
    ("tier", "prompt_tokens", "published"),
    [
        ("auto", 512_000, (0.3, 1.2, 0.06)),
        ("auto", 512_001, (0.6, 2.4, 0.12)),
        ("priority", 512_000, (0.45, 1.8, 0.09)),
        ("priority", 512_001, (0.9, 3.6, 0.18)),
    ],
)
def test_m3_bills_its_band_and_tier(
    tier: ServiceTier,
    prompt_tokens: int,
    published: tuple[float, float, float],
) -> None:
    price = models("minimax")["minimax-3.0"].prices.rate(
        service_tier=tier,
        prompt_tokens=prompt_tokens,
        at=date(2026, 10, 22),
    )
    assert (price.request, price.response, price.cache_read) == published


def test_only_m3_offers_priority() -> None:
    table = models("minimax")
    assert {name for name, row in table.items() if "priority" in row.service_tier} == {
        "minimax-3.0",
    }


def test_m2_thinking_cannot_be_disabled_and_m3_can() -> None:
    table = models("minimax")
    assert table["minimax-2.7"].thinking.budget == {"none"}
    assert table["minimax-3.0"].thinking.budget == {"none", "auto"}


def test_windows_match_the_published_limits() -> None:
    table = models("minimax")
    m3 = table["minimax-3.0"].context[""]
    assert (m3.max_request_tokens, m3.max_response_tokens) == (1_000_000, 524_288)
    assert (m3.max_request_bytes, m3.max_image_bytes) == (64 << 20, 10 << 20)
    m2 = table["minimax-2.7"].context[""]
    assert (m2.max_request_tokens, m2.max_response_tokens) == (204_800, 204_800)


@pytest.mark.parametrize(
    ("model_id", "wire_id"),
    [
        ("minimax-3.0", "MiniMax-M3"),
        ("minimax-2.7", "MiniMax-M2.7"),
        ("minimax-2.0", "MiniMax-M2"),
        ("minimax-highspeed-2.7", "MiniMax-M2.7-highspeed"),
    ],
)
def test_vendor_ids_resolve_to_their_rows(model_id: str, wire_id: str) -> None:
    table = models("minimax")
    assert table[model_id].wire_model_id == wire_id
    assert table[wire_id] is table[model_id]


def test_roles_name_live_rows() -> None:
    table = models("minimax")
    assert {table[role].model_id for role in table.roles} == {"minimax-3.0"}


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
