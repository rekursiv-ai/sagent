"""MiniMax catalog rows against the pay-as-you-go table (2026-10-03)."""

from __future__ import annotations

from datetime import date

import pytest

from sagent.catalog import minimax
from sagent.types.cost import PriceKey, ServiceTier


_TODAY = date(2026, 10, 3)


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
    price = minimax.models()[model_id].prices[PriceKey("auto")]
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
    price = minimax.models()["minimax-3.0"].prices.rate(
        service_tier=tier,
        prompt_tokens=prompt_tokens,
        at=_TODAY,
    )
    assert (price.request, price.response, price.cache_read) == pytest.approx(
        published,
    )


def test_only_m3_offers_priority() -> None:
    rows = minimax.models()
    assert {m for m, row in rows.items() if "priority" in row.service_tier} == {
        "minimax-3.0",
    }


def test_api_adds_priority_to_the_compatible_transport() -> None:
    assert minimax.api().service_tier == {"auto", "priority"}


def test_m2_thinking_cannot_be_disabled_and_m3_can() -> None:
    rows = minimax.models()
    assert rows["minimax-2.7"].thinking.budget == {"none"}
    assert rows["minimax-3.0"].thinking.budget == {"none", "auto"}


def test_windows_match_the_published_limits() -> None:
    rows = minimax.models()
    m3 = rows["minimax-3.0"].context[""]
    assert (m3.max_request_tokens, m3.max_response_tokens) == (1_000_000, 524_288)
    assert (m3.max_request_bytes, m3.max_image_bytes) == (64 << 20, 10 << 20)
    m2 = rows["minimax-2.7"].context[""]
    assert (m2.max_request_tokens, m2.max_response_tokens) == (204_800, 204_800)


def test_roles_name_live_rows() -> None:
    rows = minimax.models()
    assert rows["default"].model_id == "minimax-3.0"
    assert rows["utility"].model_id == "minimax-3.0"


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
    rows = minimax.models()
    assert rows[model_id].wire_model_id == wire_id
    assert rows[wire_id].model_id == model_id


@pytest.mark.parametrize(
    "model_id",
    ["MiniMax-M1", "MiniMax-Text-01", "abab6.5s-chat", "abab6.5-chat"],
)
def test_discontinued_models_are_gone(model_id: str) -> None:
    assert model_id not in minimax.models()


def test_scaling_a_card_scales_every_meter() -> None:
    """A priority multiple of a card with a write column bills writes at it too."""
    card = minimax._m2_card(request=1.0, response=2.0, cache_read=0.1)
    scaled = minimax._scaled(card, 1.5)
    assert (scaled.cache_write, scaled.cache_write_1h) == (
        card.cache_write * 1.5,
        card.cache_write_1h * 1.5,
    )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
