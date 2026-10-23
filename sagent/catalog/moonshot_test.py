"""Moonshot rows against the Kimi pricing and model pages (2026-10-03)."""

import pytest

from sagent.catalog.loader import models
from sagent.types.cost import PriceKey


@pytest.mark.parametrize(
    ("model_id", "published"),
    [
        ("kimi-3.0", (3.0, 15.0, 3.0, 6.0, 0.3)),
        ("kimi-2.6", (0.95, 4.0, 0.0, 0.0, 0.16)),
        ("kimi-code-2.7", (0.95, 4.0, 0.0, 0.0, 0.19)),
        ("kimi-code-highspeed-2.7", (1.9, 8.0, 0.0, 0.0, 0.38)),
    ],
)
def test_rows_bill_the_published_card(
    model_id: str,
    published: tuple[float, float, float, float, float],
) -> None:
    card = models("moonshot")[model_id].prices[PriceKey("auto")]
    assert (
        card.request,
        card.response,
        card.cache_write,
        card.cache_write_1h,
        card.cache_read,
    ) == published


def test_k3_thinks_at_a_named_level() -> None:
    k3 = models("moonshot")["kimi-3.0"]
    assert k3.effort_as_level
    assert k3.thinking.effort == {"none", "low", "high", "max"}
    assert k3.thinking.budget == {"none"}


def test_k2_6_thinking_can_be_disabled_and_k2_7_code_cannot() -> None:
    table = models("moonshot")
    assert table["kimi-2.6"].thinking.budget == {"none", "auto"}
    assert table["kimi-code-2.7"].thinking.budget == {"none"}


def test_windows_match_the_published_limits() -> None:
    table = models("moonshot")
    k3 = table["kimi-3.0"].context[""]
    assert (k3.max_request_tokens, k3.max_response_tokens) == (1_048_576, 1_048_576)
    assert k3.max_request_bytes == 100 << 20
    assert table["kimi-2.6"].context[""].max_request_tokens == 262_144


@pytest.mark.parametrize(
    ("model_id", "wire_id"),
    [
        ("kimi-3.0", "kimi-k3"),
        ("kimi-2.6", "kimi-k2.6"),
        ("kimi-code-2.7", "kimi-k2.7-code"),
        ("kimi-code-highspeed-2.7", "kimi-k2.7-code-highspeed"),
    ],
)
def test_vendor_ids_resolve_to_their_rows(model_id: str, wire_id: str) -> None:
    table = models("moonshot")
    assert table[model_id].wire_model_id == wire_id
    assert table[wire_id] is table[model_id]


def test_roles_name_live_rows() -> None:
    table = models("moonshot")
    assert (table["default"].model_id, table["utility"].model_id) == (
        "kimi-3.0",
        "kimi-2.6",
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
