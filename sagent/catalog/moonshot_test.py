"""Moonshot catalog rows against the Kimi pricing page (2026-10-03)."""

from __future__ import annotations

import pytest

from sagent.catalog import moonshot
from sagent.types.cost import PriceKey, TokenPrice


@pytest.mark.parametrize(
    ("model_id", "published"),
    [
        (
            "kimi-3.0",
            TokenPrice(
                request=3.0,
                response=15.0,
                cache_write=3.0,
                cache_write_1h=6.0,
                cache_read=0.3,
            ),
        ),
        (
            "kimi-code-2.7",
            TokenPrice(
                request=0.95,
                response=4.0,
                cache_write=0.0,
                cache_write_1h=0.0,
                cache_read=0.19,
            ),
        ),
        (
            "kimi-code-highspeed-2.7",
            TokenPrice(
                request=1.9,
                response=8.0,
                cache_write=0.0,
                cache_write_1h=0.0,
                cache_read=0.38,
            ),
        ),
        (
            "kimi-2.6",
            TokenPrice(
                request=0.95,
                response=4.0,
                cache_write=0.0,
                cache_write_1h=0.0,
                cache_read=0.16,
            ),
        ),
    ],
)
def test_rows_bill_the_published_card(model_id: str, published: TokenPrice) -> None:
    assert moonshot.models()[model_id].prices[PriceKey("auto")] == published


def test_k3_always_thinks_at_a_named_level() -> None:
    k3 = moonshot.models()["kimi-3.0"]
    assert k3.effort_as_level
    assert k3.thinking.effort == {"none", "low", "high", "max"}
    assert k3.thinking.budget == {"none"}


def test_k2_6_thinking_can_be_disabled_and_k2_7_code_cannot() -> None:
    rows = moonshot.models()
    assert rows["kimi-2.6"].thinking.budget == {"none", "auto"}
    assert rows["kimi-code-2.7"].thinking.budget == {"none"}


def test_windows_match_the_published_limits() -> None:
    rows = moonshot.models()
    k3 = rows["kimi-3.0"].context[""]
    assert (k3.max_request_tokens, k3.max_response_tokens) == (1_048_576, 1_048_576)
    assert rows["kimi-2.6"].context[""].max_request_tokens == 262_144
    assert k3.max_request_bytes == 100 << 20


def test_roles_name_live_rows() -> None:
    rows = moonshot.models()
    assert rows["default"].model_id == "kimi-3.0"
    assert rows["utility"].model_id == "kimi-2.6"


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
    rows = moonshot.models()
    assert rows[model_id].wire_model_id == wire_id
    assert rows[wire_id].model_id == model_id


@pytest.mark.parametrize(
    "model_id",
    [
        "kimi-k2.5",
        "kimi-k2-0905-preview",
        "kimi-k2-0711-preview",
        "kimi-k2-turbo-preview",
        "moonshot-v1-8k",
        "moonshot-v1-32k",
        "moonshot-v1-128k",
    ],
)
def test_discontinued_models_are_gone(model_id: str) -> None:
    assert model_id not in moonshot.models()


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
