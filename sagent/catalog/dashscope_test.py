"""DashScope catalog rows against Model Studio's published tables (2026-10-03)."""

from __future__ import annotations

from datetime import date

import pytest

from sagent.catalog import dashscope
from sagent.types.cost import PriceKey


_TODAY = date(2026, 10, 3)


@pytest.mark.parametrize(
    ("model_id", "prompt_tokens", "published"),
    [
        ("qwen-max-3.8", 1_000_000, (2.0, 6.0)),
        ("qwen-flash-3.8", 1_000, (0.15, 0.47)),
        ("qwen-max-3.7", 1_000, (2.5, 7.5)),
        ("qwen-plus-3.7", 256_000, (0.4, 1.6)),
        ("qwen-plus-3.7", 256_001, (1.2, 4.8)),
        ("qwen-flash-3.7", 32_000, (0.03, 0.13)),
        ("qwen-flash-3.7", 32_001, (0.10, 0.40)),
        ("qwen-flash-3.7", 256_001, (0.20, 0.80)),
        ("qwen-plus-3.6", 256_001, (2.0, 6.0)),
        ("qwen-flash-3.6", 1_000, (0.25, 1.5)),
        ("qwen-max-3.6", 128_001, (2.0, 12.0)),
        ("qwen-coder-480b-a35b-3.0", 128_001, (4.5, 22.5)),
        ("qwen-235b-a22b-thinking-3.0", 1_000, (0.23, 2.3)),
        ("qwen-32b-3.0", 1_000, (0.16, 0.64)),
        ("qwen-plus-3.0", 256_001, (1.2, 3.6)),
        ("qwen-turbo-3.0", 1_000, (0.05, 0.2)),
    ],
)
def test_a_prompt_bills_its_whole_band(
    model_id: str,
    prompt_tokens: int,
    published: tuple[float, float],
) -> None:
    """Every token in a request bills at its tier's rate (model-pricing)."""
    price = dashscope.models()[model_id].prices.rate(
        service_tier="auto",
        prompt_tokens=prompt_tokens,
        at=_TODAY,
    )
    assert (price.request, price.response) == published


@pytest.mark.parametrize(
    ("model_id", "cache_read"),
    [
        ("qwen-plus-3.7", 0.08),
        ("qwen-turbo-3.0", 0.01),
        # Unpublished (console only): billed at the input rate.
        ("qwen-max-3.8", 2.0),
        # Explicit-cache only in Singapore: no implicit discount.
        ("qwen-plus-3.6", 0.5),
    ],
)
def test_cache_hits_bill_the_published_rate(model_id: str, cache_read: float) -> None:
    price = dashscope.models()[model_id].prices[PriceKey("auto")]
    assert price.cache_read == pytest.approx(cache_read)


def test_roles_name_the_vendor_recommendations() -> None:
    rows = dashscope.models()
    assert rows["default"].model_id == "qwen-plus-3.7"
    assert rows["utility"].model_id == "qwen-flash-3.8"


def test_qwen_3_8_takes_its_effort_as_a_level() -> None:
    rows = dashscope.models()
    assert rows["qwen-max-3.8"].effort_as_level
    assert rows["qwen-max-3.8"].thinking.effort == {"none", "low", "medium", "xhigh"}
    assert not rows["qwen-plus-3.7"].effort_as_level


@pytest.mark.parametrize(
    ("model_id", "budget"),
    [
        ("qwen-turbo-3.0", {"none", "auto", "fixed"}),
        ("qwen-2400b-a95b-3.8", {"auto"}),
        ("qwen-235b-a22b-thinking-3.0", {"auto", "fixed"}),
        ("qwen-235b-a22b-instruct-3.0", {"none"}),
    ],
)
def test_thinking_switches_match_the_published_modes(
    model_id: str,
    budget: set[str],
) -> None:
    """``qwen-turbo`` is hybrid; ``-thinking`` ids reject the off switch."""
    assert dashscope.models()[model_id].thinking.budget == budget


@pytest.mark.parametrize(
    ("model_id", "wire_id"),
    [
        ("qwen-max-3.8", "qwen3.8-max"),
        ("qwen-2400b-a95b-3.8", "qwen3.8-2.4t-a95b"),
        ("qwen-max-3.6", "qwen3.6-max-preview"),
        ("qwen-235b-a22b-instruct-3.0", "qwen3-235b-a22b-instruct-2507"),
        ("qwen-plus-3.0", "qwen-plus"),
    ],
)
def test_vendor_ids_resolve_to_their_rows(model_id: str, wire_id: str) -> None:
    rows = dashscope.models()
    assert rows[model_id].wire_model_id == wire_id
    assert rows[wire_id].model_id == model_id


def test_a_bare_family_names_its_newest_row() -> None:
    assert dashscope.models()["qwen-max"].model_id == "qwen-max-3.8"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
