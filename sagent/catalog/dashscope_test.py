"""DashScope thinking-budget wire values and Model Studio's published rows."""

from datetime import date

import pytest

from sagent.catalog import dashscope
from sagent.catalog.loader import models
from sagent.types.capability import ThinkingBudget, ThinkingEffort
from sagent.types.cost import PriceKey


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
    """Every token in a request bills at its band's rate (model-pricing)."""
    price = models("dashscope")[model_id].prices.rate(
        service_tier="auto",
        prompt_tokens=prompt_tokens,
        at=date(2026, 10, 22),
    )
    assert (price.request, price.response) == published


@pytest.mark.parametrize(
    ("model_id", "cache_read"),
    [
        ("qwen-plus-3.7", 0.08),
        ("qwen-flash-3.7", 0.006),
        ("qwen-turbo-3.0", 0.01),
        # Unpublished (console only): billed at the input rate.
        ("qwen-max-3.8", 2.0),
        # Explicit-cache only in Singapore: no implicit discount.
        ("qwen-plus-3.6", 0.5),
    ],
)
def test_cache_hits_bill_the_published_rate(model_id: str, cache_read: float) -> None:
    assert models("dashscope")[model_id].prices[PriceKey("auto")].cache_read == (
        cache_read
    )


def test_roles_name_the_vendor_recommendations() -> None:
    table = models("dashscope")
    assert table["default"].model_id == "qwen-plus-3.7"
    assert table["utility"].model_id == "qwen-flash-3.8"
    assert table["qwen-max"].model_id == "qwen-max-3.8"


def test_qwen_3_8_takes_its_effort_as_a_level() -> None:
    table = models("dashscope")
    assert table["qwen-max-3.8"].effort_as_level
    assert table["qwen-max-3.8"].thinking.effort == {"none", "low", "medium", "xhigh"}
    assert not table["qwen-plus-3.7"].effort_as_level


@pytest.mark.parametrize(
    ("model_id", "budget"),
    [
        ("qwen-turbo-3.0", {"none", "auto", "fixed"}),
        ("qwen-2400b-a95b-3.8", {"auto"}),
        ("qwen-235b-a22b-thinking-3.0", {"auto", "fixed"}),
        ("qwen-235b-a22b-instruct-3.0", {"none"}),
        ("qwen-coder-480b-a35b-3.0", {"none"}),
    ],
)
def test_thinking_switches_match_the_published_modes(
    model_id: str,
    budget: set[ThinkingBudget],
) -> None:
    """``qwen-turbo`` is hybrid; ``-thinking`` ids reject the off switch."""
    assert models("dashscope")[model_id].thinking.budget == budget


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
    table = models("dashscope")
    assert table[model_id].wire_model_id == wire_id
    assert table[wire_id] is table[model_id]


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        ("none", "0"),
        ("min", "1024"),
        ("low", "4096"),
        ("medium", "8192"),
        ("high", "16384"),
        ("xhigh", "20480"),
        ("max", "24576"),
    ],
)
def test_each_effort_sends_its_thinking_budget(
    effort: ThinkingEffort,
    budget: str,
) -> None:
    assert dashscope.thinking_budget(effort) == budget


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
