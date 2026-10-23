"""Gemini thinking-wire mappings, transport restrictions, and published rows."""

from dataclasses import replace
from datetime import date

import pytest

from sagent.catalog import google
from sagent.catalog.loader import models
from sagent.types.capability import (
    ModelCapability,
    ModelSettings,
    ThinkingBudget,
    ThinkingCapability,
    ThinkingEffort,
    ThinkingOutput,
)


def _wire(
    *,
    level: bool = True,
    effort: ThinkingEffort = "none",
    budget: ThinkingBudget = "fixed",
    output: ThinkingOutput = "text",
) -> object:
    row = ModelCapability(
        effort_as_level=level,
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
    )
    settings = ModelSettings(
        capability=row,
        thinking_effort=effort,
        thinking_budget=budget,
        thinking_output=output,
    )
    return google.thinking_config(row, settings)


def test_token_budget_models_send_the_cap_or_auto_sentinel() -> None:
    assert _wire(level=False, effort="max") == {
        "includeThoughts": True,
        "thinkingBudget": 24_576,
    }
    assert _wire(level=False, budget="auto") == {
        "includeThoughts": True,
        "thinkingBudget": -1,
    }


@pytest.mark.parametrize(
    ("effort", "level"),
    [("min", "minimal"), ("low", "low"), ("medium", "medium"), ("high", "high")],
)
def test_level_models_send_the_effort_as_a_level(
    effort: ThinkingEffort,
    level: str,
) -> None:
    assert _wire(effort=effort) == {
        "includeThoughts": True,
        "thinkingLevel": level,
    }


@pytest.mark.parametrize("budget", ["auto", "fixed"])
def test_level_models_never_send_a_token_budget(budget: ThinkingBudget) -> None:
    assert _wire(effort="high", budget=budget) == {
        "includeThoughts": True,
        "thinkingLevel": "high",
    }


@pytest.mark.parametrize("level", [False, True])
def test_effort_none_leaves_the_models_default_in_place(level: bool) -> None:
    assert _wire(level=level) == {"includeThoughts": True}


def test_budget_none_sends_no_thinking_config() -> None:
    assert _wire(budget="none") is None


def test_output_none_disables_thought_parts() -> None:
    assert _wire(output="none") == {"includeThoughts": False}


@pytest.mark.parametrize(
    ("effort", "budget"),
    [
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
    assert google.thinking_budget(effort) == budget


def test_effort_none_has_no_thinking_budget() -> None:
    with pytest.raises(
        ValueError,
        match=r"^effort 'none' sends no budget; omit thinkingConfig$",
    ):
        google.thinking_budget("none")


@pytest.mark.parametrize("effort", ["none", "xhigh", "max"])
def test_level_models_reject_none_or_past_high(effort: ThinkingEffort) -> None:
    with pytest.raises(
        ValueError,
        match=rf"^effort '{effort}' has no Gemini thinkingLevel$",
    ):
        google._thinking_level(effort)


@pytest.mark.parametrize(
    "transport",
    [google.api(), google.cli(), google.subscription()],
    ids=["api", "cli", "subscription"],
)
def test_no_transport_offers_redacted_thinking(transport: ModelCapability) -> None:
    assert "redacted" not in transport.thinking.output


def test_api_preserves_all_supported_thinking_axes() -> None:
    assert google.api() == ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
    )


def test_cli_removes_effort_and_uses_account_managed_context() -> None:
    assert google.cli() == ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
        manage_context_server_side=frozenset({True}),
        account_auth=True,
    )


def test_subscription_adds_account_auth() -> None:
    assert google.subscription() == replace(google.api(), account_auth=True)


# ---- published facts in ``assets/google.yaml`` ------------------------------


@pytest.mark.parametrize(
    ("model_id", "wire_id", "cutoff"),
    [
        ("gemini-pro-3.1", "gemini-3.1-pro-preview", "January 2025"),
        ("gemini-pro-2.5", "gemini-2.5-pro", "January 2025"),
        ("gemini-flash-3.8", "gemini-3.8-flash", None),
        ("gemini-flash-3.7", "gemini-3.7-flash", None),
        ("gemini-flash-3.6", "gemini-3.6-flash", None),
        ("gemini-flash-3.5", "gemini-3.5-flash", None),
        ("gemini-flash-3.0", "gemini-3-flash-preview", "January 2025"),
        ("gemini-flash-2.5", "gemini-2.5-flash", "January 2025"),
        ("gemini-flash-lite-3.5", "gemini-3.5-flash-lite", None),
        ("gemini-flash-lite-3.1", "gemini-3.1-flash-lite", "January 2025"),
        ("gemini-flash-lite-2.5", "gemini-2.5-flash-lite", "January 2025"),
    ],
)
def test_rows_name_their_wire_id_and_published_cutoff(
    model_id: str,
    wire_id: str,
    cutoff: str | None,
) -> None:
    table = models("google")
    assert table[model_id].wire_model_id == wire_id
    assert table[model_id].knowledge_cutoff == cutoff
    assert table[wire_id] is table[model_id]


def test_roles_follow_googles_guidance_for_new_projects() -> None:
    table = models("google")
    assert {role: table[role].model_id for role in table.roles} == {
        "best": "gemini-pro-3.1",
        "default": "gemini-pro-3.1",
        "everyday": "gemini-flash-3.8",
        "utility": "gemini-flash-lite-3.5",
    }


@pytest.mark.parametrize(
    ("model_id", "standard"),
    [
        ("gemini-flash-3.8", (0.75, 3.75, 0.075)),
        ("gemini-flash-3.5", (1.5, 9.0, 0.15)),
        ("gemini-flash-lite-3.5", (0.3, 2.5, 0.03)),
        ("gemini-flash-lite-3.1", (0.25, 1.5, 0.025)),
        ("gemini-flash-3.0", (0.5, 3.0, 0.05)),
        ("gemini-flash-2.5", (0.3, 2.5, 0.03)),
        ("gemini-flash-lite-2.5", (0.1, 0.4, 0.01)),
    ],
)
def test_a_row_bills_the_published_2026_rate(
    model_id: str,
    standard: tuple[float, float, float],
) -> None:
    price = models("google")[model_id].prices.rate(
        service_tier="auto",
        prompt_tokens=1_000,
        at=date(2026, 12, 31),
    )
    assert (price.request, price.response, price.cache_read) == standard


@pytest.mark.parametrize(
    "model_id",
    ["gemini-flash-3.8", "gemini-flash-3.7", "gemini-flash-3.6"],
)
def test_the_introductory_flash_rate_doubles_on_2027_01_01(model_id: str) -> None:
    prices = models("google")[model_id].prices
    last = prices.rate(service_tier="auto", prompt_tokens=1, at=date(2026, 12, 31))
    first = prices.rate(service_tier="auto", prompt_tokens=1, at=date(2027, 1, 1))
    assert (last.request, last.response, last.cache_read) == (0.75, 3.75, 0.075)
    assert (first.request, first.response, first.cache_read) == (1.5, 7.5, 0.15)


@pytest.mark.parametrize(
    ("model_id", "short", "long"),
    [
        ("gemini-pro-3.1", (2.0, 12.0, 0.2), (4.0, 18.0, 0.4)),
        ("gemini-pro-2.5", (1.25, 10.0, 0.125), (2.5, 15.0, 0.25)),
    ],
)
def test_prompts_over_200k_bill_the_published_rate(
    model_id: str,
    short: tuple[float, float, float],
    long: tuple[float, float, float],
) -> None:
    """Gemini bands "prompts > 200k"; exactly 200k is still short."""
    prices = models("google")[model_id].prices
    at = date(2026, 10, 22)
    at_200k = prices.rate(service_tier="auto", prompt_tokens=200_000, at=at)
    over = prices.rate(service_tier="auto", prompt_tokens=200_001, at=at)
    assert (at_200k.request, at_200k.response, at_200k.cache_read) == short
    assert (over.request, over.response, over.cache_read) == long


def test_every_row_has_the_published_windows() -> None:
    """No per-image cap: images tile server-side, so only the 100 MB body binds."""
    for row in models("google").values():
        assert row.context.keys() == {""}
        limits = row.context[""]
        assert limits.max_request_tokens == 1_048_576
        assert limits.max_response_tokens == 65_536
        assert limits.max_request_bytes == 100 * 1024 * 1024
        assert (limits.max_image_edge_px, limits.max_image_bytes) == (0, 0)


@pytest.mark.parametrize(
    ("model_id", "efforts"),
    [
        # "minimal is not supported and returns an error" on these.
        ("gemini-pro-3.1", {"none", "low", "medium", "high"}),
        ("gemini-flash-3.8", {"none", "low", "medium", "high"}),
        ("gemini-flash-3.7", {"none", "low", "medium", "high"}),
        ("gemini-flash-3.6", {"none", "min", "low", "medium", "high"}),
        ("gemini-flash-3.5", {"none", "min", "low", "medium", "high"}),
        ("gemini-flash-lite-3.5", {"none", "min", "low", "medium", "high"}),
        ("gemini-flash-lite-3.1", {"none", "min", "low", "medium", "high"}),
        ("gemini-flash-3.0", {"none", "min", "low", "medium", "high"}),
    ],
)
def test_a_gemini_3_row_offers_only_the_levels_it_accepts(
    model_id: str,
    efforts: set[ThinkingEffort],
) -> None:
    row = models("google")[model_id]
    assert row.thinking.effort == efforts
    assert row.effort_as_level


@pytest.mark.parametrize(
    "model_id",
    ["gemini-pro-2.5", "gemini-flash-2.5", "gemini-flash-lite-2.5"],
)
def test_gemini_2_5_takes_a_token_budget_not_a_level(model_id: str) -> None:
    row = models("google")[model_id]
    assert not row.effort_as_level
    assert row.thinking.effort == {
        "none",
        "min",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
