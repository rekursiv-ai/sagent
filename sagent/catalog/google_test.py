"""Gemini thinking-wire mappings and transport restrictions."""

from dataclasses import replace

import pytest

from sagent.catalog import google
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


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
