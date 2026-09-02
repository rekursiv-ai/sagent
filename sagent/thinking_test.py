"""Tests for ``thinking``: one word names one axis, the rest are inherited."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from sagent.thinking import (
    THINKING_COMMANDS,
    apply_thinking_command,
    describe_thinking,
    thinking_offered,
)
from sagent.types.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
    ThinkingBudget,
    ThinkingOutput,
)


def _capability(
    *,
    budgets: frozenset[ThinkingBudget] = frozenset({"none", "auto", "fixed"}),
    outputs: frozenset[ThinkingOutput] = frozenset({"none", "text", "redacted"}),
) -> ModelCapability:
    return ModelCapability(
        model_id="m",
        context=MappingProxyType({"": ModelLimits(max_request_tokens=1_000)}),
        thinking_budget=budgets,
        thinking_output=outputs,
    )


def _thinking() -> ModelSettings:
    """A model that thinks, with the reasoning currently on."""
    return ModelSettings(
        capability=_capability(), thinking_budget="auto", thinking_output="text"
    )


def test_a_budget_word_leaves_the_display_alone() -> None:
    """The point of the split: "think harder" must not also un-hide."""
    assert apply_thinking_command("on", _thinking(), show=False) is False
    assert apply_thinking_command("on", _thinking(), show=True) is True


def test_a_display_word_leaves_the_budget_alone() -> None:
    """And "stop showing me" must not stop the thinking."""
    settings = _thinking()
    assert apply_thinking_command("hide", settings, show=True) is False
    assert settings.thinking_budget == "auto"
    assert settings.thinking_output == "text"


@pytest.mark.parametrize(
    ("word", "budget"), [("adaptive", "auto"), ("on", "fixed"), ("off", "none")]
)
def test_each_budget_word_selects_its_rung(word: str, budget: ThinkingBudget) -> None:
    settings = _thinking()
    _ = apply_thinking_command(word, settings, show=True)
    assert settings.thinking_budget == budget


def test_off_also_pins_the_display() -> None:
    """There is no reasoning to render, so ``show`` cannot stay true."""
    settings = _thinking()
    assert apply_thinking_command("off", settings, show=True) is False
    assert settings.thinking_budget == "none"
    assert settings.thinking_output == "none"


def test_redact_withholds_the_body_without_stopping_the_thinking() -> None:
    """Server-side redaction still spends budget; only the text is withheld."""
    settings = _thinking()
    assert apply_thinking_command("redact", settings, show=True) is False
    assert settings.thinking_output == "redacted"
    assert settings.thinking_budget == "auto"


def test_redact_from_off_turns_thinking_back_on() -> None:
    """Redacting nothing is not a state; the word implies reasoning happens."""
    settings = ModelSettings(capability=_capability())
    _ = apply_thinking_command("redact", settings, show=False)
    assert settings.thinking_budget == "auto"


def test_an_unknown_word_lists_the_valid_ones() -> None:
    with pytest.raises(ValueError, match="thinking must be one of"):
        _ = apply_thinking_command("louder", _thinking(), show=True)


def test_every_advertised_word_applies() -> None:
    """Without this a word could be offered by the REPL and then raise."""
    for word in THINKING_COMMANDS:
        _ = apply_thinking_command(word, _thinking(), show=True)


def test_offered_checks_both_axes_a_word_touches() -> None:
    """A word is two selections, so one axis passing proves nothing.

    The gate this replaced compared a fused state against ``thinking_budget``
    alone -- disjoint vocabularies, so it accepted every state on every model.
    """
    no_redaction = ModelSettings(
        capability=_capability(outputs=frozenset({"none", "text"})),
        thinking_budget="auto",
        thinking_output="text",
    )
    assert not thinking_offered("redact", no_redaction)
    assert thinking_offered("redact", _thinking())


def test_offered_rejects_a_budget_the_model_withholds() -> None:
    settings = ModelSettings(
        capability=_capability(budgets=frozenset({"none", "auto"})),
        thinking_budget="auto",
        thinking_output="text",
    )
    assert not thinking_offered("on", settings)


def test_an_unofferable_word_leaves_the_settings_untouched() -> None:
    """Half-applying is a state neither the caller nor the wire agreed to."""
    settings = ModelSettings(
        capability=_capability(outputs=frozenset({"none", "text"})),
        thinking_budget="auto",
        thinking_output="text",
    )
    with pytest.raises(ValueError, match="not offered by m"):
        _ = apply_thinking_command("redact", settings, show=True)
    assert settings.thinking_output == "text"
    assert settings.thinking_budget == "auto"


def test_off_is_offerable_even_on_a_model_that_cannot_think() -> None:
    """``none`` is on every ladder, so turning it off never fails."""
    settings = ModelSettings(
        capability=_capability(budgets=frozenset({"none"}), outputs=frozenset({"none"}))
    )
    assert thinking_offered("off", settings)
    assert not thinking_offered("adaptive", settings)


@pytest.mark.parametrize(
    ("word", "show", "described"),
    [
        ("off", True, "off"),
        ("adaptive", True, "adaptive show"),
        ("adaptive", False, "adaptive hide"),
        ("on", True, "on show"),
        ("redact", True, "redact"),
    ],
)
def test_describe_renders_the_words_that_reproduce_the_selection(
    word: str, show: bool, described: str
) -> None:
    settings = _thinking()
    shown = apply_thinking_command(word, settings, show=show)
    assert describe_thinking(settings, show=shown) == described


@pytest.mark.parametrize("word", THINKING_COMMANDS)
def test_describe_survives_every_advertised_word(word: str) -> None:
    """``describe`` maps the budget ladder by hand, so a new rung KeyErrors.

    ``apply`` is total over the words by construction (its ``match`` ends in
    a raise); ``describe``'s dict does not, and only the round trip crosses
    both. Without this, adding a ``ThinkingBudget`` rung passes every test
    and then crashes the ``/thinking`` handler that prints the result.
    """
    settings = _thinking()
    for show in (True, False):
        shown = apply_thinking_command(word, settings, show=show)
        described = describe_thinking(settings, show=shown)
        assert described.split()[0] in THINKING_COMMANDS


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
