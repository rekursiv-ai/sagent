"""Tests for ``types.model``: validation and arithmetic on data classes."""

from __future__ import annotations

from typing import cast

import pytest

from sagent.catalog import anthropic as anthropic_catalog
from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelLimits,
)
from sagent.types.cost import (
    TokenCount,
)
from sagent.types.model import (
    CONTEXT_TAGS,
    AgentSettings,
    ModelRecipe,
    ModelResponse,
    StreamInterruptedError,
    base_model_id,
    default_buffer_tokens,
    split_model_id,
)
from sagent.types.runtime import AssistantMessage


# ---- AgentSettings ---------------------------------------------------------


def _limits(*, request: int = 4_096, response: int = 1_024) -> ModelLimits:
    return ModelLimits(max_request_tokens=request, max_response_tokens=response)


def test_from_limits_handles_a_small_window() -> None:
    """B12: a floored buffer used to exceed a sub-8k window and raise."""
    _ = AgentSettings.from_limits(_limits(request=4_096))


def test_from_limits_caps_the_buffer_at_half_the_window() -> None:
    """The buffer never exceeds half the window even for tiny models."""
    settings = AgentSettings.from_limits(_limits(request=4_096))
    assert settings.buffer_tokens <= 4_096 // 2


def test_from_limits_carries_both_windows() -> None:
    settings = AgentSettings.from_limits(_limits(request=200_000, response=64_000))
    assert settings.max_request_tokens == 200_000
    assert settings.max_response_tokens == 64_000


def test_from_limits_takes_limits_not_a_model() -> None:
    """``swap_model`` sizes a candidate window before that model exists."""
    limits = _limits(request=1_000_000, response=128_000)
    assert AgentSettings.from_limits(limits).buffer_tokens == default_buffer_tokens(
        1_000_000
    )


def test_accepts_zero_buffer_tokens() -> None:
    """``buffer_tokens = 0`` is a legal "no headroom" setting."""
    settings = AgentSettings(
        max_request_tokens=1_000,
        max_response_tokens=100,
        buffer_tokens=0,
    )
    assert settings.buffer_tokens == 0


def test_rejects_negative_buffer_tokens() -> None:
    """Negative ``buffer_tokens`` would inflate the usable window past the cap."""
    with pytest.raises(ValueError, match="buffer_tokens"):
        _ = AgentSettings(
            max_request_tokens=1_000,
            max_response_tokens=100,
            buffer_tokens=-1,
        )


def test_rejects_a_buffer_that_swallows_the_window() -> None:
    with pytest.raises(ValueError, match="buffer_tokens"):
        _ = AgentSettings(
            max_request_tokens=1_000,
            max_response_tokens=100,
            buffer_tokens=1_000,
        )


def test_rejects_zero_max_attempts() -> None:
    """Zero sends nothing: the count is checked before the first send."""
    with pytest.raises(ValueError, match="max_attempts"):
        _ = AgentSettings(
            max_request_tokens=1_000,
            max_response_tokens=100,
            max_attempts=0,
        )


def test_rejects_negative_max_tool_call_rounds() -> None:
    with pytest.raises(ValueError, match="max_tool_call_rounds"):
        _ = AgentSettings(
            max_request_tokens=1_000,
            max_response_tokens=100,
            max_tool_call_rounds=-1,
        )


def test_accepts_none_max_tool_call_rounds() -> None:
    settings = AgentSettings(
        max_request_tokens=1_000,
        max_response_tokens=100,
        max_tool_call_rounds=None,
    )
    assert settings.max_tool_call_rounds is None


def test_rejects_negative_max_budget_usd() -> None:
    with pytest.raises(ValueError, match="max_budget_usd"):
        _ = AgentSettings(
            max_request_tokens=1_000,
            max_response_tokens=100,
            max_budget_usd=-1.0,
        )


def test_carries_no_compaction_policy() -> None:
    """Compaction is a heuristic; its knobs live on the ``Compactor``."""
    names = set(AgentSettings.__dataclass_fields__)
    assert not {n for n in names if n.startswith("reattach")}
    assert "persist_tokens" not in names
    assert "message_budget_tokens" not in names
    assert "keep_recent_on_compact" not in names
    assert "chars_per_token" not in names


# ---- TokenCount ------------------------------------------------------------


def test_token_count_sub_clamps_at_zero() -> None:
    """B10: ``__sub__`` never produces a negative field.

    ``CostTracker.restore_totals`` may move the cumulative total *below*
    a pre-restore snapshot; the status pane reads ``current - snapshot``
    and would render "-12 tokens" without the clamp.
    """
    small = TokenCount(request=10, response=5)
    big = TokenCount(request=100, response=50)
    diff = small - big
    assert diff.request == 0
    assert diff.response == 0
    assert diff.cache_write == 0
    assert diff.cache_read == 0


def test_token_count_sub_returns_positive_for_monotonic_case() -> None:
    """The clamp doesn't break the common monotonic path."""
    big = TokenCount(request=100, response=50)
    small = TokenCount(request=10, response=5)
    diff = big - small
    assert diff.request == 90
    assert diff.response == 45


def test_token_count_add_returns_not_implemented_for_non_token_count() -> None:
    """B15: arithmetic with a non-``TokenCount`` defers to the other operand.

    Pre-fix, ``TokenCount() + 1`` raised ``AttributeError`` mid-expression
    when Python's ``__add__`` dispatch tried to read ``other.request``.
    Returning ``NotImplemented`` lets the runtime try ``other.__radd__``
    and (if that also fails) raise a clear ``TypeError`` naming both
    operand types.
    """
    other = cast(TokenCount, 1)
    assert TokenCount().__add__(other) is NotImplemented


def test_token_count_sub_returns_not_implemented_for_non_token_count() -> None:
    """B15: ``__sub__`` similarly defers instead of raising ``AttributeError``."""
    other = cast(TokenCount, "nope")
    assert TokenCount().__sub__(other) is NotImplemented


# ---- ModelRecipe -----------------------------------------------------------


def test_model_recipe_rejects_empty_provider() -> None:
    """B14: empty ``provider`` would resolve to "no such provider" downstream."""
    with pytest.raises(ValueError, match="provider"):
        _ = ModelRecipe(provider="", auth="env", model_id="m")


def test_model_recipe_rejects_empty_auth() -> None:
    with pytest.raises(ValueError, match="auth"):
        _ = ModelRecipe(provider="P", auth="", model_id="m")


def test_model_recipe_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        _ = ModelRecipe(provider="P", auth="env", model_id="")


def test_model_recipe_allows_empty_account() -> None:
    """``account=""`` means "default backend"; not a degenerate recipe."""
    recipe = ModelRecipe(provider="P", auth="env", model_id="m", account="")
    assert recipe.account == ""


def test_model_recipe_allows_none_account() -> None:
    recipe = ModelRecipe(provider="P", auth="env", model_id="m", account=None)
    assert recipe.account is None


# ---- StreamInterruptedError ------------------------------------------------


def test_stream_interrupted_message_embeds_response_counts() -> None:
    """B8: the hardcoded message now surfaces token counts + stop_reason.

    The provider's stop signal is the operator's first hint at what
    went wrong; embedding it in the exception message saves a separate
    log dive.
    """
    response = ModelResponse(
        message=AssistantMessage(text="partial"),
        tokens=TokenCount(request=42, response=7),
        stop_reason="tool_use",
    )
    err = StreamInterruptedError(response)
    text = str(err)
    assert "input_tokens=42" in text
    assert "output_tokens=7" in text
    assert "tool_use" in text


# ---- model-id tags ---------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "base"),
    [
        ("claude-opus-4-7+1m", "claude-opus-4-7"),
        ("claude-opus-4-7+200k", "claude-opus-4-7"),
        ("claude-opus-4-7", "claude-opus-4-7"),
        ("Claude-Opus-4-7+1M", "Claude-Opus-4-7"),
    ],
)
def test_base_model_id_strips_the_context_tag(model_id: str, base: str) -> None:
    assert base_model_id(model_id) == base


@pytest.mark.parametrize(
    ("model_id", "base", "tags"),
    [
        ("claude-opus-4-8", "claude-opus-4-8", frozenset[ContextTag]()),
        ("claude-opus-4-8+1m", "claude-opus-4-8", frozenset({"+1m"})),
        ("claude-opus-4-8+200k", "claude-opus-4-8", frozenset({"+200k"})),
        ("Claude-Opus-4-8+1M", "Claude-Opus-4-8", frozenset({"+1m"})),
        ("model+unknown", "model+unknown", frozenset[ContextTag]()),
    ],
)
def test_split_model_id(model_id: str, base: str, tags: frozenset[ContextTag]) -> None:
    assert split_model_id(model_id) == (base, tags)


def test_context_tags_derive_from_the_literal() -> None:
    """A tag the type admits but the tuple omits would be unparseable."""
    assert set(CONTEXT_TAGS) == {"+1m", "+200k"}


def test_no_latency_tag_survives() -> None:
    """``+fast`` was a second spelling of ``service_tier="priority"``."""
    assert split_model_id("claude-opus-5+fast") == (
        "claude-opus-5+fast",
        frozenset(),
    )


# ---- catalog facts ---------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "priority"),
    [
        ("claude-opus-5", True),
        ("claude-opus-4-8", True),
        # Fast mode was removed from 4-7 on 2026-07-24 and never shipped on
        # 4-6: a fast request there is served at standard speed and billed at
        # standard rates, so a fast price row overstates the cost.
        ("claude-opus-4-7", False),
        ("claude-opus-4-6", False),
    ],
)
def test_only_documented_models_offer_the_priority_tier(
    model_id: str, priority: bool
) -> None:
    """Fast mode is Opus 5 and Opus 4.8 only.

    https://code.claude.com/docs/en/fast-mode
    """
    row = anthropic_catalog.models()[model_id]
    assert ("priority" in row.service_tier) is priority


@pytest.mark.parametrize(
    ("model_id", "divisor"),
    [
        ("claude-opus-5", 2.38),
        ("claude-opus-4-8", 2.38),
        ("claude-opus-4-6", 3.12),
        ("claude-haiku-4-5", 3.12),
    ],
)
def test_chars_per_token_is_provider_internal(model_id: str, divisor: float) -> None:
    """A divisor is not a capability: nothing SELECTS one."""
    assert anthropic_catalog.chars_per_token(model_id) == divisor
    assert "chars_per_token" not in ModelCapability.__dataclass_fields__


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
