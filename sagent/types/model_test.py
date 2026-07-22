"""Tests for ``types.model``: validation and arithmetic on data classes."""

from __future__ import annotations

from typing import cast

import pytest

from sagent.types.model import (
    ContextBudget,
    Model,
    ModelResponse,
    ModelSpec,
    Pricing,
    StreamInterruptedError,
    TokenCount,
    base_model_id,
    latency_from_model_id,
    split_model_id,
)
from sagent.types.runtime import AssistantMessage


class _TinyModel:
    """Smallest-possible ``Model``-shaped stub for ``ContextBudget.from_model``.

    Only the two attributes ``from_model`` reads are declared; callers
    ``cast`` it to ``Model`` because constructing the full Protocol
    surface for a one-method test is more noise than signal.
    """

    def __init__(
        self, *, max_request_tokens: int = 4_096, max_response_tokens: int = 1_024
    ) -> None:
        self.max_request_tokens = max_request_tokens
        self.max_response_tokens = max_response_tokens


def _tiny(*, max_request_tokens: int = 4_096) -> Model:
    return cast(Model, _TinyModel(max_request_tokens=max_request_tokens))


# ---- ContextBudget ---------------------------------------------------------


def test_context_budget_from_model_handles_small_window() -> None:
    """B12: small windows must not trip the ``buffer < max_request`` invariant.

    Pre-fix, ``buffer_tokens = max(inp // 15, 8_000)`` floored at
    8_000; any model with ``max_request_tokens < 8_000`` then failed
    ``__post_init__``'s ``buffer_tokens < max_request_tokens`` check.
    Test fakes and small local models broke loud.
    """
    _ = ContextBudget.from_model(_tiny(max_request_tokens=4_096))


def test_context_budget_from_model_caps_buffer_at_half_window() -> None:
    """The buffer never exceeds half the window even for tiny models."""
    budget = ContextBudget.from_model(_tiny(max_request_tokens=4_096))
    assert budget.buffer_tokens <= 4_096 // 2


def test_context_budget_from_model_keep_recent_defers_to_compactor() -> None:
    """B13: ``from_model`` leaves ``keep_recent_on_compact`` as ``None``.

    ``None`` is the "let the compactor pick" contract; baking a number
    here would silently override the strategy-specific default.
    """
    budget = ContextBudget.from_model(_tiny())
    assert budget.keep_recent_on_compact is None


def test_context_budget_rejects_negative_reattach_count() -> None:
    """S7: ``reattach_count < 0`` underflows downstream slice math."""
    with pytest.raises(ValueError, match="reattach_count"):
        _ = ContextBudget(
            max_request_tokens=1_000,
            max_response_tokens=100,
            reattach_count=-1,
        )


def test_context_budget_rejects_negative_reattach_max_chars() -> None:
    with pytest.raises(ValueError, match="reattach_max_chars"):
        _ = ContextBudget(
            max_request_tokens=1_000,
            max_response_tokens=100,
            reattach_max_chars=-1,
        )


def test_context_budget_rejects_negative_reattach_budget() -> None:
    """S7: caught for downstream char-budget math."""
    with pytest.raises(ValueError, match="reattach_budget"):
        _ = ContextBudget(
            max_request_tokens=1_000,
            max_response_tokens=100,
            reattach_budget=-1,
        )


def test_context_budget_rejects_negative_persist_threshold() -> None:
    with pytest.raises(ValueError, match="persist_threshold"):
        _ = ContextBudget(
            max_request_tokens=1_000,
            max_response_tokens=100,
            persist_threshold=-1,
        )


def test_context_budget_rejects_negative_message_budget_chars() -> None:
    with pytest.raises(ValueError, match="message_budget_chars"):
        _ = ContextBudget(
            max_request_tokens=1_000,
            max_response_tokens=100,
            message_budget_chars=-1,
        )


def test_context_budget_rejects_negative_keep_recent_on_compact() -> None:
    """``keep_recent_on_compact`` is ``None | int``; ``-1`` is neither."""
    with pytest.raises(ValueError, match="keep_recent_on_compact"):
        _ = ContextBudget(
            max_request_tokens=1_000,
            max_response_tokens=100,
            keep_recent_on_compact=-1,
        )


def test_context_budget_accepts_none_keep_recent_on_compact() -> None:
    budget = ContextBudget(
        max_request_tokens=1_000,
        max_response_tokens=100,
        keep_recent_on_compact=None,
    )
    assert budget.keep_recent_on_compact is None


def test_context_budget_accepts_zero_buffer_tokens() -> None:
    """``buffer_tokens = 0`` is a legal "no headroom" setting."""
    budget = ContextBudget(
        max_request_tokens=1_000,
        max_response_tokens=100,
        buffer_tokens=0,
    )
    assert budget.buffer_tokens == 0


def test_context_budget_rejects_negative_buffer_tokens() -> None:
    """Negative ``buffer_tokens`` would inflate the usable window past the cap."""
    with pytest.raises(ValueError, match="buffer_tokens"):
        _ = ContextBudget(
            max_request_tokens=1_000,
            max_response_tokens=100,
            buffer_tokens=-1,
        )


# ---- TokenCount ------------------------------------------------------------


def test_token_count_sub_clamps_at_zero() -> None:
    """B10: ``__sub__`` never produces a negative field.

    ``CostTracker.restore_totals`` may move the cumulative total *below*
    a pre-restore snapshot; the status pane reads ``current - snapshot``
    and would render "-12 tokens" without the clamp.
    """
    small = TokenCount(input_tokens=10, output_tokens=5)
    big = TokenCount(input_tokens=100, output_tokens=50)
    diff = small - big
    assert diff.input_tokens == 0
    assert diff.output_tokens == 0
    assert diff.cache_creation_tokens == 0
    assert diff.cache_read_tokens == 0


def test_token_count_sub_returns_positive_for_monotonic_case() -> None:
    """The clamp doesn't break the common monotonic path."""
    big = TokenCount(input_tokens=100, output_tokens=50)
    small = TokenCount(input_tokens=10, output_tokens=5)
    diff = big - small
    assert diff.input_tokens == 90
    assert diff.output_tokens == 45


def test_token_count_add_returns_not_implemented_for_non_token_count() -> None:
    """B15: arithmetic with a non-``TokenCount`` defers to the other operand.

    Pre-fix, ``TokenCount() + 1`` raised ``AttributeError`` mid-expression
    when Python's ``__add__`` dispatch tried to read ``other.input_tokens``.
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


# ---- ModelSpec -------------------------------------------------------------


def test_model_spec_rejects_empty_provider() -> None:
    """B14: empty ``provider`` would resolve to "no such provider" downstream."""
    with pytest.raises(ValueError, match="provider"):
        _ = ModelSpec(provider="", auth="env", model_id="m")


def test_model_spec_rejects_empty_auth() -> None:
    with pytest.raises(ValueError, match="auth"):
        _ = ModelSpec(provider="P", auth="", model_id="m")


def test_model_spec_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        _ = ModelSpec(provider="P", auth="env", model_id="")


def test_model_spec_allows_empty_account() -> None:
    """``account=""`` means "default backend"; not a degenerate spec."""
    spec = ModelSpec(provider="P", auth="env", model_id="m", account="")
    assert spec.account == ""


def test_model_spec_allows_none_account() -> None:
    spec = ModelSpec(provider="P", auth="env", model_id="m", account=None)
    assert spec.account is None


# ---- StreamInterruptedError ------------------------------------------------


def test_stream_interrupted_message_embeds_response_counts() -> None:
    """B8: the hardcoded message now surfaces token counts + stop_reason.

    The provider's stop signal is the operator's first hint at what
    went wrong; embedding it in the exception message saves a separate
    log dive.
    """
    response = ModelResponse(
        message=AssistantMessage(text="partial"),
        tokens=TokenCount(input_tokens=42, output_tokens=7),
        stop_reason="tool_use",
    )
    err = StreamInterruptedError(response)
    text = str(err)
    assert "input_tokens=42" in text
    assert "output_tokens=7" in text
    assert "tool_use" in text


# ---- Pricing -- sanity-check default ---------------------------------------


def test_pricing_defaults_zero() -> None:
    """Pricing fields default to zero -- non-priced models stay free."""
    p = Pricing()
    assert p.request == 0.0
    assert p.response == 0.0


@pytest.mark.parametrize(
    ("model_id", "base"),
    [
        ("claude-opus-4-7+1m", "claude-opus-4-7"),
        ("claude-opus-4-7+200k", "claude-opus-4-7"),
        ("claude-opus-4-7", "claude-opus-4-7"),
        ("Claude-Opus-4-7+1M", "Claude-Opus-4-7"),
    ],
)
def test_base_model_id_strips_window_tag(model_id: str, base: str) -> None:
    assert base_model_id(model_id) == base


@pytest.mark.parametrize(
    ("model_id", "base", "tags"),
    [
        ("claude-opus-4-8", "claude-opus-4-8", frozenset[str]()),
        ("claude-opus-4-8+fast", "claude-opus-4-8", frozenset({"+fast"})),
        ("claude-opus-4-8+1m+fast", "claude-opus-4-8", frozenset({"+1m", "+fast"})),
        ("claude-opus-4-8+fast+1m", "claude-opus-4-8", frozenset({"+1m", "+fast"})),
        ("Claude-Opus-4-8+FAST", "Claude-Opus-4-8", frozenset({"+fast"})),
        ("model+unknown", "model+unknown", frozenset[str]()),
    ],
)
def test_split_model_id(model_id: str, base: str, tags: frozenset[str]) -> None:
    assert split_model_id(model_id) == (base, tags)


def test_latency_from_model_id() -> None:
    assert latency_from_model_id("claude-opus-4-8+fast") == "fast"
    assert latency_from_model_id("claude-opus-4-8+1m+fast") == "fast"
    assert latency_from_model_id("claude-opus-4-8+1m") is None
    assert latency_from_model_id("claude-opus-4-8") is None


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
