"""Tests for ``custom_exceptions``: domain exception types."""

from __future__ import annotations

import pytest

from sagent.agent.runtime import AssistantMessage, ToolCall
from sagent.custom_exceptions import (
    ModelTerminationError,
    PromptTooLongError,
    StreamInterruptedError,
)
from sagent.custom_types import ModelResponse


def test_prompt_too_long_default_message() -> None:
    err = PromptTooLongError()
    assert str(err) == "prompt too long"
    assert err.actual_tokens is None
    assert err.limit_tokens is None
    assert err.token_gap is None


def test_prompt_too_long_custom_message() -> None:
    err = PromptTooLongError("boom", actual_tokens=150, limit_tokens=100)
    assert str(err) == "boom"
    assert err.actual_tokens == 150
    assert err.limit_tokens == 100
    assert err.token_gap == 50


def test_prompt_too_long_token_gap_zero_returns_none() -> None:
    """``token_gap`` is None when actual does not exceed limit."""
    err = PromptTooLongError(actual_tokens=100, limit_tokens=100)
    assert err.token_gap is None


def test_prompt_too_long_token_gap_negative_returns_none() -> None:
    err = PromptTooLongError(actual_tokens=50, limit_tokens=100)
    assert err.token_gap is None


def test_prompt_too_long_token_gap_unknown_when_actual_missing() -> None:
    err = PromptTooLongError(limit_tokens=100)
    assert err.token_gap is None


def test_prompt_too_long_token_gap_unknown_when_limit_missing() -> None:
    err = PromptTooLongError(actual_tokens=100)
    assert err.token_gap is None


def test_prompt_too_long_is_exception() -> None:
    err = PromptTooLongError()
    with pytest.raises(PromptTooLongError):
        raise err


def test_stream_interrupted_carries_response() -> None:
    response = ModelResponse(message=AssistantMessage(text="partial"))
    err = StreamInterruptedError(response)
    assert err.response is response
    assert "tool_use" in str(err)


def test_stream_interrupted_is_exception() -> None:
    response = ModelResponse(message=AssistantMessage(text=""))
    with pytest.raises(StreamInterruptedError):
        raise StreamInterruptedError(response)


def test_model_termination_text_only() -> None:
    response = ModelResponse(
        message=AssistantMessage(text="hello"),
        stop_reason="surprise_stop",
    )
    err = ModelTerminationError(response)
    assert err.response is response
    assert err.stop_reason == "surprise_stop"
    assert "surprise_stop" in str(err)
    assert "tool_calls=0" in str(err)
    assert "text_len=5" in str(err)


def test_model_termination_with_tool_calls() -> None:
    tc = ToolCall(id="c1", name="Echo", args={})
    response = ModelResponse(
        message=AssistantMessage(text="", tool_calls=(tc,)),
        stop_reason="weird",
    )
    err = ModelTerminationError(response)
    assert "tool_calls=1" in str(err)
    assert "text_len=0" in str(err)


def test_model_termination_is_exception() -> None:
    response = ModelResponse(message=AssistantMessage(text=""), stop_reason="x")
    with pytest.raises(ModelTerminationError):
        raise ModelTerminationError(response)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
