"""Tests for ``types.exceptions``: domain exception types."""

from __future__ import annotations

import logging

import pytest

from sagent.types.exceptions import (
    AuthRefreshError,
    ModelTerminationError,
    PromptTooLongError,
    StreamInterruptedError,
    UserFacingError,
    log_exception_or_warning,
)
from sagent.types.history import AssistantMessage, ToolCall
from sagent.types.model import ModelResponse


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


class TestLogExceptionOrWarning:
    """``log_exception_or_warning`` -- centralized user-facing-error policy.

    Three model-calling paths (``_stream_and_post``, the runtime's
    compaction handler, the agent's synchronous overflow-recovery
    compaction, ``post_compact_enrich``) all needed the same branch:
    log at WARNING without a traceback for ``UserFacingError``, log at
    ERROR with traceback for everything else. Centralizing the rule
    here gives one place to evolve the policy and keeps callsites
    short.
    """

    @staticmethod
    def _raise_user_facing() -> None:
        raise AuthRefreshError("expired; run /login")

    @staticmethod
    def _raise_runtime() -> None:
        raise RuntimeError("something broke")

    def test_user_facing_error_logs_at_warning_without_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = logging.getLogger("sagent.test.ufe")
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            try:
                self._raise_user_facing()
            except AuthRefreshError as exc:
                log_exception_or_warning(logger, "model call failed", exc)
        recs = [r for r in caplog.records if r.name == logger.name]
        assert len(recs) == 1
        rec = recs[0]
        assert rec.levelname == "WARNING"
        assert rec.exc_info is None, (
            f"UserFacingError must not carry exc_info; got {rec.exc_info!r}"
        )
        assert "expired; run /login" in rec.getMessage()
        assert "model call failed" in rec.getMessage()

    def test_plain_exception_logs_at_error_with_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = logging.getLogger("sagent.test.plain")
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            try:
                self._raise_runtime()
            except RuntimeError as exc:
                log_exception_or_warning(logger, "compaction failed", exc)
        recs = [r for r in caplog.records if r.name == logger.name]
        assert len(recs) == 1
        rec = recs[0]
        assert rec.levelname == "ERROR"
        assert rec.exc_info is not None, (
            "plain exception must carry exc_info so the operator sees the traceback"
        )

    def test_user_facing_subclass_treated_as_user_facing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Any subclass of ``UserFacingError`` follows the warning policy."""

        class _CustomUserFacingError(UserFacingError):
            pass

        def _raise() -> None:
            raise _CustomUserFacingError("oops")

        logger = logging.getLogger("sagent.test.sub")
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            try:
                _raise()
            except _CustomUserFacingError as exc:
                log_exception_or_warning(logger, "thing failed", exc)
        recs = [r for r in caplog.records if r.name == logger.name]
        assert len(recs) == 1
        assert recs[0].levelname == "WARNING"
        assert recs[0].exc_info is None


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
