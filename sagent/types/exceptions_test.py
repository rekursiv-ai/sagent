"""Tests for ``types.exceptions``: domain exception types."""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest

from sagent.types.exceptions import (
    AuthRefreshError,
    ContextOverflowError,
    UserFacingError,
    log_exception_or_warning,
    log_task_exception,
)
from sagent.types.model import (
    ModelResponse,
    ModelTerminationError,
    PromptTooLongError,
    StreamInterruptedError,
)
from sagent.types.runtime import AssistantMessage, ToolCall


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


def test_prompt_too_long_token_gap_zero_when_exactly_at_cap() -> None:
    """``actual == limit`` is at-cap, not unknown: returns ``0``.

    The provider rejected the prompt while ``actual == limit``; the gap
    is known (zero) and distinct from "we have no idea how much over".
    """
    err = PromptTooLongError(actual_tokens=100, limit_tokens=100)
    assert err.token_gap == 0


def test_prompt_too_long_token_gap_zero_when_below_cap() -> None:
    """Below-cap is still a known shape; gap clamps to ``0``."""
    err = PromptTooLongError(actual_tokens=50, limit_tokens=100)
    assert err.token_gap == 0


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


def test_context_overflow_error_is_user_facing() -> None:
    """``ContextOverflowError`` carries the user-facing-policy contract.

    The renderer renders ``UserFacingError`` instances without a
    ``ClassName:`` prefix and at WARNING (no traceback). Marking the
    overflow exhaustion error as ``UserFacingError`` keeps the message
    actionable instead of leaking ``PromptTooLongError: too long``.
    """
    err = ContextOverflowError("context window exhausted; /clear or /compact")
    assert isinstance(err, UserFacingError)
    assert "exhausted" in str(err)


class TestLogTaskException:
    """``log_task_exception`` -- done-callback for fire-and-forget tasks.

    The runtime + agent layer create many ``asyncio.create_task`` jobs
    whose results are never awaited (the model-call streamer, tool
    runners, persistent subagent drivers, REPL pumps, etc.). A task
    that raises with no ``await`` and no ``add_done_callback`` is
    silently swallowed -- only surfacing at GC time as an
    ``unretrieved exception`` warning that gets lost in normal log
    output. This helper standardises the done-callback so every
    fire-and-forget site logs a real traceback (or polished warning
    for ``UserFacingError``) at the point of failure.
    """

    @pytest.mark.asyncio
    async def test_plain_exception_logs_at_error_with_exc_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def _boom() -> None:
            raise RuntimeError("kaboom")

        logger = logging.getLogger("sagent.test.task.plain")
        task = asyncio.create_task(_boom())
        task.add_done_callback(log_task_exception(logger, "background worker"))
        with (
            caplog.at_level(logging.DEBUG, logger=logger.name),
            contextlib.suppress(RuntimeError),
        ):
            await task
        recs = [r for r in caplog.records if r.name == logger.name]
        assert len(recs) == 1, f"expected one log record, got {recs!r}"
        rec = recs[0]
        assert rec.levelname == "ERROR"
        assert rec.exc_info is not None, (
            "plain exception must carry exc_info so the operator sees the traceback"
        )
        assert "background worker" in rec.getMessage()

    @pytest.mark.asyncio
    async def test_user_facing_error_logs_at_warning_without_exc_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def _boom() -> None:
            raise AuthRefreshError("expired; run /login")

        logger = logging.getLogger("sagent.test.task.ufe")
        task = asyncio.create_task(_boom())
        task.add_done_callback(log_task_exception(logger, "auth worker"))
        with (
            caplog.at_level(logging.DEBUG, logger=logger.name),
            contextlib.suppress(AuthRefreshError),
        ):
            await task
        recs = [r for r in caplog.records if r.name == logger.name]
        assert len(recs) == 1
        rec = recs[0]
        assert rec.levelname == "WARNING"
        assert rec.exc_info is None, (
            f"UserFacingError must not carry exc_info; got {rec.exc_info!r}"
        )
        assert "expired; run /login" in rec.getMessage()
        assert "auth worker" in rec.getMessage()

    @pytest.mark.asyncio
    async def test_cancelled_task_does_not_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def _sleep() -> None:
            await asyncio.sleep(60)

        logger = logging.getLogger("sagent.test.task.cancel")
        task = asyncio.create_task(_sleep())
        task.add_done_callback(log_task_exception(logger, "sleeper"))
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            _ = task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        recs = [r for r in caplog.records if r.name == logger.name]
        assert recs == [], (
            f"cancellation must not log; got {[r.getMessage() for r in recs]!r}"
        )

    @pytest.mark.asyncio
    async def test_clean_completion_does_not_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def _ok() -> None:
            return None

        logger = logging.getLogger("sagent.test.task.ok")
        task = asyncio.create_task(_ok())
        task.add_done_callback(log_task_exception(logger, "ok worker"))
        with caplog.at_level(logging.DEBUG, logger=logger.name):
            await task
        recs = [r for r in caplog.records if r.name == logger.name]
        assert recs == [], (
            f"clean completion must not log; got {[r.getMessage() for r in recs]!r}"
        )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
