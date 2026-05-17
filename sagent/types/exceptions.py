"""Domain exceptions for the sagent framework."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    import asyncio
    import logging

    from sagent.types.model import ModelResponse


__all__ = [
    "AuthRefreshError",
    "ContextOverflowError",
    "ModelTerminationError",
    "PromptTooLongError",
    "StreamInterruptedError",
    "UserFacingError",
    "log_exception_or_warning",
    "log_task_exception",
]


class UserFacingError(Exception):
    """Marker for errors whose message is already polished for the end user.

    The runtime's error-handling path (``_stream_and_post`` and the REPL
    renderer) treats these specially: log at ``warning`` level without a
    traceback, present ``str(exc)`` verbatim, and recommend the action
    the message already encodes. Subclass this for any error path where
    the message itself is the remediation cue (auth expired, network
    down, etc.).
    """


class AuthRefreshError(UserFacingError):
    """Provider OAuth refresh failed in a way the user must act on.

    Raised when ``refresh_token`` exchange returns 400/401 -- the token
    was rotated by another process, revoked server-side, or aged out.
    Retrying the call will fail identically; the user must re-auth via
    ``/login``. The message embeds the recommended action so renderers
    can show it verbatim.
    """


class ContextOverflowError(UserFacingError):
    """Context window exhausted; auto-compaction could not make progress.

    Raised by ``_AgentModel.stream`` when reactive overflow recovery
    has run its course -- either ``compact_now`` itself failed (the
    compactor call raised, typically because the compaction request
    was also too large) or ``MAX_OVERFLOW_RECOVERY`` retries elapsed
    without the history shrinking enough to fit. The message embeds
    the recommended remediation (``/clear``, ``/compact <hints>``,
    ``/model`` to a larger window) so the REPL renderer can show it
    verbatim without ``ClassName:`` noise. The original provider
    exception is preserved via ``__cause__``.
    """


def log_exception_or_warning(
    logger: logging.Logger, msg: str, exc: BaseException
) -> None:
    """Log ``msg`` per the user-facing-error policy.

    - ``UserFacingError`` (or subclass): ``logger.warning("%s: %s", msg, exc)``
      -- no traceback. The exception's message is already polished
      remediation text the user can act on; a Python traceback is
      noise.
    - Anything else: ``logger.exception(msg)`` -- traceback retained
      so the operator can diagnose the unexpected failure.

    Call from inside an ``except`` block so the ``exception`` path
    can pick up ``sys.exc_info()``.
    """
    if isinstance(exc, UserFacingError):
        logger.warning("%s: %s", msg, exc)
    else:
        logger.exception(msg)


def log_task_exception(
    logger: logging.Logger, where: str
) -> Callable[[asyncio.Task[object]], None]:
    """Build an ``asyncio.Task`` done-callback that logs unhandled errors.

    Fire-and-forget ``asyncio.create_task`` sites swallow exceptions
    silently -- ``asyncio`` only surfaces them as
    ``Task exception was never retrieved`` warnings at GC, which arrive
    too late and too quietly to be actionable. Attach this callback
    to every such site to log the failure at the moment it happens.

    Usage::

        task = asyncio.create_task(work())
        task.add_done_callback(log_task_exception(logger, "what worker"))

    Policy mirrors :func:`log_exception_or_warning`: ``UserFacingError``
    logs at ``WARNING`` with no traceback (the message is already
    polished remediation text); anything else logs at ``ERROR`` with
    ``exc_info`` so the operator gets a traceback. ``CancelledError``
    and clean completion are silent.
    """

    def _cb(task: asyncio.Task[object]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        if isinstance(exc, UserFacingError):
            logger.warning("%s: %s", where, exc)
        else:
            logger.error("%s", where, exc_info=exc)

    return _cb


class PromptTooLongError(Exception):
    """Raised by providers when the prompt exceeds model limits.

    Args:
      message: Error message.
      actual_tokens: Actual token count that exceeded the limit.
      limit_tokens: Maximum allowed token count.

    """

    def __init__(
        self,
        message: str = "prompt too long",
        *,
        actual_tokens: int | None = None,
        limit_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.actual_tokens = actual_tokens
        self.limit_tokens = limit_tokens

    @property
    def token_gap(self) -> int | None:
        """Return the number of tokens over the limit, or None if unknown."""
        if self.actual_tokens is not None and self.limit_tokens is not None:
            gap = self.actual_tokens - self.limit_tokens
            return gap if gap > 0 else None
        return None


class StreamInterruptedError(Exception):
    """Stream finished with ``stop_reason=tool_use`` but delivered no tool blocks.

    The Anthropic API flags ``stop_reason`` as not always set correctly
    when streaming -- the SDK may reconstruct a final message that
    records the intended terminal reason while the ``tool_use`` block
    itself was dropped mid-stream. Retrying the same request usually
    recovers the tool call; if it doesn't, the carried ``response`` lets
    the agent fall back to returning whatever partial text/thinking was
    delivered instead of looping into an API 400.

    Args:
      response: The partial model response.

    """

    def __init__(self, response: ModelResponse) -> None:
        super().__init__(
            "Stream indicated tool_use but delivered no tool blocks",
        )
        self.response = response


class ModelTerminationError(Exception):
    """Model stopped with an unrecognized non-benign ``stop_reason``.

    Safety net for stop_reasons we don't have an explicit handler for
    (e.g. a new provider value, a vocabulary drift). Recognized
    non-benign reasons (``max_tokens``, ``model_context_window_exceeded``,
    ``refusal``) are handled in the agent loop without raising  --
    ``max_tokens`` triggers the recovery flow, ``refusal`` surfaces
    as a user-visible message.

    Args:
      response: The model response with the unrecognized stop reason.

    """

    def __init__(self, response: ModelResponse) -> None:
        tool_count = len(response.message.tool_calls)
        text_len = len(response.message.text)
        super().__init__(
            f"Model stopped with unrecognized stop_reason="
            f"{response.stop_reason!r} (tool_calls={tool_count}, "
            f"text_len={text_len}).",
        )
        self.response = response
        self.stop_reason = response.stop_reason
