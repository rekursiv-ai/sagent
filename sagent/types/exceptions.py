"""Domain exceptions for the sagent framework."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    import asyncio
    import logging


__all__ = [
    "AuthRefreshError",
    "BudgetExhaustedError",
    "ContextOverflowError",
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


class BudgetExhaustedError(UserFacingError):
    """Cumulative model spend reached the agent's ``max_budget_usd`` cap.

    Raised by ``Agent.record_response`` when the cost tracker crosses
    the configured cap. The message embeds the realized spend and the
    configured cap so the REPL renderer can present them verbatim
    without ``ClassName:`` noise.

    Args:
      total_cost_usd: Realized cumulative spend at the time of raise.
      max_budget_usd: The configured cap that was crossed.

    """

    def __init__(self, *, total_cost_usd: float, max_budget_usd: float) -> None:
        super().__init__(
            f"Budget exhausted: ${total_cost_usd:.2f} >= ${max_budget_usd:.2f}",
        )
        self.total_cost_usd = total_cost_usd
        self.max_budget_usd = max_budget_usd


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

    Args:
      message: Polished remediation text shown verbatim to the user.
      attempts: Number of overflow-recovery iterations that elapsed
          before exhaustion; ``0`` when the first ``compact_now`` itself
          raised.
      final_tokens: Estimated input token count after the last failed
          recovery attempt; ``None`` when unknown (e.g. compactor
          raised before a re-estimate).

    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 0,
        final_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.final_tokens = final_tokens


def log_exception_or_warning(
    logger: logging.Logger, msg: str, exc: BaseException
) -> None:
    """Log ``msg`` per the user-facing-error policy.

    - ``UserFacingError`` (or subclass): ``logger.warning("%s: %s", msg, exc)``
      -- no traceback. The exception's message is already polished
      remediation text the user can act on; a Python traceback is
      noise.
    - Anything else: ``logger.error(msg, exc_info=exc)`` -- traceback
      retained so the operator can diagnose the unexpected failure.

    Passing ``exc_info=exc`` explicitly keeps the traceback even when
    called outside an ``except`` block, where ``sys.exc_info()`` is empty.
    """
    if isinstance(exc, UserFacingError):
        logger.warning("%s: %s", msg, exc)
    else:
        logger.error(msg, exc_info=exc)


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

    Note: ``Task.exception()`` returns ``BaseException`` (asyncio
    surfaces ``KeyboardInterrupt`` / ``SystemExit`` through this same
    accessor for tasks that propagated them). The ``UserFacingError``
    branch catches only the ``Exception`` subset; non-``Exception``
    ``BaseException`` instances fall through to the ``error`` branch
    with ``exc_info`` so the operator sees the originating traceback.
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
            # Mirror ``log_exception_or_warning``: ``"%s: %s"`` lets the
            # operator scan ``where`` + the exception message on the
            # same line; ``exc_info`` adds the traceback below.
            logger.error("%s: %s", where, exc, exc_info=exc)

    return _cb
