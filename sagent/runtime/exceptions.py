"""Errors the agent raises itself, and how the runtime logs errors."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    import asyncio
    import logging


__all__ = [
    "BudgetExhaustedError",
    "RoundLimitError",
    "UserFacingError",
    "describe",
    "log_exception_or_warning",
    "log_task_exception",
]


class UserFacingError(Exception):
    """An error whose message tells the user what to do: logged with no traceback."""


class BudgetExhaustedError(UserFacingError):
    """The agent's spend reached its ``max_spend_usd`` cap.

    Args:
      total_cost_usd: What the agent has spent.
      max_budget_usd: The cap it reached.

    """

    def __init__(self, *, total_cost_usd: float, max_budget_usd: float) -> None:
        super().__init__(
            f"Budget exhausted: ${total_cost_usd:.2f} >= ${max_budget_usd:.2f}",
        )
        self.total_cost_usd = total_cost_usd
        self.max_budget_usd = max_budget_usd


class RoundLimitError(UserFacingError):
    """A turn reached the agent's cap on tool-call rounds.

    Args:
      rounds: Tool-call rounds the turn ran.
      max_rounds: The cap it reached.

    """

    def __init__(self, *, rounds: int, max_rounds: int) -> None:
        super().__init__(f"Tool-call rounds exhausted: {rounds} >= {max_rounds}")
        self.rounds = rounds
        self.max_rounds = max_rounds


# Httpx raises ``ConnectTimeout("")``: joining type and message unconditionally
# renders it ``ConnectTimeout: ``.
def describe(error: BaseException, *, status: int | None = None) -> str:
    """Name ``error`` by type, HTTP status, and message, omitting empty parts."""
    status_part = f" {status}" if status is not None else ""
    message = str(error)
    return f"{type(error).__name__}{status_part}" + (f": {message}" if message else "")


# ``exc_info=exc``, not ``True``: called outside an ``except`` block,
# ``sys.exc_info()`` is empty and the traceback would be lost.
def log_exception_or_warning(
    logger: logging.Logger,
    msg: str,
    exc: BaseException,
) -> None:
    """Log ``msg: exc``: a warning for a ``UserFacingError``, else an error with its traceback.

    Args:
      logger: Where the line goes.
      msg: What failed.
      exc: Why.

    """
    if isinstance(exc, UserFacingError):
        logger.warning("%s: %s", msg, exc)
    else:
        logger.error("%s: %s", msg, exc, exc_info=exc)


# A task nothing awaits loses its exception until garbage collection, which
# reports it as "Task exception was never retrieved", too late to act on.
def log_task_exception(
    logger: logging.Logger,
    where: str,
) -> Callable[[asyncio.Task[object]], None]:
    """Return a done-callback that logs a task's exception as ``log_exception_or_warning`` does.

    Args:
      logger: Where the line goes.
      where: What the task was doing.

    Returns:
      callback: For ``task.add_done_callback``; silent when the task was
        cancelled or ended cleanly.

    """
    return partial(_log_task, logger, where)


def _log_task(logger: logging.Logger, where: str, task: asyncio.Task[object]) -> None:
    """Log the exception ``task`` ended with, if it ended with one."""
    if not task.cancelled() and (exc := task.exception()) is not None:
        log_exception_or_warning(logger, where, exc)
