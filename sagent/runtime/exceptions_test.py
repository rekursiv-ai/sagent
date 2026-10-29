"""Tests for the runtime's own errors and its error-logging policy."""

from __future__ import annotations

import asyncio
import logging

import pytest

from sagent.runtime.exceptions import (
    BudgetExhaustedError,
    RoundLimitError,
    UserFacingError,
    describe,
    log_exception_or_warning,
    log_task_exception,
)


_LOGGER = "sagent.runtime.exceptions_test"


def test_budget_exhausted_names_spend_and_cap() -> None:
    error = BudgetExhaustedError(total_cost_usd=3.456, max_budget_usd=3.0)
    assert str(error) == "Budget exhausted: $3.46 >= $3.00"
    assert error.total_cost_usd == 3.456
    assert error.max_budget_usd == 3.0


def test_round_limit_names_rounds_and_cap() -> None:
    error = RoundLimitError(rounds=7, max_rounds=5)
    assert str(error) == "Tool-call rounds exhausted: 7 >= 5"
    assert error.rounds == 7
    assert error.max_rounds == 5


def test_describe_omits_an_empty_message() -> None:
    assert describe(TimeoutError("")) == "TimeoutError"


def test_describe_names_type_status_and_message() -> None:
    assert describe(ValueError("bad"), status=429) == "ValueError 429: bad"
    assert describe(ValueError("bad")) == "ValueError: bad"
    assert describe(ValueError(""), status=0) == "ValueError 0"


@pytest.mark.parametrize(
    "error",
    [
        UserFacingError("Run /login."),
        BudgetExhaustedError(total_cost_usd=2.0, max_budget_usd=1.0),
        RoundLimitError(rounds=3, max_rounds=3),
    ],
)
def test_user_facing_error_logs_a_warning_without_traceback(
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    logger = logging.getLogger(_LOGGER)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        log_exception_or_warning(logger, "op", error)
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == f"op: {error}"
    assert record.exc_info is None


def test_other_error_logs_an_error_with_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger(_LOGGER)
    error = RuntimeError("boom")
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        log_exception_or_warning(logger, "call", error)
    (record,) = caplog.records
    assert record.levelno == logging.ERROR
    assert record.getMessage() == "call: boom"
    assert record.exc_info == (RuntimeError, error, error.__traceback__)


async def _raise(error: BaseException) -> None:
    raise error


async def _finish() -> None:
    return


async def _block() -> None:
    await asyncio.Event().wait()


async def _settled(task: asyncio.Task[None]) -> None:
    """Wait for ``task`` to finish and its done-callbacks to run."""
    await asyncio.wait((task,))
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_task_callback_logs_a_user_facing_failure_as_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    task = asyncio.create_task(_raise(UserFacingError("expired")))
    task.add_done_callback(log_task_exception(logging.getLogger(_LOGGER), "worker"))
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await _settled(task)
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert record.getMessage() == "worker: expired"
    assert record.exc_info is None


@pytest.mark.asyncio
async def test_task_callback_logs_an_unexpected_failure_with_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = RuntimeError("boom")
    task = asyncio.create_task(_raise(error))
    task.add_done_callback(log_task_exception(logging.getLogger(_LOGGER), "worker"))
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await _settled(task)
    (record,) = caplog.records
    assert record.levelno == logging.ERROR
    assert record.getMessage() == "worker: boom"
    assert record.exc_info == (RuntimeError, error, error.__traceback__)


@pytest.mark.asyncio
async def test_task_callback_is_silent_on_success_and_cancel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger(_LOGGER)
    done = asyncio.create_task(_finish())
    done.add_done_callback(log_task_exception(logger, "done"))
    cancelled = asyncio.create_task(_block())
    cancelled.add_done_callback(log_task_exception(logger, "cancelled"))
    await asyncio.sleep(0)
    cancelled.cancel()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await _settled(done)
        await _settled(cancelled)
    assert caplog.records == []


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
