"""Tests for the model errors: which retry, and what a caller catches them as."""

from __future__ import annotations

import pytest

from sagent.model.exceptions import (
    AuthError,
    BillingError,
    ContextOverflowError,
    DisconnectedError,
    OverloadedError,
    RateLimitedError,
    RefusedError,
    RequestTooLargeError,
    RetryableError,
    UnknownModelError,
    UnknownProviderError,
    UnsupportedTagError,
)


def test_a_retryable_error_carries_the_advised_wait() -> None:
    error = RateLimitedError("slow down", retry_after_sec=30.0)
    assert error.retry_after_sec == 30.0
    assert str(error) == "slow down"


def test_no_advised_wait_is_zero() -> None:
    assert RetryableError("x").retry_after_sec == 0.0
    assert DisconnectedError("x").retry_after_sec == 0.0


@pytest.mark.parametrize("kind", [RateLimitedError, OverloadedError, DisconnectedError])
def test_transient_failures_are_retryable(kind: type[RetryableError]) -> None:
    assert isinstance(kind("x"), RetryableError)


@pytest.mark.parametrize(
    "kind",
    [AuthError, BillingError, ContextOverflowError, RequestTooLargeError, RefusedError],
)
def test_waiting_does_not_help_these(kind: type[Exception]) -> None:
    assert not isinstance(kind("x"), RetryableError)


@pytest.mark.parametrize(
    "kind",
    [UnknownModelError, UnknownProviderError, UnsupportedTagError],
)
def test_a_bad_model_name_is_a_value_error(kind: type[ValueError]) -> None:
    """``/model`` reports a bad name by catching ``ValueError``."""
    assert isinstance(kind("x"), ValueError)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
