"""Tests for ``providers.lib.errors``: shared provider-boundary helpers."""

from __future__ import annotations

import httpx

from sagent.providers.lib.errors import (
    StreamingResponseNotReadError,
    find_response_not_read,
)


def test_find_response_not_read_direct() -> None:
    err = httpx.ResponseNotRead()
    assert find_response_not_read(err) is err


def test_find_response_not_read_via_cause() -> None:
    inner = httpx.ResponseNotRead()
    outer = RuntimeError("formatting failed")
    outer.__cause__ = inner
    assert find_response_not_read(outer) is inner


def test_find_response_not_read_via_context() -> None:
    inner = httpx.ResponseNotRead()
    outer = RuntimeError("formatting failed")
    outer.__context__ = inner
    assert find_response_not_read(outer) is inner


def test_find_response_not_read_absent_returns_none() -> None:
    assert find_response_not_read(RuntimeError("unrelated")) is None


def test_find_response_not_read_handles_cyclic_chain() -> None:
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__context__ = a
    assert find_response_not_read(a) is None


def test_streaming_response_not_read_error_carries_cause() -> None:
    cause = httpx.ResponseNotRead()
    err = StreamingResponseNotReadError(provider_name="Anthropic", cause=cause)
    assert err.__cause__ is cause
    assert "Anthropic" in str(err)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
