"""Tests for ``providers.lib.errors``: shared provider-boundary helpers."""

from __future__ import annotations

import httpx
import pytest

from sagent.providers.lib.errors import (
    StreamingResponseNotReadError,
    error_status_code,
    find_response_not_read,
    is_request_too_large,
    raise_if_request_too_large,
)
from sagent.types.model import RequestTooLargeError


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


def test_is_request_too_large_status_413() -> None:
    """A bare 413 defaults to byte-limit when the body is not token-context."""
    assert is_request_too_large(413, "anything") is True


def test_is_request_too_large_body_phrases() -> None:
    """Status-less variants are caught by body phrase."""
    for body in (
        '{"error": {"type": "request_too_large"}}',
        "The maximum request size is 32 MB",
        "Request exceeds the maximum allowed number of bytes",
        "413 Payload Too Large",
        "Request Entity Too Large",
    ):
        assert is_request_too_large(None, body) is True, body


def test_is_request_too_large_false_for_token_overflow() -> None:
    """Token context-window overflow is a different condition."""
    assert is_request_too_large(400, "prompt is too long") is False
    assert is_request_too_large(400, "input too large for model context") is False
    assert is_request_too_large(400, "maximum context length exceeded") is False


def test_is_request_too_large_false_for_unrelated() -> None:
    assert is_request_too_large(None, "nope") is False
    assert is_request_too_large(500, "internal error") is False


def test_ambiguous_byte_prefix_does_not_force_byte_classification() -> None:
    """A body whose only "byte" signal is an ambiguous prefix stays token.

    "request exceeds the maximum context length" and "Request too large: ...
    context window" are TOKEN overflows (a larger window relieves them). The
    classifier no longer treats the loose prefixes "request exceeds the
    maximum" / "request too large" as byte signals -- only unambiguous
    request-size phrases qualify -- so the token context phrase decides.
    """
    assert (
        is_request_too_large(
            413, "request exceeds the maximum context length for this model"
        )
        is False
    )
    assert (
        is_request_too_large(
            400, "Request too large: prompt exceeds the model context window"
        )
        is False
    )


def test_unambiguous_byte_phrase_wins_over_co_occurring_context_phrase() -> None:
    """An unambiguous byte phrase classifies as byte even with a context phrase.

    A 413 "request entity too large" whose remediation prose mentions the
    context window is still a BYTE overflow -- it must route to byte-overflow
    recovery, not a ``/model`` larger-window suggestion the byte ceiling
    ignores.
    """
    assert (
        is_request_too_large(
            413, "request entity too large; please reduce context window usage"
        )
        is True
    )


def test_raise_if_request_too_large_raises_on_413() -> None:
    with pytest.raises(RequestTooLargeError):
        raise_if_request_too_large(413, "Request exceeds the maximum size")


def test_raise_if_request_too_large_noop_on_token_overflow() -> None:
    raise_if_request_too_large(400, "prompt is too long")  # must not raise


class _StatusError(Exception):
    """Error carrying a typed ``status_code`` attribute."""

    def __init__(self, status_code: int) -> None:
        super().__init__("x")
        self.status_code = status_code


def test_error_status_code_from_status_attr() -> None:
    assert error_status_code(_StatusError(413)) == 413


def test_error_status_code_from_response_attr() -> None:
    request = httpx.Request("POST", "https://example.test")
    err = httpx.HTTPStatusError(
        "x", request=request, response=httpx.Response(413, request=request)
    )
    assert error_status_code(err) == 413


def test_error_status_code_absent_returns_none() -> None:
    assert error_status_code(RuntimeError("no status")) is None


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
