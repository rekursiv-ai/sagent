"""Tests for ``agent.retry``: classification, backoff, send-with-retry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import override

import time

import httpx
import pytest

from sagent.agent.retry import (
    MAX_RETRY_DELAY,
    PERSISTENT_MAX_BACKOFF_SEC,
    RETRY_BASE_SEC,
    RETRYABLE_STATUS_CODES,
    RateLimitError,
    RetriesExhaustedError,
    error_status,
    extract_retry_after,
    is_retryable,
    send_with_retry,
)
from sagent.testing import MockModelCaps
from sagent.types.exceptions import StreamInterruptedError
from sagent.types.history import AssistantMessage
from sagent.types.model import ModelRequest, ModelResponse


@dataclass(slots=True, kw_only=True)
class _ScriptedModel(MockModelCaps):
    """Model with a scripted response queue and optional fault injection."""

    model_id: str = "scripted"
    max_request_tokens: int = 100_000
    stream_responses: list[BaseException | ModelResponse] = field(default_factory=list)
    buffer_responses: list[BaseException | ModelResponse] = field(default_factory=list)
    is_retryable_provider: bool = False
    is_overflow: bool = False
    _stream_idx: int = field(default=0, init=False)
    _buffer_idx: int = field(default=0, init=False)
    stream_calls: int = field(default=0, init=False)
    buffer_calls: int = field(default=0, init=False)

    @override
    def is_retryable_provider_error(self, error: Exception) -> bool:
        del error
        return self.is_retryable_provider

    @override
    def is_context_overflow(self, error: Exception) -> bool:
        del error
        return self.is_overflow

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del request, on_thinking
        self.stream_calls += 1
        item = self.stream_responses[self._stream_idx]
        self._stream_idx += 1
        if isinstance(item, BaseException):
            raise item
        if on_text is not None and item.message.text:
            for ch in item.message.text:
                on_text(ch)
        return item

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        del request
        self.buffer_calls += 1
        item = self.buffer_responses[self._buffer_idx]
        self._buffer_idx += 1
        if isinstance(item, BaseException):
            raise item
        return item


def _request() -> ModelRequest:
    return ModelRequest(messages=[])


def _resp(text: str = "ok") -> ModelResponse:
    return ModelResponse(message=AssistantMessage(text=text))


def _silent(_text: str) -> None:
    return None


class _FakeResponse:
    """Minimal stand-in for ``httpx.Response`` carrying status + headers."""

    def __init__(
        self,
        status_code: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}


class _HTTPError(Exception):
    """Exception with attached response attribute."""

    def __init__(self, response: _FakeResponse) -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


class _StatusCodeError(Exception):
    """Exception carrying ``status_code`` directly (not under ``.response``)."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_is_retryable_provider_short_circuit() -> None:
    model = _ScriptedModel(is_retryable_provider=True)
    assert is_retryable(ValueError("anything"), model) is True


def test_is_retryable_connection_error() -> None:
    assert is_retryable(ConnectionError("nope"), _ScriptedModel()) is True


def test_is_retryable_timeout_error() -> None:
    assert is_retryable(TimeoutError("slow"), _ScriptedModel()) is True


def test_is_retryable_httpx_transport_error() -> None:
    err = httpx.ConnectError("conn refused")
    assert is_retryable(err, _ScriptedModel()) is True


def test_is_retryable_status_5xx() -> None:
    err = _HTTPError(_FakeResponse(503))
    assert is_retryable(err, _ScriptedModel()) is True


def test_is_retryable_status_429() -> None:
    err = _HTTPError(_FakeResponse(429))
    assert is_retryable(err, _ScriptedModel()) is True


def test_is_retryable_status_400_is_fatal() -> None:
    err = _HTTPError(_FakeResponse(400))
    assert is_retryable(err, _ScriptedModel()) is False


def test_is_retryable_walks_cause_chain() -> None:
    """A wrapped retryable cause makes the outer error retryable."""
    inner = ConnectionError("inner")
    outer = RuntimeError("outer")
    outer.__cause__ = inner
    assert is_retryable(outer, _ScriptedModel()) is True


def test_is_retryable_cause_depth_capped() -> None:
    """Pathological cause cycles don't hang the classifier."""
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a  # cycle
    # The depth cap returns False once the cap is hit.
    assert is_retryable(a, _ScriptedModel()) is False


def test_is_retryable_unknown_error_is_fatal() -> None:
    assert is_retryable(ValueError("plain"), _ScriptedModel()) is False


def test_retryable_status_codes_membership() -> None:
    assert 408 in RETRYABLE_STATUS_CODES
    assert 409 in RETRYABLE_STATUS_CODES
    assert 429 in RETRYABLE_STATUS_CODES


def test_constants_have_sane_values() -> None:
    assert RETRY_BASE_SEC > 0.0
    assert MAX_RETRY_DELAY > RETRY_BASE_SEC
    assert PERSISTENT_MAX_BACKOFF_SEC > MAX_RETRY_DELAY


def test_error_status_via_status_code_attr() -> None:
    assert error_status(_StatusCodeError(503)) == 503


def test_error_status_via_response_attr() -> None:
    assert error_status(_HTTPError(_FakeResponse(404))) == 404


def test_error_status_missing_returns_none() -> None:
    assert error_status(ValueError("plain")) is None


def test_error_status_via_cause_chain() -> None:
    inner = _HTTPError(_FakeResponse(429))
    outer = RuntimeError("wrapped")
    outer.__cause__ = inner
    assert error_status(outer) == 429


def test_extract_retry_after_missing_response_returns_none() -> None:
    assert extract_retry_after(ValueError("plain")) is None


def test_extract_retry_after_seconds_header() -> None:
    err = _HTTPError(_FakeResponse(429, {"retry-after": "12"}))
    assert extract_retry_after(err) == pytest.approx(12.0)


def test_extract_retry_after_invalid_seconds_header_falls_through() -> None:
    err = _HTTPError(_FakeResponse(429, {"retry-after": "no-way"}))
    assert extract_retry_after(err) is None


def test_extract_retry_after_anthropic_unified_reset() -> None:
    reset = time.time() + 30.0
    err = _HTTPError(
        _FakeResponse(429, {"anthropic-ratelimit-unified-reset": str(reset)})
    )
    delay = extract_retry_after(err)
    assert delay is not None
    assert 25.0 <= delay <= 30.5


def test_extract_retry_after_invalid_unified_reset_returns_none() -> None:
    err = _HTTPError(
        _FakeResponse(429, {"anthropic-ratelimit-unified-reset": "not-a-time"})
    )
    assert extract_retry_after(err) is None


def test_extract_retry_after_no_relevant_headers() -> None:
    err = _HTTPError(_FakeResponse(503, {"x-other": "v"}))
    assert extract_retry_after(err) is None


def test_rate_limit_error_with_future_reset_includes_clock() -> None:
    reset = time.time() + 30.0
    original = _HTTPError(_FakeResponse(429))
    e = RateLimitError(reset, original)
    assert "Resumes at" in str(e)
    assert e.reset_time == reset
    assert e.original is original


def test_rate_limit_error_no_reset_falls_back() -> None:
    original = _HTTPError(_FakeResponse(429))
    e = RateLimitError(None, original)
    assert "Try again shortly" in str(e)
    assert e.reset_time is None


def test_rate_limit_error_past_reset_falls_back() -> None:
    original = _HTTPError(_FakeResponse(429))
    e = RateLimitError(time.time() - 100, original)
    assert "Try again shortly" in str(e)


@pytest.mark.asyncio
async def test_send_with_retry_streaming_first_try() -> None:
    model = _ScriptedModel(stream_responses=[_resp("hi")])
    chunks: list[str] = []
    resp = await send_with_retry(
        model,
        _request(),
        on_text=chunks.append,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
    )
    assert resp.message.text == "hi"
    assert chunks == list("hi")
    assert model.stream_calls == 1
    assert model.buffer_calls == 0


@pytest.mark.asyncio
async def test_send_with_retry_buffered_path_no_on_text() -> None:
    """``on_text=None`` always uses buffer; never streams."""
    model = _ScriptedModel(buffer_responses=[_resp("ok")])
    resp = await send_with_retry(
        model,
        _request(),
        on_text=None,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
    )
    assert resp.message.text == "ok"
    assert model.stream_calls == 0
    assert model.buffer_calls == 1


@pytest.mark.asyncio
async def test_send_with_retry_retries_on_retryable_error() -> None:
    err = _HTTPError(_FakeResponse(503))
    model = _ScriptedModel(stream_responses=[err, _resp("recovered")])
    notes: list[str] = []
    resp = await send_with_retry(
        model,
        _request(),
        on_text=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=notes.append,
    )
    assert resp.message.text == "recovered"
    assert model.stream_calls == 2
    assert any("retry attempt" in n for n in notes)


@pytest.mark.asyncio
async def test_send_with_retry_raises_on_fatal_error() -> None:
    err = _HTTPError(_FakeResponse(400))
    model = _ScriptedModel(stream_responses=[err])
    with pytest.raises(_HTTPError):
        _ = await send_with_retry(
            model,
            _request(),
            on_text=_silent,
            max_attempts=3,
            persistent_retry=False,
            publish_recoverable=_silent,
        )


@pytest.mark.asyncio
async def test_send_with_retry_context_overflow_propagates() -> None:
    err = RuntimeError("overflow")
    model = _ScriptedModel(stream_responses=[err], is_overflow=True)
    with pytest.raises(RuntimeError):
        _ = await send_with_retry(
            model,
            _request(),
            on_text=_silent,
            max_attempts=3,
            persistent_retry=False,
            publish_recoverable=_silent,
        )


@pytest.mark.asyncio
async def test_send_with_retry_retries_exhausted() -> None:
    err = _HTTPError(_FakeResponse(503))
    model = _ScriptedModel(stream_responses=[err, err, err])
    with pytest.raises(RetriesExhaustedError):
        _ = await send_with_retry(
            model,
            _request(),
            on_text=_silent,
            max_attempts=2,
            persistent_retry=False,
            publish_recoverable=_silent,
        )


@pytest.mark.asyncio
async def test_send_with_retry_429_raises_rate_limit_when_not_persistent() -> None:
    err = _HTTPError(_FakeResponse(429, {"retry-after": "5"}))
    model = _ScriptedModel(stream_responses=[err])
    with pytest.raises(RateLimitError):
        _ = await send_with_retry(
            model,
            _request(),
            on_text=_silent,
            max_attempts=3,
            persistent_retry=False,
            publish_recoverable=_silent,
        )


@dataclass(slots=True, kw_only=True)
class _PersistentModel(_ScriptedModel):
    """Variant of ``_ScriptedModel`` that opts into persistent retry."""

    supports_persistent_retry: bool = True


@pytest.mark.asyncio
async def test_send_with_retry_persistent_loops_on_429() -> None:
    """In persistent mode, 429s loop until success."""
    err = _HTTPError(_FakeResponse(429))
    model = _PersistentModel(
        stream_responses=[err, _resp("ok")],
    )
    resp = await send_with_retry(
        model,
        _request(),
        on_text=_silent,
        max_attempts=3,
        persistent_retry=True,
        publish_recoverable=_silent,
    )
    assert resp.message.text == "ok"


@pytest.mark.asyncio
async def test_send_with_retry_stream_interruption_retries() -> None:
    partial = _resp("partial")
    model = _ScriptedModel(
        stream_responses=[
            StreamInterruptedError(partial),
            _resp("done"),
        ],
    )
    resp = await send_with_retry(
        model,
        _request(),
        on_text=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
    )
    assert resp.message.text == "done"


@pytest.mark.asyncio
async def test_send_with_retry_stream_interruption_returns_partial_after_cap() -> None:
    partial = _resp("partial")
    err = StreamInterruptedError(partial)
    model = _ScriptedModel(stream_responses=[err, err, err])
    resp = await send_with_retry(
        model,
        _request(),
        on_text=_silent,
        max_attempts=5,
        persistent_retry=False,
        publish_recoverable=_silent,
    )
    assert resp is partial


@pytest.mark.asyncio
async def test_send_with_retry_stream_interrupt_on_discarded_response_called() -> None:
    partial = _resp("partial")
    model = _ScriptedModel(
        stream_responses=[
            StreamInterruptedError(partial),
            _resp("done"),
        ],
    )
    discarded: list[ModelResponse] = []
    resp = await send_with_retry(
        model,
        _request(),
        on_text=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
        on_discarded_response=discarded.append,
    )
    assert resp.message.text == "done"
    assert discarded == [partial]


@pytest.mark.asyncio
async def test_send_with_retry_falls_back_to_buffer_after_two_stream_failures() -> None:
    err = _HTTPError(_FakeResponse(503))
    model = _ScriptedModel(
        stream_responses=[err, err],
        buffer_responses=[_resp("buffered ok")],
    )
    chunks: list[str] = []
    resp = await send_with_retry(
        model,
        _request(),
        on_text=chunks.append,
        max_attempts=5,
        persistent_retry=False,
        publish_recoverable=_silent,
    )
    assert resp.message.text == "buffered ok"
    assert model.buffer_calls == 1
    # The buffered text is re-emitted on the live ``on_text`` callback
    # so the renderer sees the response.
    assert "buffered ok" in "".join(chunks)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
