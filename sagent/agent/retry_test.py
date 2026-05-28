"""Tests for ``agent.retry``: classification, backoff, send-with-retry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import override

import asyncio
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
    error_diagnostics,
    error_status,
    extract_retry_after,
    is_retryable,
    send_with_retry,
    service_error_snapshot,
)
from sagent.testing import MockModelCaps
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    StreamInterruptedError,
)
from sagent.types.runtime import AssistantMessage


@dataclass(slots=True, kw_only=True)
class _ScriptedModel(MockModelCaps):
    """Model with a scripted response queue and optional fault injection."""

    model_id: str = "scripted"
    max_request_tokens: int = 100_000
    stream_responses: list[BaseException | ModelResponse] = field(default_factory=list)
    is_retryable_provider: bool = False
    is_overflow: bool = False
    _stream_idx: int = field(default=0, init=False)
    stream_calls: int = field(default=0, init=False)

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
        return await self.stream(request, on_text=None, on_thinking=None)


def _request() -> ModelRequest:
    return ModelRequest(messages=[])


def _resp(text: str = "ok") -> ModelResponse:
    return ModelResponse(message=AssistantMessage(text=text))


def _silent(_text: str) -> None:
    return None


class _FakeResponse:
    """Minimal stand-in for ``httpx.Response`` carrying status + headers + body."""

    def __init__(
        self,
        status_code: int,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text


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


@pytest.mark.asyncio
async def test_send_with_retry_no_on_text_still_streams() -> None:
    """``on_text=None`` still streams; buffered transport is never used."""
    model = _ScriptedModel(stream_responses=[_resp("ok")])
    resp = await send_with_retry(
        model,
        _request(),
        on_text=None,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
    )
    assert resp.message.text == "ok"
    assert model.stream_calls == 1


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
async def test_send_with_retry_service_suspension_callback_replaces_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    err = _HTTPError(_FakeResponse(429, {"retry-after": "60"}))
    model = _PersistentModel(stream_responses=[err, _resp("ok")])
    chunks: list[str] = []
    suspensions: list[tuple[float, float, bool, Exception]] = []
    sleeps: list[float] = []

    async def fake_sleep(delay_sec: float) -> None:
        sleeps.append(delay_sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    resp = await send_with_retry(
        model,
        _request(),
        on_text=chunks.append,
        max_attempts=3,
        persistent_retry=True,
        publish_recoverable=_silent,
        on_service_suspended=lambda retry_at, delay_sec, server_supplied, error: (
            suspensions.append((retry_at, delay_sec, server_supplied, error))
        ),
    )

    assert resp.message.text == "ok"
    assert "retrying" not in "".join(chunks)
    assert "resumes at" not in "".join(chunks)
    assert len(suspensions) == 1
    retry_at, delay_sec, server_supplied, error = suspensions[0]
    assert retry_at > time.time()
    assert delay_sec == pytest.approx(60.0)
    assert server_supplied is True
    assert error is err
    assert sleeps == [pytest.approx(60.0)]


@pytest.mark.asyncio
async def test_send_with_retry_stream_interruption_retries() -> None:
    partial = _resp("partial")
    model = _ScriptedModel(
        stream_responses=[
            StreamInterruptedError(partial),
            _resp("done"),
        ],
    )
    notes: list[str] = []
    resp = await send_with_retry(
        model,
        _request(),
        on_text=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=notes.append,
    )
    assert resp.message.text == "done"
    assert notes[0].startswith("stream interrupted (attempt 1,")


@pytest.mark.asyncio
async def test_send_with_retry_stream_interruption_retries_count_attempts() -> None:
    first = StreamInterruptedError(_resp("first"))
    second = StreamInterruptedError(_resp("second"))
    model = _ScriptedModel(stream_responses=[first, second, _resp("done")])
    notes: list[str] = []

    resp = await send_with_retry(
        model,
        _request(),
        on_text=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=notes.append,
    )

    assert resp.message.text == "done"
    assert notes[0].startswith("stream interrupted (attempt 1,")
    assert notes[1].startswith("stream interrupted (attempt 2,")


@pytest.mark.asyncio
async def test_send_with_retry_corrects_divergent_retry_output() -> None:
    @dataclass(slots=True, kw_only=True)
    class DivergentRetryModel(_ScriptedModel):
        @override
        async def stream(
            self,
            request: ModelRequest,
            on_text: Callable[[str], None] | None = None,
            on_thinking: Callable[[str], None] | None = None,
        ) -> ModelResponse:
            del request, on_thinking
            self.stream_calls += 1
            if self.stream_calls == 1:
                if on_text is not None:
                    on_text("abc")
                raise StreamInterruptedError(_resp("abc"))
            if on_text is not None:
                on_text("xyz")
            return _resp("xyz")

    chunks: list[str] = []
    model = DivergentRetryModel(stream_responses=[])

    resp = await send_with_retry(
        model,
        _request(),
        on_text=chunks.append,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
    )

    assert resp.message.text == "xyz"
    assert (
        "".join(chunks) == "abc\n[retry response diverged; final response follows]\nxyz"
    )


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


def test_error_diagnostics_includes_status_headers_body() -> None:
    err = _HTTPError(
        _FakeResponse(
            429,
            {
                "retry-after": "14868",
                "anthropic-ratelimit-unified-reset": "1234567890",
                "x-other": "ignored",
            },
            text='{"type":"error","error":{"type":"rate_limit_error"}}',
        )
    )
    diag = error_diagnostics(err)
    assert "status=429" in diag
    assert "retry-after" in diag
    assert "anthropic-ratelimit-unified-reset" in diag
    assert "x-other" not in diag
    assert "rate_limit_error" in diag


def test_service_error_snapshot_allowlists_forensic_fields() -> None:
    err = _HTTPError(
        _FakeResponse(
            429,
            {
                "authorization": "secret",
                "cookie": "secret",
                "request-id": "req-1",
                "retry-after": "14868",
                "x-ratelimit-reset": "soon",
            },
            text="bucket details" + "x" * 10_000,
        )
    )

    snapshot = service_error_snapshot(err)

    assert snapshot.type_name == "_HTTPError"
    assert snapshot.message == "HTTP 429"
    assert snapshot.status == 429
    assert dict(snapshot.headers) == {
        "request-id": "req-1",
        "retry-after": "14868",
        "x-ratelimit-reset": "soon",
    }
    assert snapshot.body.startswith("bucket details")
    assert len(snapshot.body) == 500


def test_error_diagnostics_no_response_returns_empty() -> None:
    assert error_diagnostics(ValueError("plain")) == ""


def test_error_diagnostics_truncates_long_body() -> None:
    err = _HTTPError(_FakeResponse(500, text="A" * 10_000))
    diag = error_diagnostics(err)
    assert len(diag) < 1_000


def test_rate_limit_error_carries_diagnostics() -> None:
    original = _HTTPError(
        _FakeResponse(429, {"retry-after": "14868"}, text='{"weekly":"limit"}')
    )
    e = RateLimitError(time.time() + 14_868, original)
    assert "status=429" in e.diagnostics
    assert "weekly" in e.diagnostics


@pytest.mark.asyncio
async def test_send_with_retry_short_wait_banner_uses_seconds() -> None:
    err = _HTTPError(_FakeResponse(503))
    model = _ScriptedModel(stream_responses=[err, _resp("ok")])
    chunks: list[str] = []
    _ = await send_with_retry(
        model,
        _request(),
        on_text=chunks.append,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
    )
    banner = "".join(c for c in chunks if c.startswith(("\n[", "[")))
    assert "retrying in" in banner
    assert "resumes at" not in banner


@pytest.mark.asyncio
async def test_send_with_retry_long_server_wait_banner_shows_resume_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``retry-after`` exceeds 60s, banner switches to wall-clock format."""
    err = _HTTPError(_FakeResponse(503, {"retry-after": "3600"}))
    model = _PersistentModel(stream_responses=[err, _resp("ok")])
    chunks: list[str] = []
    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    _ = await send_with_retry(
        model,
        _request(),
        on_text=chunks.append,
        max_attempts=3,
        persistent_retry=True,
        publish_recoverable=_silent,
    )
    banner = "".join(c for c in chunks if "[" in c)
    assert "rate-limited" in banner
    assert "resumes at" in banner
    assert sleeps
    assert sleeps[0] >= 3600.0


@pytest.mark.asyncio
async def test_publish_recoverable_includes_diagnostics_on_retry() -> None:
    """``publish_recoverable`` payloads append ``[status=... body=...]`` for HTTP errors."""
    err = _HTTPError(_FakeResponse(503, text="upstream gone"))
    model = _ScriptedModel(stream_responses=[err, _resp("ok")])
    notes: list[str] = []
    _ = await send_with_retry(
        model,
        _request(),
        on_text=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=notes.append,
    )
    retry_notes = [n for n in notes if n.startswith("retry attempt")]
    assert retry_notes
    assert "status=503" in retry_notes[0]
    assert "upstream gone" in retry_notes[0]


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
