"""Tests for ``agent.retry``: classification, backoff, send-with-retry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, cast, override

import asyncio
import time

import httpx
import pytest

from sagent.agent.retry import (
    _MAX_SERVER_RETRY_AFTER_SEC,
    DEFAULT_MAX_PERSISTENT_ATTEMPTS,
    INTERACTIVE_MAX_SLEEP_SEC,
    MAX_RETRY_DELAY,
    PERSISTENT_MAX_BACKOFF_SEC,
    RETRY_BASE_SEC,
    RETRYABLE_STATUS_CODES,
    RateLimitError,
    RetriesExhaustedError,
    _backoff_delay,
    error_diagnostics,
    error_status,
    extract_retry_after,
    is_rate_limited,
    is_retryable,
    send_with_retry,
    service_error_snapshot,
)
from sagent.testing import MockModelCaps
from sagent.types.model import (
    Model,
    ModelRequest,
    ModelResponse,
    RequestTooLargeError,
    StreamInterruptedError,
)
from sagent.types.runtime import (
    AssistantMessage,
    ModelResponsePartial,
    RuntimeEvent,
)


def test_backoff_delay_keeps_jitter_visible_at_the_cap() -> None:
    """Late attempts must still vary -- jitter survives the clamp.

    The prior ``min(base + jitter, cap)`` shape collapsed every attempt whose
    raw base exceeded the cap to exactly ``cap`` (jitter invisible), which
    re-synchronizes concurrent retriers into a thundering herd. Clamping the
    base BEFORE jitter keeps delays de-correlated at the ceiling.
    """
    # attempt=20 -> raw base 0.5 * 2**20 >> the cap, so every sample is at the
    # ceiling band; they must NOT all be identical.
    samples = {_backoff_delay(20, cap=PERSISTENT_MAX_BACKOFF_SEC) for _ in range(50)}
    assert len(samples) > 1, "jitter collapsed at the cap"
    # Bounded to [0.75 * cap, cap]: cap is a HARD ceiling, jitter subtracts.
    assert all(
        0.75 * PERSISTENT_MAX_BACKOFF_SEC <= s <= PERSISTENT_MAX_BACKOFF_SEC
        for s in samples
    )


def test_backoff_delay_never_exceeds_cap() -> None:
    """``cap`` is a hard ceiling -- the interactive limit must never be breached."""
    assert all(
        _backoff_delay(n, cap=INTERACTIVE_MAX_SLEEP_SEC) <= INTERACTIVE_MAX_SLEEP_SEC
        for n in range(30)
        for _ in range(5)
    )


def test_backoff_delay_below_cap_is_exponential() -> None:
    """Early attempts ramp exponentially from ``RETRY_BASE_SEC``."""
    # attempt=0: base = RETRY_BASE_SEC, jitter subtracts <= 25%.
    d0 = _backoff_delay(0, cap=MAX_RETRY_DELAY)
    assert 0.75 * RETRY_BASE_SEC <= d0 <= RETRY_BASE_SEC


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
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        del request
        self.stream_calls += 1
        item = self.stream_responses[self._stream_idx]
        self._stream_idx += 1
        if isinstance(item, BaseException):
            raise item
        if publish is not None and item.message.text:
            for ch in item.message.text:
                publish(ModelResponsePartial(ch))
        return item

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        return await self.stream(request, publish=None)


def _request() -> ModelRequest:
    return ModelRequest(messages=[])


def _resp(text: str = "ok") -> ModelResponse:
    return ModelResponse(message=AssistantMessage(text=text))


def _silent(_arg: object) -> None:
    return None


def _collect_text(chunks: list[str]) -> Callable[[RuntimeEvent], None]:
    """Build a ``publish`` sink that appends streamed text to ``chunks``."""

    def _sink(event: RuntimeEvent) -> None:
        if isinstance(event, ModelResponsePartial):
            chunks.append(event.text)

    return _sink


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


class _InBandRateLimitError(Exception):
    """Anthropic in-band body-typed error on a 200 stream.

    Mirrors the transcript shape (session 6efa990c, repl.log:38860): the
    SDK raises ``APIStatusError`` with a body-declared error type (e.g.
    ``rate_limit_error``) while the HTTP status is 200 and every
    unified-ratelimit header reports ``allowed`` -- so neither the status
    path nor ``extract_retry_after`` sees anything wrong.
    """

    def __init__(self, error_type: str = "rate_limit_error") -> None:
        super().__init__(f"{error_type} in-band")
        self.body = {"type": "error", "error": {"type": error_type}}
        self.response = _FakeResponse(
            200,
            {
                "anthropic-ratelimit-unified-status": "allowed",
                "anthropic-ratelimit-unified-5h-status": "allowed",
                "anthropic-ratelimit-unified-reset": str(time.time() + 15_000.0),
            },
        )


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


def test_is_retryable_request_too_large_is_fatal() -> None:
    """A 413 byte-overflow must NOT retry transiently.

    It is handled by the agent's overflow-recovery loop (compaction),
    not by the transient-retry path; retrying the identical oversized
    request would just burn the retry budget on the same 413.
    """
    err = RequestTooLargeError("Request exceeds the maximum size")
    err.__cause__ = _HTTPError(_FakeResponse(413))
    assert is_retryable(err, _ScriptedModel()) is False


def test_is_retryable_request_too_large_fatal_even_with_5xx_cause() -> None:
    """``RequestTooLargeError`` is fatal regardless of its cause's status.

    The byte ceiling is not relieved by retrying. A CDN/front-end can wrap
    the byte rejection in a 5xx; the classifier must treat the typed error
    as fatal by TYPE, not walk the cause chain and flip retryable on the
    5xx (which would burn the retry budget on an unfixable request).
    """
    err = RequestTooLargeError("request entity too large")
    err.__cause__ = _HTTPError(_FakeResponse(503))
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


def test_is_rate_limited_429_status() -> None:
    assert is_rate_limited(_HTTPError(_FakeResponse(429))) is True


def test_is_rate_limited_in_band_body() -> None:
    assert is_rate_limited(_InBandRateLimitError()) is True


def test_is_rate_limited_other_errors_false() -> None:
    assert is_rate_limited(_HTTPError(_FakeResponse(503))) is False
    assert is_rate_limited(ValueError("plain")) is False


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


def test_extract_retry_after_absolute_epoch_treated_as_timestamp() -> None:
    # Some servers/proxies send ``retry-after`` as an absolute Unix
    # timestamp rather than RFC delta-seconds. Using it raw as a delay
    # yields a ~56-year suspension; it must be converted to a delta.
    reset = time.time() + 45.0
    err = _HTTPError(_FakeResponse(429, {"retry-after": str(int(reset))}))
    delay = extract_retry_after(err)
    assert delay is not None
    assert 40.0 <= delay <= 46.0


def test_extract_retry_after_structured_ms_attribute() -> None:
    """A ``retry_after_ms`` attribute (CLI providers with no HTTP response)
    is honored, converted ms -> sec, ahead of the response path.
    """

    class _CliRetryableError(Exception):
        retry_after_ms = 30000.0

    assert extract_retry_after(_CliRetryableError()) == pytest.approx(30.0)


def test_extract_retry_after_structured_ms_none_falls_through() -> None:
    """A ``None`` / absent ``retry_after_ms`` does not short-circuit the
    response path.
    """

    class _CliRetryableError(Exception):
        retry_after_ms = None
        response = _FakeResponse(429, {"retry-after": "7"})

    assert extract_retry_after(_CliRetryableError()) == pytest.approx(7.0)


def test_extract_retry_after_far_future_epoch_does_not_explode() -> None:
    # Regression for the "retrying in 20602d" status-pane bug: a retry-after
    # equal to the current epoch must not become a multi-decade delay.
    err = _HTTPError(_FakeResponse(429, {"retry-after": str(int(time.time()))}))
    delay = extract_retry_after(err)
    assert delay is not None
    assert delay < 60.0


def test_extract_retry_after_anthropic_unified_reset() -> None:
    reset = time.time() + 30.0
    err = _HTTPError(
        _FakeResponse(429, {"anthropic-ratelimit-unified-reset": str(reset)})
    )
    delay = extract_retry_after(err)
    assert delay is not None
    assert 25.0 <= delay <= 30.5


def test_extract_retry_after_unified_reset_ignored_when_allowed() -> None:
    # Incident repro (repl.log line 554): a transient in-band rate_limit_error
    # on a 200 stream carries unified-status=allowed plus the always-present
    # unified-reset (= next window rollover, hours away). It must NOT be read
    # as a retry-after, or the call sleeps for hours on a non-limited request.
    reset = time.time() + 15_000.0
    err = _HTTPError(
        _FakeResponse(
            200,
            {
                "anthropic-ratelimit-unified-status": "allowed",
                "anthropic-ratelimit-unified-5h-status": "allowed",
                "anthropic-ratelimit-unified-overage-status": "rejected",
                "anthropic-ratelimit-unified-reset": str(reset),
            },
        )
    )
    assert extract_retry_after(err) is None


def test_extract_retry_after_unified_reset_honored_when_rejected() -> None:
    # When the request actually hit the limit (status header says rejected),
    # the unified-reset clock IS a real retry-after.
    reset = time.time() + 30.0
    err = _HTTPError(
        _FakeResponse(
            200,
            {
                "anthropic-ratelimit-unified-status": "rejected",
                "anthropic-ratelimit-unified-reset": str(reset),
            },
        )
    )
    delay = extract_retry_after(err)
    assert delay is not None
    assert 25.0 <= delay <= 30.5


def test_extract_retry_after_unified_reset_honored_on_429_without_status() -> None:
    # A bare 429 is itself proof of rejection; honor the reset even when no
    # unified status header is present.
    reset = time.time() + 30.0
    err = _HTTPError(
        _FakeResponse(429, {"anthropic-ratelimit-unified-reset": str(reset)})
    )
    delay = extract_retry_after(err)
    assert delay is not None
    assert 25.0 <= delay <= 30.5


def test_extract_retry_after_unified_warning_status_not_a_limit() -> None:
    # Captured incident (Issue#316, repl.log:3822): a 200 stream carrying an
    # in-band rate_limit_error whose representative window is at
    # ``allowed_warning`` (a heads-up, NOT a block). The always-present
    # unified-reset is the 7d window rollover (~24h). Treating
    # ``allowed_warning`` as a rejection promoted the warning into a 24h halt.
    # Only ``rejected`` / ``rate_limited`` constitute a real throttle.
    reset = time.time() + 86_000.0
    err = _HTTPError(
        _FakeResponse(
            200,
            {
                "anthropic-ratelimit-unified-status": "allowed_warning",
                "anthropic-ratelimit-unified-5h-status": "allowed",
                "anthropic-ratelimit-unified-7d-status": "allowed_warning",
                "anthropic-ratelimit-unified-7d-surpassed-threshold": "0.75",
                "anthropic-ratelimit-unified-representative-claim": "seven_day",
                "anthropic-ratelimit-unified-reset": str(reset),
            },
        )
    )
    assert extract_retry_after(err) is None


def test_extract_retry_after_unified_rate_limited_status_honored() -> None:
    # ``rate_limited`` is a genuine block (alongside ``rejected``); honor the
    # reset clock as a real retry-after.
    reset = time.time() + 30.0
    err = _HTTPError(
        _FakeResponse(
            200,
            {
                "anthropic-ratelimit-unified-status": "rate_limited",
                "anthropic-ratelimit-unified-reset": str(reset),
            },
        )
    )
    delay = extract_retry_after(err)
    assert delay is not None
    assert 25.0 <= delay <= 30.5


def test_extract_retry_after_google_retry_info_body() -> None:
    # Gemini advertises its 429 backoff in the JSON body via a
    # ``google.rpc.RetryInfo`` detail (``retryDelay: "16s"``), NOT a header.
    # Honor it so we wait the server-stated delay instead of guessing with
    # local backoff (Issue#316; gemini-cli #5119 retries the same request
    # uselessly when this is ignored).
    body = (
        '{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","details":['
        '{"@type":"type.googleapis.com/google.rpc.RetryInfo",'
        '"retryDelay":"16s"}]}}'
    )
    err = _HTTPError(_FakeResponse(429, text=body))
    assert extract_retry_after(err) == pytest.approx(16.0)


def test_extract_retry_after_google_retry_info_fractional_seconds() -> None:
    # ``retryDelay`` may carry fractional seconds (e.g. ``"7.5s"``).
    body = (
        '{"error":{"details":['
        '{"@type":"type.googleapis.com/google.rpc.RetryInfo",'
        '"retryDelay":"7.5s"}]}}'
    )
    err = _HTTPError(_FakeResponse(429, text=body))
    assert extract_retry_after(err) == pytest.approx(7.5)


def test_extract_retry_after_header_precedes_google_body() -> None:
    # A ``retry-after`` header (if present) wins over the body delay; the
    # body is only a fallback for providers that omit the header.
    body = (
        '{"error":{"details":['
        '{"@type":"type.googleapis.com/google.rpc.RetryInfo",'
        '"retryDelay":"16s"}]}}'
    )
    err = _HTTPError(_FakeResponse(429, {"retry-after": "3"}, text=body))
    assert extract_retry_after(err) == pytest.approx(3.0)


def test_extract_retry_after_non_google_body_ignored() -> None:
    # An arbitrary JSON body without a RetryInfo detail yields no delay.
    err = _HTTPError(_FakeResponse(429, text='{"error":{"message":"nope"}}'))
    assert extract_retry_after(err) is None


def test_extract_retry_after_invalid_unified_reset_returns_none() -> None:
    err = _HTTPError(
        _FakeResponse(429, {"anthropic-ratelimit-unified-reset": "not-a-time"})
    )
    assert extract_retry_after(err) is None


def test_extract_retry_after_no_relevant_headers() -> None:
    err = _HTTPError(_FakeResponse(503, {"x-other": "v"}))
    assert extract_retry_after(err) is None


def test_extract_retry_after_handles_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """RFC 7231 allows ``Retry-After`` as an HTTP-date, not just delta-seconds."""
    monkeypatch.setattr(time, "time", lambda: 4_102_444_800.0)  # 2100-01-01T00:00:00Z
    err = _HTTPError(
        _FakeResponse(429, {"retry-after": "Fri, 01 Jan 2100 00:01:00 GMT"})
    )
    delay = extract_retry_after(err)
    assert delay == pytest.approx(60.0)


def test_extract_retry_after_handles_capitalized_header() -> None:
    """SDKs sometimes pass capitalized header keys; lookup must be case-insensitive."""
    err = _HTTPError(_FakeResponse(429, {"Retry-After": "7"}))
    assert extract_retry_after(err) == pytest.approx(7.0)


def test_extract_retry_after_clamps_anthropic_reset_to_24h(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misconfigured reset advertising 30 days must not wedge retry for weeks."""
    now = 1_700_000_000.0
    monkeypatch.setattr(time, "time", lambda: now)
    far_future = now + 30 * 24 * 60 * 60
    err = _HTTPError(
        _FakeResponse(429, {"anthropic-ratelimit-unified-reset": str(far_future)})
    )
    delay = extract_retry_after(err)
    assert delay == pytest.approx(24 * 60 * 60)


def test_extract_retry_after_clamps_seconds_form() -> None:
    """Seconds-form ``retry-after`` is bounded by ``_MAX_SERVER_RETRY_AFTER_SEC``."""
    err = _HTTPError(_FakeResponse(429, {"retry-after": "999999999"}))
    delay = extract_retry_after(err)
    assert delay == _MAX_SERVER_RETRY_AFTER_SEC


def test_extract_retry_after_large_delta_clamps_not_zeroed() -> None:
    # A legal RFC-7231 delta-seconds at/above the 1-year epoch threshold must
    # clamp to _MAX_SERVER_RETRY_AFTER_SEC, not be misread as an absolute epoch
    # and collapse to 0.0 -- which would silently drop the server's backoff.
    err = _HTTPError(_FakeResponse(429, {"retry-after": str(365 * 24 * 60 * 60)}))
    assert extract_retry_after(err) == _MAX_SERVER_RETRY_AFTER_SEC


def test_extract_retry_after_clamps_http_date_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP-date form is bounded by ``_MAX_SERVER_RETRY_AFTER_SEC``."""
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)
    err = _HTTPError(
        _FakeResponse(429, {"retry-after": "Wed, 21 Oct 2099 07:28:00 GMT"})
    )
    delay = extract_retry_after(err)
    assert delay is not None
    assert delay <= _MAX_SERVER_RETRY_AFTER_SEC


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
        publish=_collect_text(chunks),
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
        publish=None,
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
        publish=_silent,
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
            publish=_silent,
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
            publish=_silent,
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
            publish=_silent,
            max_attempts=2,
            persistent_retry=False,
            publish_recoverable=_silent,
        )


@pytest.mark.asyncio
async def test_send_with_retry_429_long_reset_halts_when_not_persistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 429 advertising a reset beyond the interactive ceiling halts
    # immediately with the reset clock; sleeping through it would wedge
    # the REPL for hours.
    slept: list[float] = []

    async def fake_sleep(delay_sec: float) -> None:
        slept.append(delay_sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    err = _HTTPError(_FakeResponse(429, {"retry-after": "15000"}))
    model = _ScriptedModel(stream_responses=[err])
    with pytest.raises(RateLimitError):
        _ = await send_with_retry(
            model,
            _request(),
            publish=_silent,
            max_attempts=3,
            persistent_retry=False,
            publish_recoverable=_silent,
        )
    assert slept == []


@pytest.mark.asyncio
async def test_send_with_retry_429_short_retry_after_retries_to_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Issue#424: a 429 with a short advertised wait is a transient throttle,
    # not a quota halt -- honor the wait and retry instead of halting the
    # REPL over a delay the user would happily sit out.
    slept: list[float] = []

    async def fake_sleep(delay_sec: float) -> None:
        slept.append(delay_sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    err = _HTTPError(_FakeResponse(429, {"retry-after": "5"}))
    model = _ScriptedModel(stream_responses=[err, _resp("ok")])
    resp = await send_with_retry(
        model,
        _request(),
        publish=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
    )
    assert resp.message.text == "ok"
    assert slept == [pytest.approx(5.0)]


@pytest.mark.asyncio
async def test_send_with_retry_exhaustion_skips_final_backoff_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Issue#424 bug #1: the loop slept the final backoff ("resumes in 9s"),
    # then re-entered, hit the attempt cap, and raised WITHOUT sending --
    # the user waits out the banner and still gets the error. Exhaustion
    # must raise instead of sleeping a wait no send will follow.
    slept: list[float] = []

    async def fake_sleep(delay_sec: float) -> None:
        slept.append(delay_sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    err = _HTTPError(_FakeResponse(503))
    model = _ScriptedModel(stream_responses=[err, err])
    with pytest.raises(RetriesExhaustedError):
        _ = await send_with_retry(
            model,
            _request(),
            publish=_silent,
            max_attempts=2,
            persistent_retry=False,
            publish_recoverable=_silent,
        )
    assert model.stream_calls == 2
    assert len(slept) == 1, "no sleep after the final failed attempt"


@pytest.mark.asyncio
async def test_send_with_retry_in_band_rate_limit_outlasts_attempt_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Issue#424 bug #2: an in-band rate_limit_error (200 stream, headers all
    # ``allowed``) clears in tens of seconds, but the generic transient
    # budget (5 attempts, ~9s effective wait) gives up first. Rate-limit
    # retries run on a wall-clock budget, not the attempt counter.
    async def fake_sleep(delay_sec: float) -> None:
        del delay_sec

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    errors: list[BaseException | ModelResponse] = [
        _InBandRateLimitError() for _ in range(7)
    ]
    model = _ScriptedModel(
        stream_responses=[*errors, _resp("ok")], is_retryable_provider=True
    )
    notes: list[str] = []
    resp = await send_with_retry(
        model,
        _request(),
        publish=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=notes.append,
    )
    assert resp.message.text == "ok"
    assert model.stream_calls == 8
    assert notes[0].startswith("retry attempt 0"), "labels count throttle retries"
    assert notes[6].startswith("retry attempt 6")


@pytest.mark.asyncio
async def test_send_with_retry_in_band_rate_limit_halts_as_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When the throttle does NOT clear, the wall-clock budget halts the loop
    # as a RateLimitError ("type to retry"), never RetriesExhaustedError, and
    # no single sleep exceeds the interactive ceiling.
    clock = {"now": 1_000.0}
    slept: list[float] = []

    async def fake_sleep(delay_sec: float) -> None:
        clock["now"] += delay_sec
        slept.append(delay_sec)

    monkeypatch.setattr(time, "time", lambda: clock["now"])
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    model = _ScriptedModel(
        stream_responses=[_InBandRateLimitError() for _ in range(50)],
        is_retryable_provider=True,
    )
    with pytest.raises(RateLimitError):
        _ = await send_with_retry(
            model,
            _request(),
            publish=_silent,
            max_attempts=5,
            persistent_retry=False,
            publish_recoverable=_silent,
        )
    assert all(s <= INTERACTIVE_MAX_SLEEP_SEC for s in slept)
    assert sum(slept) > 60.0, "budget must outlast the old ~9s attempt cap"
    assert sum(slept) < 300.0, "budget must still halt within minutes"


@pytest.mark.asyncio
async def test_send_with_retry_rate_limit_backstop_survives_frozen_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The throttle loop's normal exit is the wall-clock budget; a frozen
    # clock (mocked time, fake event loop) never reaches it. The attempt
    # backstop must halt the loop as a RateLimitError instead of spinning.
    monkeypatch.setattr(time, "time", lambda: 1_000.0)

    async def fake_sleep(delay_sec: float) -> None:
        del delay_sec

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    model = _ScriptedModel(
        stream_responses=[_InBandRateLimitError() for _ in range(50)],
        is_retryable_provider=True,
    )
    with pytest.raises(RateLimitError):
        _ = await send_with_retry(
            model,
            _request(),
            publish=_silent,
            max_attempts=5,
            persistent_retry=False,
            publish_recoverable=_silent,
        )
    assert model.stream_calls < 50


@pytest.mark.asyncio
async def test_send_with_retry_interactive_halts_on_long_server_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A retryable, non-429 error (e.g. in-band rate_limit_error on a 200
    # stream) that carries a multi-hour server backoff must NOT become a
    # blocking sleep in interactive mode -- it halts as a RateLimitError.
    slept: list[float] = []

    async def fake_sleep(d: float) -> None:
        slept.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    err = _HTTPError(
        _FakeResponse(
            200,
            {
                "anthropic-ratelimit-unified-status": "rejected",
                "anthropic-ratelimit-unified-reset": str(time.time() + 15_000.0),
            },
        )
    )
    model = _ScriptedModel(stream_responses=[err], is_retryable_provider=True)
    with pytest.raises(RateLimitError):
        _ = await send_with_retry(
            model,
            _request(),
            publish=_silent,
            max_attempts=3,
            persistent_retry=False,
            publish_recoverable=_silent,
        )
    assert slept == [], "interactive mode must not sleep on a long server delay"


@pytest.mark.asyncio
async def test_send_with_retry_in_band_warning_retries_to_success() -> None:
    # Issue#316 bug #3: an in-band rate_limit_error on a 200 stream whose
    # representative window is ``allowed_warning`` (a heads-up, not a block)
    # must NOT halt -- it is a transient retryable, so the next attempt
    # succeeds. Before the gate fix this raised RateLimitError (24h halt).
    err = _HTTPError(
        _FakeResponse(
            200,
            {
                "anthropic-ratelimit-unified-status": "allowed_warning",
                "anthropic-ratelimit-unified-7d-status": "allowed_warning",
                "anthropic-ratelimit-unified-reset": str(time.time() + 86_000.0),
            },
        )
    )
    model = _ScriptedModel(
        stream_responses=[err, _resp("ok")], is_retryable_provider=True
    )
    response = await send_with_retry(
        model,
        _request(),
        publish=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
    )
    assert response.message.text == "ok"
    assert model.stream_calls == 2


@pytest.mark.asyncio
async def test_send_with_retry_429_with_unread_streaming_body_still_classified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # End-to-end regression for the transcript bug: a 429 whose response is
    # an unread streaming body (`.text`/`.content` raise ResponseNotRead)
    # must still be classified as a rate limit -- the diagnostics body-read
    # must not let ResponseNotRead escape and turn a retryable error fatal.
    # Under the Issue#424 policy the short advertised wait (5s) is honored
    # and the send retries instead of halting.
    slept: list[float] = []

    async def fake_sleep(delay_sec: float) -> None:
        slept.append(delay_sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    err = _HTTPError(cast("_FakeResponse", _UnreadStreamingResponse()))
    assert is_rate_limited(err) is True
    model = _ScriptedModel(stream_responses=[err, _resp("ok")])
    resp = await send_with_retry(
        model,
        _request(),
        publish=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
    )
    assert resp.message.text == "ok"
    assert slept == [pytest.approx(5.0)]


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
        publish=_silent,
        max_attempts=3,
        persistent_retry=True,
        publish_recoverable=_silent,
    )
    assert resp.message.text == "ok"


@pytest.mark.asyncio
async def test_send_with_retry_persistent_loops_on_in_band_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Issue#424: an in-band rate_limit_error is the same condition as a 429
    # wearing a 200 status; persistent (unattended) mode must route it into
    # the long-backoff branch instead of the 5-attempt transient loop.
    async def fake_sleep(delay_sec: float) -> None:
        del delay_sec

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    errors: list[BaseException | ModelResponse] = [
        _InBandRateLimitError() for _ in range(5)
    ]
    model = _PersistentModel(
        stream_responses=[*errors, _resp("ok")], is_retryable_provider=True
    )
    resp = await send_with_retry(
        model,
        _request(),
        publish=_silent,
        max_attempts=3,
        persistent_retry=True,
        publish_recoverable=_silent,
    )
    assert resp.message.text == "ok"
    assert model.stream_calls == 6


@pytest.mark.asyncio
async def test_send_with_retry_persistent_loops_on_in_band_overload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 529's body type is ``overloaded_error``; in-band it rides a 200
    # stream, so the ``status == 529`` gate alone misses it and unattended
    # compaction would die in the generic 5-attempt loop.
    async def fake_sleep(delay_sec: float) -> None:
        del delay_sec

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    errors: list[BaseException | ModelResponse] = [
        _InBandRateLimitError("overloaded_error") for _ in range(5)
    ]
    model = _PersistentModel(
        stream_responses=[*errors, _resp("ok")], is_retryable_provider=True
    )
    resp = await send_with_retry(
        model,
        _request(),
        publish=_silent,
        max_attempts=3,
        persistent_retry=True,
        publish_recoverable=_silent,
    )
    assert resp.message.text == "ok"
    assert model.stream_calls == 6


@pytest.mark.asyncio
async def test_send_with_retry_persistent_sleeps_through_long_server_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The interactive long-reset halt is gated on ``not persistent``:
    # unattended mode must sleep out a long advertised reset, not halt --
    # there is no user at the REPL to hand the RateLimitError to.
    slept: list[float] = []

    async def fake_sleep(delay_sec: float) -> None:
        slept.append(delay_sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    err = _HTTPError(_FakeResponse(429, {"retry-after": "120"}))
    model = _PersistentModel(stream_responses=[err, _resp("ok")])
    resp = await send_with_retry(
        model,
        _request(),
        publish=_silent,
        max_attempts=3,
        persistent_retry=True,
        publish_recoverable=_silent,
    )
    assert resp.message.text == "ok"
    assert slept == [pytest.approx(120.0)]


@pytest.mark.asyncio
async def test_send_with_retry_persistent_loops_through_remote_protocol_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent mode treats ``httpx.RemoteProtocolError`` as long-backoff.

    Bug repro: large compaction payloads cause OpenAI to close the
    streamed request body mid-flight. The error has no HTTP status, so
    the normal 32s-cap loop bails out after a handful of attempts and
    the compactor fails. With ``persistent_retry=True`` these protocol
    flakes should now stay in the persistent-attempt branch indefinitely.
    """
    sleeps: list[float] = []

    async def fake_sleep(delay_sec: float) -> None:
        sleeps.append(delay_sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    err = httpx.RemoteProtocolError("peer closed connection")
    model = _PersistentModel(
        stream_responses=[err, err, err, _resp("ok")],
    )
    resp = await send_with_retry(
        model,
        _request(),
        publish=_silent,
        max_attempts=3,
        persistent_retry=True,
        publish_recoverable=_silent,
    )
    assert resp.message.text == "ok"
    # 3 persistent attempts means 3 sleeps; without the fix the
    # short-backoff loop would have raised RetriesExhaustedError.
    assert len(sleeps) == 3


@pytest.mark.asyncio
async def test_send_with_retry_persistent_attempts_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent retry honors ``max_persistent_attempts`` and fails loud.

    Without the cap, a permanently-429'ing server wedges the loop
    indefinitely; only ``CancelledError`` can interrupt.
    """
    sleeps: list[float] = []

    async def fake_sleep(delay_sec: float) -> None:
        sleeps.append(delay_sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    err = _HTTPError(_FakeResponse(429))
    model = _PersistentModel(stream_responses=[err] * 20)
    with pytest.raises(RetriesExhaustedError):
        _ = await send_with_retry(
            model,
            _request(),
            publish=_silent,
            max_attempts=1_000,
            persistent_retry=True,
            publish_recoverable=_silent,
            max_persistent_attempts=3,
        )
    assert model.stream_calls == 3


def test_default_max_persistent_attempts_finite() -> None:
    """The default persistent cap must be bounded; infinite is a wedge."""
    assert 0 < DEFAULT_MAX_PERSISTENT_ATTEMPTS < 10_000


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
        publish=_collect_text(chunks),
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
        publish=_silent,
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
        publish=_silent,
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
            publish: Callable[[RuntimeEvent], None] | None = None,
        ) -> ModelResponse:
            del request
            self.stream_calls += 1
            if self.stream_calls == 1:
                if publish is not None:
                    publish(ModelResponsePartial("abc"))
                raise StreamInterruptedError(_resp("abc"))
            if publish is not None:
                publish(ModelResponsePartial("xyz"))
            return _resp("xyz")

    chunks: list[str] = []
    model = DivergentRetryModel(stream_responses=[])

    resp = await send_with_retry(
        model,
        _request(),
        publish=_collect_text(chunks),
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
    )

    assert resp.message.text == "xyz"
    assert "".join(chunks) == (
        "abc\n[retry diverged; discard the text above -- "
        "the corrected response follows]\nxyz"
    )


@pytest.mark.asyncio
async def test_send_with_retry_stream_interruption_returns_partial_after_cap() -> None:
    partial = _resp("partial")
    err = StreamInterruptedError(partial)
    model = _ScriptedModel(stream_responses=[err, err, err])
    resp = await send_with_retry(
        model,
        _request(),
        publish=_silent,
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
        publish=_silent,
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


class _UnreadStreamingResponse:
    """Mimics an httpx streaming response whose body was never read.

    ``.text`` / ``.content`` raise ``ResponseNotRead``, reproducing the
    transcript failure where Anthropic reported a ``rate_limit_error`` via
    an in-band SSE ``error`` event on a 200 stream and the SDK attached the
    unread streaming response to the resulting ``APIStatusError``.
    """

    status_code: ClassVar[int] = 429
    headers: ClassVar[dict[str, str]] = {"request-id": "req-1", "retry-after": "5"}

    @property
    def text(self) -> str:
        raise httpx.ResponseNotRead

    @property
    def content(self) -> bytes:
        raise httpx.ResponseNotRead


def test_service_error_snapshot_survives_unread_streaming_body() -> None:
    err = _HTTPError(cast("_FakeResponse", _UnreadStreamingResponse()))
    snapshot = service_error_snapshot(err)
    assert snapshot.status == 429
    assert snapshot.body == ""
    assert snapshot.headers["request-id"] == "req-1"


def test_error_diagnostics_survives_unread_streaming_body() -> None:
    err = _HTTPError(cast("_FakeResponse", _UnreadStreamingResponse()))
    diag = error_diagnostics(err)
    assert "status=429" in diag


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
async def test_send_with_retry_honors_resume_retry_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ScriptedModel(stream_responses=[_resp("ok")])
    sleeps: list[float] = []
    monkeypatch.setattr(time, "time", lambda: 100.0)

    async def fake_sleep(delay_sec: float) -> None:
        sleeps.append(delay_sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    response = await send_with_retry(
        model,
        _request(),
        publish=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
        resume_retry_at=112.5,
    )

    assert response.message.text == "ok"
    assert sleeps == [12.5]
    assert model.stream_calls == 1


@pytest.mark.asyncio
async def test_send_with_retry_resume_retry_at_over_ceiling_does_not_wedge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Issue#316 bug #5: a stale ``resume_retry_at`` carrying a bogus ~24h
    # timestamp (itself produced by the allowed_warning->halt bug) was replayed
    # as an uninterruptible sleep on the next user send. The delay is
    # ``resume_retry_at - now`` -- wall-clock decay, so waiting minutes does not
    # help (repro: +5s -> 86395s, +5m -> 86095s). A resume wait beyond the
    # interactive ceiling must not block the REPL; the send proceeds instead.
    model = _ScriptedModel(stream_responses=[_resp("ok")])
    sleeps: list[float] = []
    monkeypatch.setattr(time, "time", lambda: 100.0)

    async def fake_sleep(delay_sec: float) -> None:
        sleeps.append(delay_sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    response = await send_with_retry(
        model,
        _request(),
        publish=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
        resume_retry_at=100.0 + INTERACTIVE_MAX_SLEEP_SEC + 86_400.0,
    )

    assert response.message.text == "ok"
    assert sleeps == [], "resume wait beyond ceiling must not sleep-wedge"
    assert model.stream_calls == 1


@pytest.mark.asyncio
async def test_send_with_retry_resume_over_ceiling_notifies_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skipping a stale over-ceiling resume wait must NOTIFY, not vanish silently.

    The prior session painted a "service suspended until <T>" banner from the
    persisted ``retry_at``. When that wait is past the interactive ceiling we
    skip it and proceed -- but with no signal the user is left staring at a
    suspension banner the code silently abandoned. Emit a recoverable notice so
    the abandonment is visible.
    """
    model = _ScriptedModel(stream_responses=[_resp("ok")])
    monkeypatch.setattr(time, "time", lambda: 100.0)

    async def fake_sleep(delay_sec: float) -> None:
        del delay_sec

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    notes: list[str] = []

    response = await send_with_retry(
        model,
        _request(),
        publish=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=notes.append,
        resume_retry_at=100.0 + INTERACTIVE_MAX_SLEEP_SEC + 86_400.0,
    )

    assert response.message.text == "ok"
    assert any("resume" in n.lower() for n in notes), (
        f"skipped over-ceiling resume wait must notify the user; got {notes}"
    )


@pytest.mark.asyncio
async def test_send_with_retry_resume_retry_at_under_ceiling_still_sleeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A short resume wait (within the interactive ceiling) is a legitimate
    # backoff and is still honored.
    model = _ScriptedModel(stream_responses=[_resp("ok")])
    sleeps: list[float] = []
    monkeypatch.setattr(time, "time", lambda: 100.0)

    async def fake_sleep(delay_sec: float) -> None:
        sleeps.append(delay_sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    response = await send_with_retry(
        model,
        _request(),
        publish=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
        resume_retry_at=100.0 + INTERACTIVE_MAX_SLEEP_SEC - 5.0,
    )

    assert response.message.text == "ok"
    assert sleeps == [INTERACTIVE_MAX_SLEEP_SEC - 5.0]
    assert model.stream_calls == 1


@pytest.mark.asyncio
async def test_send_with_retry_does_not_emit_banner_into_on_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry waits no longer pollute ``on_text``; suspensions go through callbacks.

    Uses a server-advertised delay so the suspension banner is ``notable``
    (sub-second transport retries are intentionally silent -- see
    ``test_send_with_retry_silent_on_short_transient_retry``).
    """
    err = _HTTPError(_FakeResponse(503, {"retry-after": "1"}))
    model = _ScriptedModel(stream_responses=[err, _resp("ok")])
    chunks: list[str] = []
    suspensions: list[float] = []

    async def fake_sleep(_d: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def _record(
        retry_at: float, delay_sec: float, server_supplied: bool, error: Exception
    ) -> None:
        del delay_sec, server_supplied, error
        suspensions.append(retry_at)

    _ = await send_with_retry(
        model,
        _request(),
        publish=_collect_text(chunks),
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=_silent,
        on_service_suspended=_record,
    )
    assert "retrying" not in "".join(chunks)
    assert "resumes at" not in "".join(chunks)
    assert len(suspensions) == 1


@pytest.mark.asyncio
async def test_send_with_retry_silent_on_short_transient_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sub-second transport blip retries WITHOUT a suspension banner.

    Regression: every recovered httpx ``ReadError`` was publishing a
    ``ModelServiceSuspended`` event that rendered "[model service suspended:
    temporarily blocked; resumes in 1s]" -- alarming the user over a wait
    they never perceive. Local backoffs under ``SUSPENSION_NOTICE_SEC`` with
    no server-advertised delay must stay silent (still logged via
    ``publish_recoverable``).
    """
    err = httpx.ReadError("")
    model = _ScriptedModel(stream_responses=[err, _resp("ok")])
    suspensions: list[float] = []
    notes: list[str] = []

    async def fake_sleep(_d: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    def _record(
        retry_at: float, delay_sec: float, server_supplied: bool, error: Exception
    ) -> None:
        del delay_sec, server_supplied, error
        suspensions.append(retry_at)

    resp = await send_with_retry(
        model,
        _request(),
        publish=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=notes.append,
        on_service_suspended=_record,
    )
    assert resp.message.text == "ok"
    assert suspensions == [], "short transient retry must not publish a suspension"
    assert any(n.startswith("retry attempt") for n in notes), "still logged"


@pytest.mark.asyncio
async def test_publish_recoverable_includes_diagnostics_on_retry() -> None:
    """``publish_recoverable`` payloads append ``[status=... body=...]`` for HTTP errors."""
    err = _HTTPError(_FakeResponse(503, text="upstream gone"))
    model = _ScriptedModel(stream_responses=[err, _resp("ok")])
    notes: list[str] = []
    _ = await send_with_retry(
        model,
        _request(),
        publish=_silent,
        max_attempts=3,
        persistent_retry=False,
        publish_recoverable=notes.append,
    )
    retry_notes = [n for n in notes if n.startswith("retry attempt")]
    assert retry_notes
    assert "status=503" in retry_notes[0]
    assert "upstream gone" in retry_notes[0]


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)


class _EntitlementError(Exception):
    """A 429 whose body says the account lacks an entitlement, not a quota."""

    status_code = 429
    body: ClassVar[Mapping[str, object]] = {
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": "Usage credits are required for fast mode.",
        },
    }


def test_entitlement_429_is_fatal_not_rate_limited() -> None:
    """A credits/entitlement 429 can never clear by waiting.

    A provider returns "Usage credits are required for fast mode." as a
    ``rate_limit_error`` with status 429, indistinguishable from real
    throttling by status alone. Retrying burns the whole budget on a
    request that will never succeed, and the banner says only
    "temporarily blocked" while it does.
    """
    assert is_rate_limited(_EntitlementError()) is False


def test_entitlement_429_is_not_retryable() -> None:
    """The fatal classification must hold on the retry path too.

    ``is_rate_limited`` gates the long-backoff branch, but ``is_retryable``
    decides whether to retry at all -- and it walks the status chain
    independently, so a 429 reads retryable there unless carved out.
    """
    model = _ScriptedModel()
    assert is_retryable(_EntitlementError(), cast("Model", model)) is False


def test_ordinary_429_stays_rate_limited() -> None:
    """The entitlement carve-out must not swallow real throttling."""

    class _ThrottledError(Exception):
        status_code = 429
        body: ClassVar[Mapping[str, object]] = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "rate limit exceeded"},
        }

    assert is_rate_limited(_ThrottledError()) is True


def test_snapshot_prefers_the_body_message_over_the_raw_json() -> None:
    """``str(error)`` on an SDK error is the whole JSON body.

    Rendering that verbatim buried the one useful sentence in a wall of
    braces; the body's own ``message`` is what the user needs to read.
    """
    snap = service_error_snapshot(_EntitlementError())
    assert snap.message == "Usage credits are required for fast mode."


@pytest.mark.parametrize(
    "message",
    [
        "Usage credits are required for fast mode.",
        "Your credit balance is too low to access the API.",
        "You have exceeded your usage credits for this month.",
        "Your org is out of extra usage for the month.",
    ],
)
def test_exhausted_quota_messages_are_fatal(message: str) -> None:
    """Out-of-quota reads as a 429 but no wait refills the account.

    A provider phrases the same fatal condition several ways -- missing
    credits, drained balance, org extra-usage exhausted -- all as
    ``rate_limit_error``. Matching only one wording leaves the others
    retrying against a wall.
    """

    class _ExhaustedError(Exception):
        status_code = 429
        body: ClassVar[Mapping[str, object]] = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": message},
        }

    assert is_rate_limited(_ExhaustedError()) is False
