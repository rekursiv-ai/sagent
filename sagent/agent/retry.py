"""Retry classification, backoff, and send-with-retry loop.

Decides whether an error is transient (retry) or fatal (propagate),
extracts a server-advertised delay, and surfaces rate limits as
``RateLimitError`` so interactive callers can pretty-print
"resumes at HH:MM:SS".

Constants:

- ``RETRY_BASE_SEC`` - exponential-backoff base (500 ms, doubled each attempt).
- ``MAX_RETRY_DELAY`` - normal-mode backoff cap.
- ``RATE_LIMIT_RETRY_BUDGET_SEC`` - wall-clock budget for rate-limit
  retries without a long advertised reset.
- ``PERSISTENT_MAX_BACKOFF_SEC`` - persistent-mode cap (5 min) for
  unattended 429/529 loops.
- ``RETRYABLE_STATUS_CODES`` - explicit fast-path; anything ≥ 500 also
  retries via the numeric comparison in :func:`is_retryable`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Final, cast

import asyncio
import logging
import random
import re
import time


if TYPE_CHECKING:
    import httpx
else:
    from wrapt import lazy_import

    httpx = lazy_import("httpx")  # 168ms

from sagent.lib.durations import humanize_duration
from sagent.types import runtime as runtime_types
from sagent.types.model import (
    Model,
    ModelRequest,
    ModelResponse,
    RequestTooLargeError,
    StreamInterruptedError,
)


logger = logging.getLogger(__name__)

RETRY_BASE_SEC = 0.5  # config-globals: ignore -- exponential backoff base
MAX_RETRY_DELAY = 32.0  # config-globals: ignore -- normal-mode backoff cap
# Interactive ceiling: in non-persistent mode a server-advertised backoff
# longer than this is surfaced as a ``RateLimitError`` halt rather than a
# blocking sleep, so a multi-hour rate-limit reset can't freeze the REPL.
INTERACTIVE_MAX_SLEEP_SEC = (
    60.0  # config-globals: ignore -- interactive backoff ceiling
)
# Notification threshold for a LOCAL-backoff retry: at or below this the wait
# is short enough to retry silently (a sub-5s transport blip the user never
# perceives); a local backoff longer than this publishes the "model service
# suspended" banner so the user understands the pause. Server-advertised waits
# always notify regardless of length.
SUSPENSION_NOTICE_SEC = (
    5.0  # config-globals: ignore -- suspension-banner notice threshold
)
# A throttle without a long advertised reset clears in seconds-to-minutes
# (Issue#424, session 6efa990c: every halt recovered on an immediate manual
# retry), so rate-limit retries run on this wall-clock budget instead of the
# attempt counter -- the generic ``max_attempts`` cap gives up after ~9s of
# effective waiting, just before the limiter relents.
RATE_LIMIT_RETRY_BUDGET_SEC = (
    180.0  # config-globals: ignore -- rate-limit retry wall-clock budget
)
PERSISTENT_MAX_BACKOFF_SEC = (
    300.0  # config-globals: ignore -- persistent-mode backoff cap
)
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})

# Persistent-retry cap. Without this, a permanently-429'ing server wedges
# the persistent loop indefinitely; only ``CancelledError`` can interrupt.
# At ``PERSISTENT_MAX_BACKOFF_SEC=300`` and exponential ramp-up, 60 attempts
# is roughly 5 hours of wall-clock retry -- long enough that a transient
# upstream outage typically clears, short enough that an unattended job
# eventually fails loud instead of stalling forever.
DEFAULT_MAX_PERSISTENT_ATTEMPTS = (
    60  # config-globals: ignore -- persistent-retry attempt cap
)

# Depth cap when unwinding ``__cause__`` chains - a pathological
# self-referencing cycle must not hang us.
_MAX_CAUSE_DEPTH = 5  # config-globals: ignore -- __cause__ unwind depth cap

# Backstop on throttle retries: termination normally comes from the
# wall-clock budget, which a frozen or mocked clock cannot advance. The
# ramp reaches the budget in ~9 attempts, so a legitimate loop never
# gets here -- mirrors the counter backstops on the other branches.
_MAX_RATE_LIMIT_ATTEMPTS = 20  # config-globals: ignore -- rate-limit attempt backstop

_MAX_STREAM_INTERRUPT_RETRIES = (
    2  # config-globals: ignore -- stream-interrupt retry count
)


class RetriesExhaustedError(Exception):
    """All retry attempts failed."""


class RateLimitError(Exception):
    """Provider rate limit that interactive mode will not sleep through.

    Raised immediately when the server advertises a reset beyond
    ``INTERACTIVE_MAX_SLEEP_SEC``, or once ``RATE_LIMIT_RETRY_BUDGET_SEC``
    of capped-backoff retries fails to outlast the throttle. Carries the
    reset timestamp (seconds since epoch) when the provider advertised
    one via ``retry-after`` or ``anthropic-ratelimit-unified-reset``.
    The message is pre-formatted for REPL display.

    Args:
      reset_time: Unix timestamp when the limit lifts, or ``None``.
      original: The underlying provider exception, retained for context.

    """

    def __init__(self, reset_time: float | None, original: Exception) -> None:
        self.reset_time = reset_time
        self.original = original
        self.diagnostics = error_diagnostics(original)
        if reset_time is not None and reset_time > time.time():
            clock = time.strftime("%H:%M:%S", time.localtime(reset_time))
            delta = reset_time - time.time()
            msg = f"Rate limited. Resumes at {clock} (~{humanize_duration(delta)})."
        else:
            msg = "Rate limited. Try again shortly."
        super().__init__(msg)
        if self.diagnostics:
            logger.info("RateLimitError diagnostics: %s", self.diagnostics)


def is_retryable(error: Exception, model: Model) -> bool:
    """Classify an error as retryable (transient) or fatal.

    Asks the model's provider once (for SDK quirks the status-code
    path can't see); on miss, walks transport/status/cause across
    the cross-provider classifier. Providers receive the outer
    error and own walking their own ``__cause__`` chain.

    Args:
      error: Exception to classify.
      model: Active model; consulted for provider-specific edge cases.

    Returns:
      retryable: True if the error is transient.

    """
    # The request-byte wire-limit is fatal by type: retrying the identical
    # oversized request never helps, and a 5xx/429 wrapper in the cause
    # chain must not flip it retryable. Classify before walking the chain.
    if isinstance(error, RequestTooLargeError):
        return False
    # An entitlement error is fatal whatever status it wears: Anthropic
    # sends "credit balance is too low" as both 400 and 429, and no wait
    # refills an account. Classified before the status walk, which would
    # otherwise read every 429/5xx as retryable.
    if _is_entitlement_error(_body_error_message(error)):
        return False
    if model.is_retryable_provider_error(error):
        return True
    return _is_retryable(error, 0)


def is_rate_limited(error: Exception) -> bool:
    """Classify an error as a provider rate limit.

    Catches both a real 429 status and a body-declared
    ``rate_limit_error`` arriving in-band on a 200 stream: Anthropic
    reports mid-flight throttling via an SSE ``error`` event whose HTTP
    status is 200 and whose unified-ratelimit headers all read
    ``allowed``, so neither the status path nor
    :func:`extract_retry_after` sees it (Issue#424).

    Args:
      error: Exception to classify.

    Returns:
      rate_limited: True when the provider throttled the request.

    """
    if not (
        error_status(error) == 429 or _body_error_type(error) == "rate_limit_error"
    ):
        return False
    # An entitlement 429 is fatal wearing a throttle's status: Anthropic
    # reports "Usage credits are required for fast mode." as a
    # ``rate_limit_error``, and no amount of waiting turns credits on.
    # Treating it as transient burns the whole retry budget while the
    # banner says only "temporarily blocked".
    return not _is_entitlement_error(_body_error_message(error))


def error_status(error: Exception) -> int | None:
    """Extract an HTTP status code from an error, recursively.

    Args:
      error: Exception (or chained cause) to inspect.

    Returns:
      status: HTTP status code, or None if not found.

    """
    return _error_status(error, 0)


def extract_retry_after(error: Exception) -> float | None:
    """Extract retry delay (sec) from a provider error.

    Checks in order:

    1. ``error.retry_after_ms`` attribute (a structured hint a CLI-transport
       provider parses out of a stream-json event; it has no HTTP response).
    2. ``retry-after`` header (RFC 7231, either delta-seconds or HTTP-date).
    3. ``anthropic-ratelimit-unified-reset`` (Unix timestamp when rate
       limit fully clears) - converted to delta-from-now.
    4. Google ``google.rpc.RetryInfo.retryDelay`` (e.g. ``"16s"``),
       carried in the JSON error body rather than a header.

    All return paths are clamped to ``_MAX_SERVER_RETRY_AFTER_SEC`` (24h)
    to neutralize a misconfigured upstream advertising a far-future reset
    that would otherwise wedge persistent retry for days or years.

    Args:
      error: Exception with an attached HTTP response, or carrying a
          structured ``retry_after_ms`` attribute (e.g. a CLI provider
          that parses the hint out of a stream-json event rather than an
          HTTP header).

    Returns:
      delay_sec: Seconds to wait, or None if no hint found.

    """
    # Structured hint on the exception itself, ahead of the HTTP-response
    # path: CLI-transport providers (no ``.response`` object) surface the
    # server's retry hint as a ``retry_after_ms`` attribute. Honor it so
    # the homogeneous retry loop respects it the same as an HTTP header.
    retry_after_ms = getattr(error, "retry_after_ms", None)
    if isinstance(retry_after_ms, (int, float)) and retry_after_ms >= 0:
        return _clamp_retry_after(retry_after_ms / 1000.0)
    response = getattr(error, "response", None)
    if response is None:
        return None
    raw_headers = getattr(response, "headers", {}) or {}
    headers = _lower_headers(raw_headers)
    val = headers.get("retry-after")
    if val is not None:
        try:
            return _clamp_retry_after(_retry_after_seconds(float(val)))
        except (ValueError, TypeError):
            pass
        try:
            dt = parsedate_to_datetime(val)
        except (TypeError, ValueError):
            dt = None
        if dt is not None:
            return _clamp_retry_after(dt.timestamp() - time.time())
    google_delay = _google_retry_delay(_response_body_text(response))
    if google_delay is not None:
        return _clamp_retry_after(google_delay)
    # ``anthropic-ratelimit-unified-reset`` is present on EVERY response --
    # it is the wall-clock when the current usage window rolls over (often
    # hours away, e.g. midnight), NOT a retry instruction. Honoring it
    # unconditionally turns a transient in-band ``rate_limit_error`` on an
    # otherwise-``allowed`` 200 stream into a multi-hour backoff. Only treat
    # it as a retry-after when the limit was actually hit: either a real
    # 429 status, or a unified status header reporting the request was
    # rejected/blocked.
    if _error_status(error, 0) != 429 and not _unified_limit_rejected(headers):
        return None
    reset = headers.get("anthropic-ratelimit-unified-reset")
    if reset is not None:
        try:
            delta = float(reset) - time.time()
        except (ValueError, TypeError):
            return None
        return _clamp_retry_after(delta)
    return None


_UNIFIED_REJECTED_STATUSES = frozenset({"rejected", "rate_limited"})


def _unified_limit_rejected(headers: Mapping[str, str]) -> bool:
    """True when a unified-ratelimit status header reports a blocked request.

    Consults Anthropic's per-window status headers. A ``rejected`` /
    ``rate_limited`` value on any window means the request was actually
    throttled, so the ``-reset`` clock becomes a real retry-after.
    ``allowed`` and ``allowed_warning`` are both non-blocking:
    ``allowed_warning`` is a heads-up that a window is filling (it rides the
    always-present ``-reset`` clock, often the 7d rollover ~24h away), NOT an
    instruction to wait -- honoring it halts a still-serviceable session for
    hours (Issue#316). ``-overage-status`` is excluded: it describes overage
    billing eligibility, not whether THIS request was limited.
    """
    for name in (
        "anthropic-ratelimit-unified-status",
        "anthropic-ratelimit-unified-5h-status",
        "anthropic-ratelimit-unified-7d-status",
    ):
        status = headers.get(name)
        if status is not None and status.strip().lower() in _UNIFIED_REJECTED_STATUSES:
            return True
    return False


_MAX_SERVER_RETRY_AFTER_SEC = (
    24 * 60 * 60
)  # config-globals: ignore -- epoch-vs-delay Retry-After cutoff


def _retry_after_seconds(value: float) -> float:
    """Interpret a numeric ``Retry-After`` as a delay, converting epochs.

    Some servers and proxies send an absolute Unix timestamp instead of
    RFC 7231 delta-seconds. An epoch reset is close to ``now``; a delta --
    even a large one -- is not. Convert to a delta only when ``value`` lands
    within a clamp-window of ``now`` (i.e. it plausibly *is* an epoch), so a
    far-from-now large delta stays a delta and clamps rather than collapsing
    to ~0 via ``value - now``.

    Args:
      value: Parsed ``retry-after`` number (delta-seconds or epoch).

    Returns:
      delay_sec: Non-negative seconds to wait.

    """
    now = time.time()
    if value >= now - _MAX_SERVER_RETRY_AFTER_SEC:
        return max(0.0, value - now)
    return max(0.0, value)


def _clamp_retry_after(delta_sec: float) -> float:
    """Clamp a server-advertised retry delay into ``[0, _MAX_SERVER_RETRY_AFTER_SEC]``."""
    return min(max(0.0, delta_sec), _MAX_SERVER_RETRY_AFTER_SEC)


# Google's 429 backoff arrives in the JSON error body as a
# ``google.rpc.RetryInfo`` detail rather than a header. The protobuf Duration
# JSON encoding is a decimal-seconds string with a trailing ``s`` (e.g.
# ``"16s"`` or ``"7.5s"``).
_GOOGLE_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


def _google_retry_delay(body: str) -> float | None:
    """Parse a Google ``RetryInfo.retryDelay`` (seconds) from an error body."""
    if "retrydelay" not in body.lower():
        return None
    match = _GOOGLE_RETRY_DELAY_RE.search(body)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _response_body_text(response: object) -> str:
    """Return the full response body text, never raising.

    Uncapped: a provider error body is the whole forensic payload, and a
    character clamp cut JSON mid-object exactly when the detail mattered
    -- a ``retryDelay`` far into a large body, say. Guards
    ``ResponseNotRead`` for unread streaming responses.
    """
    if response is None:
        return ""
    try:
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text
        raw = getattr(response, "content", None)
    except httpx.ResponseNotRead:
        return ""
    if isinstance(raw, (bytes, bytearray)):
        try:
            return raw.decode("utf-8", errors="replace")
        except (AttributeError, ValueError):
            return ""
    return ""


def _lower_headers(headers: object) -> dict[str, str]:
    """Return ``headers`` as a flat lowercase-key dict; tolerate any Mapping."""
    if not isinstance(headers, Mapping):
        return {}
    return {
        str(k).lower(): str(v)
        for k, v in cast(Mapping[object, object], headers).items()
    }


def error_diagnostics(error: Exception) -> str:
    """Render a one-line forensic summary of an HTTP error.

    Args:
      error: Exception to inspect, typically carrying ``.response``.

    Returns:
      summary: ``status=... headers={...} body=...`` or empty when
          there is nothing useful to surface.

    """
    snapshot = service_error_snapshot(error)
    parts: list[str] = []
    if snapshot.status is not None:
        parts.append(f"status={snapshot.status}")
    if snapshot.headers:
        parts.append(f"headers={dict(snapshot.headers)}")
    if snapshot.body:
        parts.append(f"body={snapshot.body!r}")
    return " ".join(parts)


def service_error_snapshot(error: Exception) -> runtime_types.ServiceErrorSnapshot:
    """Capture sanitized provider error details for durable runtime events.

    Args:
      error: Exception to inspect, typically carrying ``.response``.

    Returns:
      snapshot: JSON-serializable error details safe for session logs.

    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    # An SDK error stringifies to its entire JSON body; the body's own
    # ``message`` is the one sentence worth showing a user.
    return runtime_types.ServiceErrorSnapshot(
        type_name=type(error).__name__,
        message=_body_error_message(error) or str(error),
        status=error_status(error),
        headers=_diagnostic_headers(headers),
        body=_response_body_text(response),
    )


async def send_with_retry(
    model: Model,
    request: ModelRequest,
    *,
    publish: Callable[[runtime_types.RuntimeEvent], None] | None = None,
    show_thinking: bool = True,
    max_attempts: int,
    persistent_retry: bool,
    publish_recoverable: Callable[[str], None],
    on_discarded_response: Callable[[ModelResponse], None] | None = None,
    on_service_suspended: Callable[[float, float, bool, Exception], None] | None = None,
    resume_retry_at: float | None = None,
    max_persistent_attempts: int = DEFAULT_MAX_PERSISTENT_ATTEMPTS,
) -> ModelResponse:
    """Send with backoff and error classification.

    On retryable errors: exponential backoff with jitter, respects
    Retry-After headers. Tracks text already shown live on a failed
    first attempt and skips that prefix on retry.

    Args:
      model: Model to send the request to.
      request: Fully-built model request.
      publish: Runtime event sink for streamed events
          (``ModelResponsePartial`` text chunks,
          ``ModelResponseThinking`` chunks, and ``ToolLabel`` items from
          CLI transports). ``None`` discards stream events; the request
          still streams under the hood (buffered transports are
          unreliable at large prompt sizes). Live text is tracked across
          a failed first attempt so the prefix is skipped on retry;
          thinking only fires on the first attempt (on retry it is read
          from the final response so the renderer never repeats it).
      show_thinking: When ``False``, ``ModelResponseThinking`` events are
          dropped before reaching ``publish``.
      max_attempts: Maximum number of retry attempts.
      persistent_retry: Enable persistent backoff for 429/529 errors.
      publish_recoverable: Callback for transient errors that recovered;
          each retry attempt invokes it with a ``multipart/x-error`` Message
          carrying the underlying exception and a structured stack trace.
      on_discarded_response: Called with the response from a completed
          request that will be retried (e.g. StreamInterruptedError).
          The API billed for these tokens; this callback lets the
          caller account them.
      on_service_suspended: Called when a recoverable provider error
          schedules a retry sleep. Arguments are retry_at, delay_sec,
          server_supplied, and the original exception.
      resume_retry_at: Optional wall-clock timestamp loaded from a prior
          service suspension; sleeps remaining time before the first send.
      max_persistent_attempts: Hard cap on persistent-mode retry attempts.
          Without this, a permanently-failing server wedges the loop
          indefinitely. Defaults to
          :data:`DEFAULT_MAX_PERSISTENT_ATTEMPTS`.

    Returns:
      response: Completed model response.

    Raises:
      RetriesExhaustedError: If all attempts fail (including persistent
          mode hitting ``max_persistent_attempts``).
      RateLimitError: On a rate limit (429 or in-band) in non-persistent
          mode: immediately when the server advertises a reset beyond
          ``INTERACTIVE_MAX_SLEEP_SEC``, otherwise once
          ``RATE_LIMIT_RETRY_BUDGET_SEC`` of capped-backoff retries
          fails to outlast the throttle.

    """
    if resume_retry_at is not None:
        delay = max(0.0, resume_retry_at - time.time())
        # A resume wait past the interactive ceiling is replayed (not advanced)
        # on every send, so it never decays into range -- typing to resume just
        # re-enters a multi-hour uninterruptible sleep (Issue#316 bug #5). The
        # user typing IS an explicit "try now"; skip the stale wait and let the
        # send proceed rather than wedge the REPL.
        if 0 < delay <= INTERACTIVE_MAX_SLEEP_SEC:
            await asyncio.sleep(delay)
        elif delay > INTERACTIVE_MAX_SLEEP_SEC:
            # Skipping silently leaves the prior session's "service suspended"
            # banner uncleared with no signal it was abandoned. Surface the skip
            # in-band so the user knows why the send is proceeding now.
            publish_recoverable(
                f"prior resume wait ({delay:.0f}s) exceeds the interactive"
                f" ceiling ({INTERACTIVE_MAX_SLEEP_SEC:.0f}s); skipping it and"
                " sending now"
            )
    last_error: Exception | None = None
    prior_emitted = ""
    attempt = -1
    stream_attempt = 0
    persistent_attempt = 0
    rate_limit_attempt = 0
    rate_limit_deadline: float | None = None
    stream_interrupts = 0
    while True:
        attempt += 1
        if attempt >= max_attempts:
            break
        live = attempt == 0 and not prior_emitted
        chunks: list[str] = []
        sink = _make_stream_sink(chunks, publish if live else None, show_thinking)
        stream_attempt += 1
        try:
            resp = await model.stream(request=request, publish=sink)
            full = "".join(chunks)
            if publish is not None and not live and full != prior_emitted:
                if full.startswith(prior_emitted):
                    suffix = full.removeprefix(prior_emitted)
                    if suffix:
                        publish(runtime_types.ModelResponsePartial(suffix))
                else:
                    # The partial streamed before the interrupt does not prefix
                    # the retry, so it cannot be extended -- the text above this
                    # banner is superseded by what follows. (A structural fix
                    # that drops the stale text outright needs a renderer-side
                    # reset event; the explicit banner is the in-band signal.)
                    publish(
                        runtime_types.ModelResponsePartial(
                            "\n[retry diverged; discard the text above -- "
                            "the corrected response follows]\n"
                        )
                    )
                    publish(runtime_types.ModelResponsePartial(full))
            return resp
        except StreamInterruptedError as e:
            stream_interrupts += 1
            prior_emitted = "".join(chunks) if live else prior_emitted
            publish_recoverable(
                f"stream interrupted (attempt {stream_attempt},"
                f" {stream_interrupts} interrupts so far): {e}"
            )
            if stream_interrupts > _MAX_STREAM_INTERRUPT_RETRIES:
                logger.warning(
                    "Stream indicated tool_use but delivered"
                    " no blocks after %d retries;"
                    " returning partial response.",
                    stream_interrupts - 1,
                )
                return e.response
            if on_discarded_response is not None:
                on_discarded_response(
                    e.response
                )  # may raise (e.g. budget exhaustion) -- intentional
            attempt -= 1
            continue
        except Exception as e:
            if model.is_context_overflow(e):
                publish_recoverable(
                    f"context overflow (attempt {attempt}): {type(e).__name__}: {e}"
                )
                raise
            if not is_retryable(e, model):
                raise
            last_error = e
            if live:
                prior_emitted = "".join(chunks)
            status = error_status(e)
            # ``httpx.RemoteProtocolError`` ("peer closed connection without
            # sending complete message body") shows up when an upstream edge
            # cuts a streamed request body mid-flight -- typical on very
            # large compaction payloads. These carry no HTTP status, so the
            # normal 32s-cap loop bails after a handful of attempts. When
            # the caller opted into ``persistent_retry`` (compactor, batch
            # jobs), promote protocol errors into the long-backoff branch
            # so transient connection cuts don't fail the operation.
            is_protocol_flake = isinstance(e, httpx.RemoteProtocolError)
            rate_limited = is_rate_limited(e)
            # A 529's body type is ``overloaded_error``; in-band it arrives
            # on a 200 stream, so the status check alone misses it -- the
            # same 200-wearing-an-error shape as the in-band rate limit.
            overloaded = status == 529 or _body_error_type(e) == "overloaded_error"
            persistent = (
                persistent_retry
                and model.spec.retries_internally
                and (rate_limited or overloaded or is_protocol_flake)
            )
            server_delay = extract_retry_after(e)
            # Interactive (non-persistent) mode: never silently sleep for a
            # long server-advertised backoff. A multi-minute+ wait blocks the
            # whole REPL on one uninterruptible ``asyncio.sleep`` (the user
            # can't even ``/login`` out of it). Surface it as a
            # ``RateLimitError`` halt carrying the reset time so the user
            # can switch models or wait deliberately.
            if (
                not persistent
                and server_delay is not None
                and server_delay > INTERACTIVE_MAX_SLEEP_SEC
            ):
                raise RateLimitError(time.time() + server_delay, e) from e
            if persistent:
                persistent_attempt += 1
                if persistent_attempt >= max_persistent_attempts:
                    raise RetriesExhaustedError(
                        f"Failed after {persistent_attempt} persistent attempts: {e}",
                    ) from e
                attempt_label = persistent_attempt - 1
                # Exponent is the 0-based attempt index, uniform with the
                # rate-limit and generic branches below (first backoff starts at
                # ``RETRY_BASE_SEC``, not one rung higher).
                delay = _backoff_delay(attempt_label, cap=PERSISTENT_MAX_BACKOFF_SEC)
                if server_delay is not None:
                    delay = max(delay, server_delay)
                attempt -= 1
            elif rate_limited:
                # A throttle with no (or a short) advertised reset clears in
                # seconds-to-minutes, so retry on a wall-clock budget rather
                # than the attempt counter -- ``max_attempts`` local backoffs
                # total ~9s, which gives up just before the limiter relents
                # (Issue#424). Budget exhaustion halts as a rate limit
                # ("type to retry"), not a generic retry failure.
                if rate_limit_deadline is None:
                    rate_limit_deadline = time.time() + RATE_LIMIT_RETRY_BUDGET_SEC
                rate_limit_attempt += 1
                if rate_limit_attempt >= _MAX_RATE_LIMIT_ATTEMPTS:
                    raise RateLimitError(None, e) from e
                attempt_label = rate_limit_attempt - 1
                delay = _backoff_delay(attempt_label, cap=INTERACTIVE_MAX_SLEEP_SEC)
                if server_delay is not None:
                    delay = max(delay, server_delay)
                if time.time() + delay > rate_limit_deadline:
                    reset_time = (
                        time.time() + server_delay if server_delay is not None else None
                    )
                    raise RateLimitError(reset_time, e) from e
                attempt -= 1
            else:
                # Exhaustion raises here, BEFORE the sleep: sleeping first
                # paints "resumes in Ns", waits the full backoff, then fails
                # without sending -- the retry the banner promised never goes
                # out (Issue#424).
                if attempt + 1 >= max_attempts:
                    raise RetriesExhaustedError(
                        f"Failed after {max_attempts} attempts: {e}",
                    ) from e
                delay = _backoff_delay(attempt, cap=MAX_RETRY_DELAY)
                if server_delay is not None:
                    delay = max(delay, server_delay)
                attempt_label = attempt
            diagnostics = error_diagnostics(e)
            publish_recoverable(
                f"retry attempt {attempt_label}, waiting {delay:.1f}s:"
                f" {type(e).__name__}: {e}"
                + (f" [{diagnostics}]" if diagnostics else "")
            )
            logger.warning(
                "API error (attempt %d/%d): %s%s: %s. Retrying in %.0fs.%s",
                attempt_label + 1,
                max_attempts,
                type(e).__name__,
                f" {status}" if status is not None else "",
                e,
                delay,
                f" {diagnostics}" if diagnostics else "",
            )
            retry_at = time.time() + delay
            # Only surface the user-facing "model service suspended" banner
            # for waits that are actually worth interrupting the user over:
            # a server-advertised backoff, or a local backoff past a short
            # threshold. Sub-second transport blips (e.g. an httpx ReadError
            # that recovers on the next attempt) retry silently -- otherwise
            # every transient network hiccup paints a scary suspension banner
            # for a wait the user never even perceives. The retry is still
            # logged via ``publish_recoverable`` above.
            notable = server_delay is not None or delay >= SUSPENSION_NOTICE_SEC
            if on_service_suspended is not None and notable:
                on_service_suspended(retry_at, delay, server_delay is not None, e)
            await asyncio.sleep(delay)
    raise RetriesExhaustedError(
        f"Failed after {max_attempts} attempts: {last_error}",
    ) from last_error


def _backoff_delay(attempt: int, *, cap: float) -> float:
    """Exponential backoff with downward jitter; ``cap`` is a hard ceiling.

    The base doubles per attempt (``RETRY_BASE_SEC * 2**attempt``) and is
    clamped to ``cap``. Jitter is then subtracted (up to 25% of the clamped
    base), so the result is always in ``[0.75 * cap_base, cap_base]`` and never
    exceeds ``cap`` -- ``cap`` stays a true upper bound (the interactive
    ceiling must never be breached). Jittering DOWN rather than the prior
    ``min(base + jitter, cap)`` keeps the jitter visible at the ceiling:
    that shape collapsed every late attempt to exactly ``cap``, re-synchronizing
    concurrent retriers into a thundering herd.

    Args:
      attempt: Zero-based backoff attempt number.
      cap: Hard ceiling on the returned delay, in seconds.

    Returns:
      delay_sec: Backoff delay in ``[0.75 * cap_base, cap_base]`` where
          ``cap_base = min(RETRY_BASE_SEC * 2**attempt, cap)``.

    """
    base = min(RETRY_BASE_SEC * (2.0**attempt), cap)
    return base - random.uniform(0, 0.25 * base)  # noqa: S311 -- jitter, not security


def _is_retryable(error: Exception, depth: int) -> bool:
    """Walk transport/status/cause chain (depth-capped) for retryability."""
    if isinstance(error, (httpx.TransportError, ConnectionError, TimeoutError)):
        return True
    status = _error_status(error, 0)
    if status is not None and (status in RETRYABLE_STATUS_CODES or status >= 500):
        return True
    cause = error.__cause__
    if cause is not None and isinstance(cause, Exception) and depth < _MAX_CAUSE_DEPTH:
        return _is_retryable(cause, depth + 1)
    return False


_ENTITLEMENT_PHRASES: Final = (
    "usage credits are required",
    "credit balance is too low",
    "insufficient credits",
    "exceeded your usage credits",
    "out of extra usage",
)
"""Body-message markers for a 429 that describes entitlement, not throughput.

Anthropic phrases one fatal condition several ways -- credits not enabled,
balance drained, org extra-usage exhausted -- all as ``rate_limit_error``.
Waiting refills none of them, so each must classify fatal.
"""


def _is_entitlement_error(message: str | None) -> bool:
    """Whether a 429 body message describes a missing entitlement."""
    if message is None:
        return False
    lowered = message.lower()
    return any(phrase in lowered for phrase in _ENTITLEMENT_PHRASES)


def _body_error_message(error: Exception) -> str | None:
    """Extract the provider-declared error ``message`` from a JSON body."""
    body = getattr(error, "body", None)
    if not isinstance(body, Mapping):
        return None
    typed = cast(Mapping[object, object], body)
    nested = typed.get("error")
    if isinstance(nested, Mapping):
        message = cast(Mapping[object, object], nested).get("message")
        if isinstance(message, str):
            return message
    message = typed.get("message")
    return message if isinstance(message, str) else None


def _body_error_type(error: Exception) -> str | None:
    """Extract a provider-declared error ``type`` from a JSON error body."""
    body = getattr(error, "body", None)
    if not isinstance(body, Mapping):
        return None
    typed = cast(Mapping[object, object], body)
    error_type = typed.get("type")
    if error_type == "error":
        nested = typed.get("error")
        if isinstance(nested, Mapping):
            error_type = cast(Mapping[object, object], nested).get("type")
    return error_type if isinstance(error_type, str) else None


def _error_status(error: Exception, depth: int) -> int | None:
    """Walk ``status_code`` / ``response.status_code`` / ``__cause__`` for a status."""
    status = getattr(error, "status_code", None)
    if status is None:
        resp = getattr(error, "response", None)
        status = getattr(resp, "status_code", None)
    if isinstance(status, int):
        return status
    cause = error.__cause__
    if cause is not None and isinstance(cause, Exception) and depth < _MAX_CAUSE_DEPTH:
        return _error_status(cause, depth + 1)
    return None


def _make_stream_sink(
    chunks: list[str],
    live: Callable[[runtime_types.RuntimeEvent], None] | None,
    show_thinking: bool,
) -> Callable[[runtime_types.RuntimeEvent], None]:
    """Build the ``publish`` sink handed to ``model.stream``.

    Always captures text chunks into ``chunks`` for cross-attempt
    retry-dedup. When ``live`` is set (first attempt, nothing emitted
    yet) it also forwards events to the runtime sink: text and labels
    unconditionally, thinking only when ``show_thinking``.
    """

    def _sink(event: runtime_types.RuntimeEvent) -> None:
        if isinstance(event, runtime_types.ModelResponsePartial):
            chunks.append(event.text)
        if live is None:
            return
        if isinstance(event, runtime_types.ModelResponseThinking) and not show_thinking:
            return
        live(event)

    return _sink


def _diagnostic_headers(headers: object) -> dict[str, str]:
    """Return allowlisted response headers useful for rate-limit forensics."""
    if not isinstance(headers, Mapping):
        return {}
    out: dict[str, str] = {}
    for key, value in cast(Mapping[object, object], headers).items():
        name = str(key).lower()
        if (
            name == "retry-after"
            or name.startswith(("anthropic-ratelimit", "x-ratelimit"))
            or name in {"request-id", "x-request-id"}
        ):
            out[str(key)] = str(value)
    return out
