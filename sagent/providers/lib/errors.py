"""Shared provider-boundary error helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.types.exceptions import UserFacingError
from sagent.types.model import RequestTooLargeError


if TYPE_CHECKING:
    import httpx
else:
    from wrapt import lazy_import

    httpx = lazy_import("httpx")


# HTTP 413 ("Payload Too Large") is the cross-provider signal for the
# request-byte wire-limit: distinct from a token context-window overflow
# and from rate limits. Anthropic, OpenAI, and Google all surface their
# byte ceilings as 413. Body phrases catch the rare status-less variants.
_REQUEST_TOO_LARGE_STATUS = 413
# Unambiguous byte-limit phrases: each names the REQUEST/PAYLOAD/ENTITY size,
# never the token window. These win over any co-occurring context phrase.
# The looser prefix "request exceeds the maximum" is deliberately NOT here --
# it is a prefix of both "...maximum allowed number of bytes" (byte) and
# "...maximum context length" (token), so it cannot disambiguate on its own;
# the fuller "...number of bytes" form is covered by "request entity too
# large" / "maximum request size" instead.
_REQUEST_TOO_LARGE_PHRASES = (
    "request_too_large",
    "maximum request size",
    "number of bytes",
    "payload too large",
    "request entity too large",
)
# Token context-window overflow phrases. Real token overflow is a 400 with
# code ``context_length_exceeded`` / "maximum context length is N tokens"
# (Gemini and OpenAI-compatible servers also sometimes return these on 413).
# Bare "context length" is deliberately excluded: it is subsumed by the
# specific forms below and would mismatch a byte error's remediation prose.
_CONTEXT_OVERFLOW_PHRASES = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "model context",
    "input too large",
)


def error_status_code(error: BaseException) -> int | None:
    """Extract an HTTP status from an exception's ``status_code``/response.

    Walks ``error.status_code`` then ``error.response.status_code``. Used
    by provider classifiers that receive a raised exception rather than a
    raw status at a stream boundary.

    Args:
      error: Exception to inspect.

    Returns:
      status: HTTP status code, or ``None`` when not present.

    """
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_request_too_large(status: int | None, body: str) -> bool:
    """Classify a provider error as the byte wire-limit (HTTP 413).

    Shared across every provider so the byte-vs-token distinction is made
    once, not re-derived per vendor. A byte-phrase body is authoritative.
    A bare 413 status defaults to the byte limit UNLESS the body names a
    token context-window overflow -- some providers reuse 413 for token
    overflow, which a larger-window model relieves, so it must stay a
    ``PromptTooLongError``.

    Args:
      status: HTTP status code of the failing response, when known.
      body: Decoded response body (or stringified error).

    Returns:
      too_large: True when the error indicates the request-byte ceiling.

    """
    lower = body.lower()
    # Disambiguate by phrase SPECIFICITY, not status or order:
    # 1. An unambiguous byte phrase (names request/payload/entity size) wins
    #    -- it is byte overflow even when the body also mentions the context
    #    window in remediation prose.
    # 2. Else a token context-window phrase makes it token overflow (a
    #    larger-window model relieves it; stays a ``PromptTooLongError``).
    # 3. Else fall back to the 413 status as the byte signal.
    if any(phrase in lower for phrase in _REQUEST_TOO_LARGE_PHRASES):
        return True
    if any(phrase in lower for phrase in _CONTEXT_OVERFLOW_PHRASES):
        return False
    return status == _REQUEST_TOO_LARGE_STATUS


def raise_if_request_too_large(
    status: int | None, body: str, *, cause: BaseException | None = None
) -> None:
    """Raise :class:`RequestTooLargeError` when the error is the byte limit.

    Providers call this at their stream error boundary, before their
    token-context-overflow check, so the byte wire-limit routes to the
    agent's byte-overflow recovery (which sheds attachment bytes) rather
    than being mislabeled a context-window overflow.

    Args:
      status: HTTP status code of the failing response, when known.
      body: Decoded response body (or stringified error).
      cause: Original provider exception to chain via ``__cause__`` so
          diagnostics and the retry classifier can inspect it. Pass the
          caught exception whenever one exists. It is intentionally ``None``
          at HTTP-body boundaries (Google / OpenAI-compatible stream paths
          inspect a parsed response body with no exception in scope), so
          ``__cause__`` is preserved only on the exception-handling paths
          (Anthropic API, OpenAI subscription).

    Raises:
      RequestTooLargeError: When ``status``/``body`` indicate the byte limit.

    """
    if is_request_too_large(status, body):
        raise RequestTooLargeError(body) from cause


class StreamingResponseNotReadError(UserFacingError):
    """Provider SDK hid a streaming HTTP error body before sagent saw it."""

    def __init__(self, *, provider_name: str, cause: httpx.ResponseNotRead) -> None:
        super().__init__(
            f"{provider_name} streaming request failed before sagent could read "
            "the provider error body. The underlying HTTP error was hidden by "
            "the provider SDK while formatting a streaming response. Retry "
            "after running /compact, use /clear for a fresh session, or switch "
            "providers with /model."
        )
        self.__cause__ = cause


def find_response_not_read(exc: BaseException) -> httpx.ResponseNotRead | None:
    """Walk the ``__cause__``/``__context__`` chain for a ``ResponseNotRead``.

    A streaming provider SDK can raise ``httpx.ResponseNotRead`` (directly
    or chained) when it formats an error message by touching a response
    body that was never read. ``ResponseNotRead`` is a usage error, not a
    transport error, so the retry classifier treats it as fatal; callers
    use this to detect and re-wrap it into a user-facing error instead.

    Args:
      exc: Exception to inspect, including its cause/context chain.

    Returns:
      cause: The first ``ResponseNotRead`` found, or ``None``.

    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, httpx.ResponseNotRead):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None
