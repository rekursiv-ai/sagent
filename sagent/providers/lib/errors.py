"""Shared provider-boundary error helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, cast

import contextlib
import json

from sagent.types.exceptions import UserFacingError
from sagent.types.model import RequestTooLargeError


if TYPE_CHECKING:
    # httpx2, not httpx2: the exception walked here is raised by the provider
    # SDKs, and both anthropic 1.2 and openai depend on httpx2. Checking the
    # httpx2 class instead silently never matches -- the classes are unrelated.
    import httpx2
else:
    from wrapt import lazy_import

    httpx2 = lazy_import("httpx2")


# HTTP 413 ("Payload Too Large") is the cross-provider signal for the
# request-byte wire-limit: distinct from a token context-window overflow
# and from rate limits. Anthropic, OpenAI, and Google all surface their
# byte ceilings as 413. Body phrases catch the rare status-less variants.
_REQUEST_TOO_LARGE_STATUS: Final = 413
# Unambiguous byte-limit phrases: each names the REQUEST/PAYLOAD/ENTITY size,
# never the token window. These win over any co-occurring context phrase.
# The looser prefix "request exceeds the maximum" is deliberately NOT here --
# it is a prefix of both "...maximum allowed number of bytes" (byte) and
# "...maximum context length" (token), so it cannot disambiguate on its own;
# the fuller "...number of bytes" form is covered by "request entity too
# large" / "maximum request size" instead.
_REQUEST_TOO_LARGE_PHRASES: Final = (
    "request_too_large",
    "maximum request size",
    "number of bytes",
    "payload too large",
    "request entity too large",
)
# Token context-window overflow phrases, pooled across every vendor: each
# provider used to keep its own list, so a spelling only one of them knew
# propagated raw from the other four (session ``190b6baec7ed``). Bare
# "context length" is deliberately excluded: it is subsumed by the specific
# forms below and would mismatch a byte error's remediation prose.
_CONTEXT_OVERFLOW_PHRASES: Final = (
    "context_length_exceeded",
    "maximum context length",
    "input too large",
    "input too long",
    # A per-ITEM string cap, not the request-byte ceiling: shedding
    # attachment bytes cannot clear it, but the compactor's tool-result
    # shrink can, so it must classify as token overflow.
    "string_above_max_length",
    # Anthropic (API + CLI).
    "prompt is too long",
    "prompt too long",
    "too_long",
)
"""Unambiguous token-overflow phrases: each names the PROMPT or the
CONTEXT, never a request size. Safe for :func:`is_request_too_large` to
read as a byte-classification veto."""

_AMBIGUOUS_CONTEXT_OVERFLOW_PHRASES: Final = ("exceeds the maximum",)
"""Phrases that name neither unit. ``exceeds the maximum`` prefixes both
``...context length`` (token) and ``...request size`` (byte), so it is
sound ONLY after the byte case is excluded -- which is why
:func:`is_request_too_large` must not read it, or a 413 whose body says
"exceeds the maximum size" would veto its own byte classification."""

_GUARDED_CONTEXT_OVERFLOW_PHRASES: Final = (
    ("context window", ("exceed", "overflow", "maximum")),
    ("model context", ("exceed",)),
)
"""Phrases needing a co-occurring marker: each appears benignly in
tool-schema validation errors, where a bare match is a false positive."""

_CONTEXT_OVERFLOW_CODES: Final = frozenset(
    {"context_length_exceeded", "string_above_max_length"}
)
"""Vendor ``error.code`` values that name the condition outright."""

PER_ITEM_STRING_CAP_BODY: Final = (
    "Error code: 400 - {'error': {'message': \"Invalid 'input[388].output': "
    "string too long. Expected a string with maximum length 10485760, but got "
    "a string with length 11143438 instead.\", 'type': 'invalid_request_error', "
    "'param': 'input[388].output', 'code': 'string_above_max_length'}}"
)
"""Verbatim 400 from session ``190b6baec7ed``; shared by the tests that
pin both halves of its classification."""


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
    if _names_token_overflow(lower):
        return False
    return status == _REQUEST_TOO_LARGE_STATUS


def is_context_overflow_text(body: str) -> bool:
    """Whether ``body`` names a token-side overflow the compactor can shed.

    Callers MUST exclude the byte wire-limit first (see
    :func:`is_request_too_large`): several phrases here are byte-ambiguous
    and only sound once that case is gone.

    Shared so the phrase set lives in ONE place. Every provider used to
    hardcode its own list, so a spelling only one of them knew propagated
    raw from the other four -- the shape that wedged session
    ``190b6baec7ed``. The union is deliberate: a missed overflow is fatal
    (the raw error reaches the runtime and every retry repeats it) while a
    false positive merely costs one wasted compaction.

    ``model context`` / ``context window`` stay GUARDED rather than bare:
    both appear benignly in tool-schema validation errors, and a bare match
    classified "'model context' field missing in tools schema" as overflow.

    Args:
      body: Decoded response body (or stringified error).

    Returns:
      overflow: True when the text describes a token-side overflow.

    """
    # A structured ``error.code`` is authoritative when present: it is the
    # vendor naming the condition, not prose that happens to contain a phrase.
    with contextlib.suppress(json.JSONDecodeError):
        parsed = json.loads(body)
        if isinstance(parsed, Mapping):
            error = cast(Mapping[str, object], parsed).get("error")
            if isinstance(error, Mapping):
                code = cast(Mapping[str, object], error).get("code")
                # ``str`` before membership: ``code`` is whatever the
                # vendor sent, and an unhashable value (``{}``, ``[]``)
                # raises ``TypeError`` against a ``frozenset`` -- a crash
                # inside the handler whose whole job is to classify the
                # error it was handed.
                if isinstance(code, str):
                    return code in _CONTEXT_OVERFLOW_CODES
    lower = body.lower()
    if any(phrase in lower for phrase in _AMBIGUOUS_CONTEXT_OVERFLOW_PHRASES):
        return True
    return _names_token_overflow(lower)


def _names_token_overflow(lower: str) -> bool:
    """Unambiguous token-overflow signals in an already-lowercased body.

    Excludes :data:`_AMBIGUOUS_CONTEXT_OVERFLOW_PHRASES`, so this is also
    what :func:`is_request_too_large` reads to veto byte classification --
    a phrase naming neither unit must not decide that question.
    """
    if any(phrase in lower for phrase in _CONTEXT_OVERFLOW_PHRASES):
        return True
    return any(
        phrase in lower and any(marker in lower for marker in markers)
        for phrase, markers in _GUARDED_CONTEXT_OVERFLOW_PHRASES
    )


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

    def __init__(self, *, provider_name: str) -> None:
        super().__init__(
            f"{provider_name} streaming request failed before sagent could read "
            "the provider error body. The underlying HTTP error was hidden by "
            "the provider SDK while formatting a streaming response. Retry "
            "after running /compact, use /clear for a fresh session, or switch "
            "providers with /model."
        )


def find_response_not_read(exc: BaseException) -> httpx2.ResponseNotRead | None:
    """Walk the ``__cause__``/``__context__`` chain for a ``ResponseNotRead``.

    A streaming provider SDK can raise ``httpx2.ResponseNotRead`` (directly
    or chained) when it formats an error message by touching a response
    body that was never read. ``ResponseNotRead`` is a usage error, not a
    transport error, so the retry classifier treats it as fatal; callers
    use this to detect and re-wrap it into a user-facing error instead.

    Args:
      exc: Exception to inspect, including its cause/context chain.

    Returns:
      cause: The first ``ResponseNotRead`` found, or ``None``.

    """
    # Both links are followed, not ``cause or context``: the two form a
    # TREE, and an SDK that raises while formatting a hidden body puts the
    # ``ResponseNotRead`` on the context of an exception that already has
    # an unrelated cause. Taking one branch missed exactly that shape.
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        if isinstance(current, httpx2.ResponseNotRead):
            return current
        seen.add(id(current))
        pending.extend(
            link
            for link in (current.__cause__, current.__context__)
            if link is not None
        )
    return None
