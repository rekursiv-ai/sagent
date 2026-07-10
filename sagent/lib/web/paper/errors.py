"""Exception hierarchy for :mod:`sagent.lib.web.paper`.

The library raises; call sites (e.g. sagent tools) catch and render. This
replaces the ``ToolResult``-or-value return union the sagent tools threaded
through their old shared helpers, keeping the library free of any tool shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from sagent.lib.web.errors import FetchError


__all__ = [
    "BackendError",
    "InvalidIdError",
    "NotFoundError",
    "PaperError",
    "RateLimitError",
    "translate_http_error",
]


class PaperError(Exception):
    """Base class for every error raised by :mod:`sagent.lib.web.paper`."""


class InvalidIdError(PaperError):
    """An identifier did not match a known DOI or arXiv shape."""


class NotFoundError(PaperError):
    """A backend reported that the requested entity does not exist (HTTP 404)."""


class RateLimitError(PaperError):
    """A backend throttled the request (HTTP 429) after exhausting backoff."""


class BackendError(PaperError):
    """A backend failed for any other reason (HTTP 5xx, bad JSON, timeout).

    Attributes:
      status: HTTP status when one was seen, else ``0`` (timeout / no response).

    """

    def __init__(self, message: str, *, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


def translate_http_error(
    e: FetchError,
    *,
    backend: str,
    rate_limit_message: str = "",
    not_found_on_404: bool = True,
) -> PaperError:
    """Map a transport :class:`FetchError` to the paper exception hierarchy.

    The single source of truth for the status -> :class:`PaperError` mapping that
    every provider shares, so two backends cannot drift on what a given HTTP
    status means (they did: S2 mapped 404 -> NotFound, OpenAlex did not).

    Args:
      e: The transport error carrying ``status``/``body``.
      backend: Human name for messages (e.g. ``"Semantic Scholar"``).
      rate_limit_message: Backend-specific 429 guidance (API-key hint, budget
        reset, ...). Empty for a generic message.
      not_found_on_404: When True (default) a 404 is :class:`NotFoundError`.
        Set False for a backend whose real not-found is SEMANTIC (e.g. OpenAlex
        returns 200 + empty results), so an HTTP 404 there is a bad endpoint --
        a :class:`BackendError`, not a missing entity.

    Returns:
      The matching :class:`PaperError` subclass (caller ``raise ... from e``).

    """
    if e.status == 404 and not_found_on_404:
        return NotFoundError(f"{backend}: not found.")
    if e.status == 429:
        return RateLimitError(
            rate_limit_message or f"{backend} rate limit hit; retry shortly."
        )
    body = e.body[:200].decode(errors="replace")
    return BackendError(f"{backend} HTTP {e.status}: {body}", status=e.status)
