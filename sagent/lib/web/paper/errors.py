"""Exception hierarchy for :mod:`sagent.lib.web.paper`.

The library raises; call sites (e.g. sagent tools) catch and render. This
replaces the ``ToolResult``-or-value return union the sagent tools threaded
through their old shared helpers, keeping the library free of any tool shape.
"""

from __future__ import annotations


__all__ = [
    "BackendError",
    "InvalidIdError",
    "NotFoundError",
    "PaperError",
    "RateLimitError",
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
