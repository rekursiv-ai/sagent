"""Shared provider-boundary error helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.types.exceptions import UserFacingError


if TYPE_CHECKING:
    import httpx
else:
    from sagent.lib.lazy_import import lazy_import

    httpx = lazy_import("httpx")


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
