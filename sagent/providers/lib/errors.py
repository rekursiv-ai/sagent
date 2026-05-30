"""Shared provider-boundary error helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.types.exceptions import UserFacingError


if TYPE_CHECKING:
    import httpx


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
