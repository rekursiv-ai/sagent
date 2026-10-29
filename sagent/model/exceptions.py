"""Errors a model raises while answering, and a provider raises naming one.

A provider retries short glitches itself, so a ``RetryableError`` that reaches
the caller means the advised wait is long, or text had already streamed.

Typical usage example:

  try:
      async for event in opus.stream(request):
          ...
  except RateLimitedError as error:
      await asyncio.sleep(error.retry_after_sec)
  except ContextOverflowError:
      ...  # Compact the session, then send again.
"""

from __future__ import annotations


__all__ = [
    "AuthError",
    "BillingError",
    "ContextOverflowError",
    "DisconnectedError",
    "OverloadedError",
    "RateLimitedError",
    "RefusedError",
    "RequestTooLargeError",
    "RetryableError",
    "UnknownModelError",
    "UnknownProviderError",
    "UnsupportedTagError",
]


class RetryableError(Exception):
    """A failure that the same request may get past later.

    Args:
      message: What failed.
      retry_after_sec: The wait the server advised; ``0`` when it gave none.

    """

    def __init__(self, message: str, *, retry_after_sec: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after_sec = retry_after_sec


class RateLimitedError(RetryableError):
    """The account's rate limit is spent.

    Args:
      message: What failed.
      retry_after_sec: The wait the server advised; ``0`` when it gave none.
      limit: The window spent, such as ``5-hour`` or ``weekly``; ``""`` when
        the server did not say.

    """

    def __init__(
        self,
        message: str,
        *,
        retry_after_sec: float = 0.0,
        limit: str = "",
    ) -> None:
        super().__init__(message, retry_after_sec=retry_after_sec)
        self.limit = limit


class OverloadedError(RetryableError):
    """The vendor's servers are busy or failing."""


class DisconnectedError(RetryableError):
    """The connection dropped."""


class AuthError(Exception):
    """The credentials are missing, expired, or revoked."""


class BillingError(Exception):
    """The account has no credit for the request; waiting does not help."""


class ContextOverflowError(Exception):
    """The request has more tokens than the model's window."""


class RequestTooLargeError(Exception):
    """The request has more bytes than the vendor accepts."""


class RefusedError(Exception):
    """The vendor's safeguards declined the request."""


class UnknownModelError(ValueError):
    """No model answers to the name."""


class UnknownProviderError(ValueError):
    """No provider, or more than one, answers to the name."""


class UnsupportedTagError(ValueError):
    """The name asks for a context window the model does not offer."""
