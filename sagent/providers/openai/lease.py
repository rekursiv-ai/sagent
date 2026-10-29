"""A ChatGPT login lent by another process, which alone holds its refresh token.

A refresh token works once, so only one process may renew a login. Another
borrows access tokens from it: ``CodexLogin.lend`` is a lease, and however the
borrower reaches it (a pipe, a socket, a call in the same process) is the
caller's to choose.

Typical usage example:

  holder = CodexLogin(resources=shared)                 # Holds the refresh token.
  borrowed = OpenAISub(lease=holder.lend)               # Never touches the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import asyncio
import time

from sagent.model.exceptions import AuthError


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = [
    "BorrowedLogin",
    "LeasedToken",
    "TokenLease",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class LeasedToken:
    """An access token lent by whoever holds the refresh token."""

    access_token: str
    account_id: str
    expires_at: float
    """When the token expires, in seconds since the epoch."""


class TokenLease(Protocol):
    """Lends an access token: a fresh one when ``rejected`` was refused."""

    async def __call__(self, *, rejected: str) -> LeasedToken:
        """Return a usable access token.

        Args:
          rejected: The token the server refused; empty when it only aged out
            by the local clock.

        Returns:
          token: A token that is not ``rejected``, unless the holder has no newer.

        Raises:
          AuthError: The holder's login is expired or revoked.

        """
        ...


class BorrowedLogin:
    """A login whose tokens are borrowed, never renewed or stored here.

    Args:
      lease: Lends the next token when this one ages out or is refused.

    """

    def __init__(self, lease: TokenLease) -> None:
        self._lease = lease
        self._token = LeasedToken(access_token="", account_id="", expires_at=0.0)
        self._lock = asyncio.Lock()

    async def token(self) -> tuple[str, str]:
        """Return an access token good for at least five more minutes.

        Returns:
          token: The token lent last, or a newer one when it was about to expire.
          account_id: The ChatGPT account the token is for.

        """
        async with self._lock:
            # Five minutes early, so a token never expires mid-request.
            if time.time() > self._token.expires_at - 300.0:
                self._token = await self._lease(rejected="")
        return self._token.access_token, self._token.account_id

    async def renew(self) -> None:
        """Borrow a token other than the one the server just refused."""
        # Named before the wait: a call that borrowed while this one waited
        # holds a token the lender does not see as refused, so it lends that.
        rejected = self._token.access_token
        async with self._lock:
            if self._token.access_token == rejected:
                self._token = await self._lease(rejected=rejected)

    async def sign_in(
        self,
        *,
        headless: bool = False,
        show: Callable[[str], object] = print,
    ) -> None:
        """Refuse: a borrowed login is signed in by the process lending it.

        Args:
          headless: Unused; there is no browser to skip.
          show: Unused; there is nothing to show.

        Raises:
          AuthError: Always.

        """
        del headless, show
        raise AuthError("This ChatGPT login is lent by another process; sign in there.")
