"""A short-lived vendor token in place of a stored API key.

A workload proves who it is with a token from its own identity provider, and the vendor
swaps that for a short-lived token of its own (workload identity). Nothing
long-lived is stored, so nothing long-lived can leak.

A command prints the identity token, so this knows no cloud: on AWS it is
``aws sts get-web-identity-token``. It runs once per exchange, because a vendor may refuse
an identity token it has seen.

Typical usage example:

  identity = WorkloadIdentity(
      resources=shared,
      url="https://auth.example.com/oauth/token",
      command="print-identity-token",
      request=lambda subject: {"subject_token": subject},
  )
  headers = {"authorization": f"Bearer {await identity.token()}"}
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import asyncio
import shlex
import time

from treekle import from_plain, json_loads

from sagent.model.exceptions import AuthError


if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx2


__all__ = [
    "WorkloadIdentity",
]


class _Http(Protocol):
    """What this needs of a provider's resources."""

    @property
    def http(self) -> httpx2.AsyncClient:
        """Return the HTTP client."""
        ...


class WorkloadIdentity:
    """One vendor token, swapped for an identity token and renewed before it expires.

    Args:
      resources: Where the HTTP client comes from.
      url: The vendor's token endpoint.
      command: Prints an identity token. Split as a shell would, and run with no shell.
      request: Builds the exchange request from an identity token.
      clock: Seconds, for measuring a token's age.
      command_timeout_sec: How long the command has.
      exchange_timeout_sec: How long the vendor has. Every request of this provider
        waits on the exchange, so it can't have the HTTP client's stream allowance.
      renew_margin_sec: How long before expiry a token is replaced, so no request starts
        on a dying one.

    """

    def __init__(
        self,
        *,
        resources: _Http,
        url: str,
        command: str,
        request: Callable[[str], dict[str, object]],
        clock: Callable[[], float] = time.monotonic,
        command_timeout_sec: float = 30.0,
        exchange_timeout_sec: float = 30.0,
        renew_margin_sec: float = 60.0,
    ) -> None:
        self._resources = resources
        self._url = url
        self._command = command
        self._request = request
        self._clock = clock
        self._command_timeout_sec = command_timeout_sec
        self._exchange_timeout_sec = exchange_timeout_sec
        self._renew_margin_sec = renew_margin_sec
        self._lock = asyncio.Lock()
        self._token = ""
        self._expires = 0.0

    async def token(self) -> str:
        """Return a live vendor token, exchanging for a new one when needed.

        Returns:
          token: The bearer token.

        Raises:
          AuthError: The command failed, or the vendor refused its token.

        """
        async with self._lock:
            if (
                not self._token
                or self._clock() >= self._expires - self._renew_margin_sec
            ):
                await self._exchange()
            return self._token

    def forget(self) -> None:
        """Drop the token, so the next request exchanges for a new one."""
        self._token = ""

    async def _exchange(self) -> None:
        """Swap a fresh identity token for a vendor token."""
        started = self._clock()
        response = await self._resources.http.post(
            self._url,
            json=self._request(await self._identity()),
            timeout=self._exchange_timeout_sec,
        )
        if response.status_code != 200:
            raise AuthError(
                f"{self._url} refused the identity token: HTTP {response.status_code}.",
            )
        answer = json_loads(response.content, dict[str, object])
        token = from_plain(answer.get("access_token"), str, default="")
        if not token:
            raise AuthError(f"{self._url} returned no access token.")
        self._token = token
        self._expires = started + from_plain(
            answer.get("expires_in"),
            float,
            default=0.0,
        )

    async def _identity(self) -> str:
        """Return what the command prints: a fresh identity token."""
        try:
            process = await asyncio.create_subprocess_exec(
                *shlex.split(self._command),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as error:
            raise AuthError(f"Could not run the identity command: {error}.") from error
        try:
            async with asyncio.timeout(self._command_timeout_sec):
                output, _ = await process.communicate()
        except TimeoutError:
            process.kill()
            await process.wait()
            raise AuthError("The identity command timed out.") from None
        identity = output.decode(errors="replace").strip()
        if process.returncode != 0 or not identity:
            raise AuthError(f"The identity command failed: {process.returncode}.")
        return identity
