"""Signing in to remote MCP servers with OAuth, and keeping what the sign-in gives.

A remote server may answer ``401`` and name an OAuth authority, as the MCP
spec says. ``sagent mcp login <server>`` (``/mcp.login`` at the prompt) then
registers sagent as a client,
opens the authority's page in the browser, takes the redirect back on a
loopback port, and keeps the client, its tokens, and when they expire in
``config_dir() / "rekursiv-ai" / "secrets" / "mcp" / "<key>.json"``, the
server's ``McpServer.key``, private to the user. The sign-in runs against nothing saved and saves only on
success, so signing in again, or giving up part way, never leaves tokens
without their client. Every later connection sends the tokens and renews them
before they expire; one that finds none sends none, and a server that
refuses it says to sign in, rather than opening a browser mid-session.
``sagent mcp logout <server>`` forgets them.

Typical usage example:

  http = create_mcp_http_client(auth=await session_auth("linear", url, key=key))
  async with login_auth(url, key=key, show=print) as auth:
      ...  # Connect with ``auth``; the server's ``401`` runs the sign-in.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager, nullcontext, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, NotRequired, TypedDict, cast, override
from urllib.parse import parse_qs, urlparse

import asyncio
import time

from treekle import json_dumps, json_loads

import httpx2

from sagent.lib.files.owner_only import write_private
from sagent.lib.userdirs import config_dir
from sagent.model.exceptions import AuthError
from sagent.providers.login import (
    file_lock,
    open_browser,
    read_pasted_code,
    redirected,
)


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import (
        AuthorizationCodeResult,
        OAuthClientInformationFull,
        OAuthClientMetadata,
        OAuthToken,
    )
    from pydantic import AnyUrl
else:
    from wrapt import lazy_import

    # ``mcp`` takes ~450 ms to import; a session with no server never pays it.
    AnyUrl = lazy_import("pydantic", "AnyUrl")
    OAuthClientProvider = lazy_import("mcp.client.auth", "OAuthClientProvider")
    AuthorizationCodeResult = lazy_import("mcp.shared.auth", "AuthorizationCodeResult")
    OAuthClientInformationFull = lazy_import(
        "mcp.shared.auth",
        "OAuthClientInformationFull",
    )
    OAuthClientMetadata = lazy_import("mcp.shared.auth", "OAuthClientMetadata")
    OAuthToken = lazy_import("mcp.shared.auth", "OAuthToken")


__all__ = [
    "McpTokenStorage",
    "login_auth",
    "session_auth",
]


_HELD = ContextVar[frozenset[tuple[Path, asyncio.Task[object] | None]]](
    "held",
    default=frozenset(),
)
"""Each token file whose lock is held, with the task holding it."""


@dataclass(slots=True, kw_only=True)
class _MemoryStorage:
    """A sign-in's tokens and client, held until the sign-in succeeds."""

    tokens: OAuthToken | None = None
    client: OAuthClientInformationFull | None = None
    expires_at: float | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.tokens = tokens
        self.expires_at = _expiry(tokens)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self.client

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.client = client_info


class McpTokenStorage:
    """One server's tokens, their expiry, and its client, in a file private to the user.

    The three change together: a sign-in saves them in one write, and a renewal
    replaces the tokens and their expiry under the same client.

    Args:
      key: The server's ``McpServer.key``, which names its file.

    """

    def __init__(self, key: str) -> None:
        self.path = config_dir() / "rekursiv-ai" / "secrets" / "mcp" / f"{key}.json"

    async def get_tokens(self) -> OAuthToken | None:
        """Return the tokens kept; ``None`` before a sign-in."""
        return (await self.read()).tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Keep renewed ``tokens`` and when they expire."""
        async with self.locked():
            kept = await self.read()
            kept.tokens, kept.expires_at = tokens, _expiry(tokens)
            await self._write(kept)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Return the client the server registered; ``None`` before a sign-in."""
        return (await self.read()).client

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Keep the client the server registered."""
        async with self.locked():
            kept = await self.read()
            kept.client = client_info
            await self._write(kept)

    async def read(self) -> _MemoryStorage:
        """Return everything kept; nothing before a sign-in."""
        return _parse(await asyncio.to_thread(_load, self.path))

    async def save(self, kept: _MemoryStorage) -> None:
        """Replace everything kept with ``kept``, in one write.

        Args:
          kept: The client, its tokens, and when they expire.

        """
        async with self.locked():
            await self._write(kept)

    # A lock taken again by the task holding it waits on itself forever: ``flock``
    # sees each open of the lock file as another holder. A renewal holds it while
    # the SDK saves, so the saving must not take it again.
    @asynccontextmanager
    async def locked(self) -> AsyncGenerator[None]:
        """Hold the file's lock against every other task and process; reentrant.

        Yields:
          nothing: The lock is held until the block ends.

        """
        held, mine = _HELD.get(), (self.path, asyncio.current_task())
        # A task started inside the block copies the context, so the task is kept too.
        if mine in held:
            yield
            return
        async with file_lock(self.path.with_name(f"{self.path.name}.lock")):
            token = _HELD.set(held | {mine})
            try:
                yield
            finally:
                _HELD.reset(token)

    async def _write(self, kept: _MemoryStorage) -> None:
        await asyncio.to_thread(
            write_private,
            self.path,
            json_dumps(dict(_plain(kept))),
        )


async def session_auth(name: str, url: str, *, key: str) -> httpx2.Auth:
    """Return the auth a session's requests to server ``name`` at ``url`` carry.

    Args:
      name: The server's configured name, which messages name it by.
      url: Its endpoint.
      key: Its ``McpServer.key``, which names its tokens' file.

    Returns:
      auth: Sends the tokens kept and renews them before they expire. Before a
        sign-in it sends none, and a server's ``401`` fails the connection
        saying how to sign in, so a session never registers a client or opens
        a browser.

    """
    storage = McpTokenStorage(key)
    kept = await storage.read()
    needed = _SignInNeeded(name=name)
    if kept.tokens is None or kept.client is None:
        return needed
    auth = OAuthClientProvider(
        server_url=url,
        client_metadata=_metadata("http://127.0.0.1/callback"),
        storage=storage,
        redirect_handler=needed.redirect,
        callback_handler=needed.callback,
    )
    # The SDK learns a token's expiry only from the exchange that issued it, and
    # without one it sends an expired token until the server refuses it.
    auth.context.token_expiry_time = kept.expires_at
    return _Shared(auth=auth, storage=storage)


@asynccontextmanager
async def login_auth(
    url: str,
    *,
    key: str,
    show: Callable[[str], object],
    headless: bool = False,
    wait_sec: float = 300.0,
) -> AsyncGenerator[OAuthClientProvider]:
    """Yield auth that signs in to the server at ``url`` when it is asked to.

    The loopback port listens until the block ends.

    Args:
      url: Its endpoint.
      key: Its ``McpServer.key``, which names its tokens' file.
      show: Shows the user the sign-in page and what to do.
      headless: Open no browser; the user opens the page anywhere and pastes
        the address it lands on.
      wait_sec: How long the browser has to come back before a paste is asked.

    Yields:
      auth: An httpx2 ``auth=`` that runs the whole sign-in on the server's
        ``401``; what it signs in with is saved once the block ends without
        raising, so a failed sign-in leaves the one before it kept.

    """
    sign_in = _SignIn(show=show, headless=headless, wait_sec=wait_sec)
    # Loopback alone: a sign-in code must not be offered to the network.
    listener = await asyncio.start_server(sign_in.answer, "127.0.0.1")
    # Typeshed types an address as ``Any``; an IPv4 one is ``(host, port)``.
    _, port = cast(tuple[str, int], listener.sockets[0].getsockname())
    # The literal address, not ``localhost``: that can resolve to ``::1``,
    # which this IPv4 listener never hears.
    redirect = f"http://127.0.0.1:{port}/callback"
    # Empty, so the server's ``401`` runs every step: a client registered for this
    # port, then tokens issued to it.
    pending = _MemoryStorage()
    async with listener:
        yield OAuthClientProvider(
            server_url=url,
            client_metadata=_metadata(redirect),
            storage=pending,
            redirect_handler=sign_in.redirect,
            callback_handler=sign_in.callback,
        )
    await McpTokenStorage(key).save(pending)


@dataclass(slots=True, kw_only=True)
class _SignIn:
    """One sign-in: the page shown, then the code the browser or the user brings back."""

    show: Callable[[str], object]
    headless: bool
    wait_sec: float
    state: str = ""
    code: asyncio.Future[str] = field(default_factory=asyncio.Future[str])

    async def answer(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Answer the browser's redirect to the loopback port.

        Args:
          reader: The browser's request.
          writer: Where the answer goes.

        """
        await redirected(
            reader,
            writer,
            path="/callback",
            state=self.state,
            code=self.code,
        )

    async def redirect(self, url: str) -> None:
        """Show the authority's page; open it in the browser unless headless.

        Args:
          url: The page, whose ``state`` the redirect back must carry.

        """
        # The SDK puts a fresh ``state`` in every authorize URL it builds.
        self.state = parse_qs(urlparse(url).query)["state"][0]
        self.show(f"Sign in to the MCP server in your browser:\n\n  {url}\n")
        if not self.headless:
            open_browser(url)

    async def callback(self) -> AuthorizationCodeResult:
        """Return the code the browser brings back, or the user pastes.

        Returns:
          result: The code, with this sign-in's ``state``.

        Raises:
          AuthError: The pasted address is another sign-in's, or holds no code.

        """
        if not self.headless:
            # Over ssh the browser cannot reach us; its address bar holds the code.
            with suppress(TimeoutError):
                code = await asyncio.wait_for(asyncio.shield(self.code), self.wait_sec)
                return AuthorizationCodeResult(code=code, state=self.state)
        code = await read_pasted_code(
            state=self.state,
            show=self.show,
            browser_tried=not self.headless,
        )
        return AuthorizationCodeResult(code=code, state=self.state)


# The SDK reads the file once and renews with the refresh token it read then. A
# server that hands out a new refresh token with each renewal refuses the old one,
# so a session that renews after another did would lose its sign-in.
class _Shared(httpx2.Auth):
    """A session's auth that renews as one with every other session of its server.

    Args:
      auth: The SDK's auth, which sends and renews the tokens.
      storage: The server's file, which every session shares.

    """

    def __init__(self, *, auth: OAuthClientProvider, storage: McpTokenStorage) -> None:
        self.auth = auth
        self.storage = storage
        self._renewal = asyncio.Lock()

    @override
    async def async_auth_flow(
        self,
        request: httpx2.Request,
    ) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        renews = not self.auth.context.is_token_valid()
        async with self._renewing() if renews else nullcontext():
            flow = self.auth.async_auth_flow(request)
            try:
                sent = await anext(flow)
                while True:
                    sent = await flow.asend((yield sent))  # codespell:ignore asend
            except StopAsyncIteration:
                return
            finally:
                await flow.aclose()

    @asynccontextmanager
    async def _renewing(self) -> AsyncGenerator[None]:
        """Hold the server's file, and take the tokens another session may have kept."""
        async with self._renewal, self.storage.locked():
            kept = await self.storage.read()
            if kept.tokens is not None:
                self.auth.context.current_tokens = kept.tokens
                self.auth.context.token_expiry_time = kept.expires_at
            yield


class _SignInNeeded(httpx2.Auth):
    """A session's stand-in for a sign-in: it says how to make one.

    It is the auth of a session with nothing saved, and the redirect a saved
    sign-in falls back on when its tokens are refused.

    Args:
      name: The server's configured name.

    """

    def __init__(self, *, name: str) -> None:
        self.name = name

    @override
    def auth_flow(
        self,
        request: httpx2.Request,
    ) -> Generator[httpx2.Request, httpx2.Response]:
        response = yield request
        if response.status_code == 401:
            self._refuse()

    async def redirect(self, url: str) -> None:
        del url
        self._refuse()

    async def callback(self) -> AuthorizationCodeResult:
        # The SDK asks for the code only after the redirect, which raised.
        self._refuse()

    def _refuse(self) -> NoReturn:
        raise AuthError(
            f"MCP server {self.name!r} needs a sign-in: run"
            f" `sagent mcp login {self.name}`.",
        )


def _metadata(redirect: str) -> OAuthClientMetadata:
    """Return the client sagent registers as: public, with PKCE, redirected to us."""
    return OAuthClientMetadata(
        client_name="sagent",
        redirect_uris=[AnyUrl(redirect)],
        token_endpoint_auth_method="none",  # noqa: S106 -- Flagged for the keyword's name; the value is OAuth's public-client method.
    )


class _Kept(TypedDict):
    """What a server's file holds, as JSON."""

    tokens: NotRequired[dict[str, object]]
    client: NotRequired[dict[str, object]]
    expires_at: NotRequired[float]


def _expiry(tokens: OAuthToken) -> float | None:
    """Return when ``tokens`` expire, in seconds since the epoch; ``None`` for never."""
    return None if tokens.expires_in is None else time.time() + tokens.expires_in


def _parse(kept: _Kept) -> _MemoryStorage:
    tokens, client = kept.get("tokens"), kept.get("client")
    return _MemoryStorage(
        tokens=OAuthToken.model_validate(tokens) if tokens is not None else None,
        client=(
            OAuthClientInformationFull.model_validate(client)
            if client is not None
            else None
        ),
        expires_at=kept.get("expires_at"),
    )


def _plain(kept: _MemoryStorage) -> _Kept:
    plain: _Kept = {}
    if kept.tokens is not None:
        plain["tokens"] = kept.tokens.model_dump(mode="json", exclude_none=True)
    if kept.client is not None:
        plain["client"] = kept.client.model_dump(mode="json", exclude_none=True)
    if kept.expires_at is not None:
        plain["expires_at"] = kept.expires_at
    return plain


def _load(path: Path) -> _Kept:
    """Return what ``path`` keeps; nothing when it does not exist."""
    if not path.is_file():
        return {}
    return json_loads(path.read_text(encoding="utf-8"), _Kept)
