"""Tests for signing in to MCP servers, against a real OAuth-protected server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast, override
from urllib.parse import parse_qs, urlencode, urlparse

import asyncio
import logging
import re
import secrets
import socket
import time

from mcp import StdioServerParameters
from mcp.client.session_group import StreamableHttpParameters
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    IdentityAssertionParams,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions
from mcp.server.mcpserver import MCPServer
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyHttpUrl, AnyUrl
from treekle import json_dumps, json_loads

import httpx2
import pytest
import uvicorn

from sagent.lib.userdirs import config_dir
from sagent.model.exceptions import AuthError
from sagent.tools import mcp_auth
from sagent.tools.mcp import McpServer, McpSession, login
from sagent.tools.mcp_auth import McpTokenStorage
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest.mark.asyncio
async def test_tokens_and_client_are_kept_per_server_in_a_private_file() -> None:
    storage = McpTokenStorage("web")
    assert await storage.get_tokens() is None
    assert await storage.get_client_info() is None
    token = OAuthToken(
        access_token="a1",  # noqa: S106 -- A test's token, no secret.
        token_type="Bearer",  # noqa: S106 -- The scheme's name, no secret.
        refresh_token="r1",  # noqa: S106 -- A test's token, no secret.
    )
    client = OAuthClientInformationFull(
        client_id="c1",
        redirect_uris=[AnyUrl("http://127.0.0.1:1/callback")],
    )
    await storage.set_tokens(token)
    await storage.set_client_info(client)
    again = McpTokenStorage("web")
    assert await again.get_tokens() == token
    assert await again.get_client_info() == client
    assert await McpTokenStorage("other").get_tokens() is None
    path = config_dir() / "rekursiv-ai" / "secrets" / "mcp" / "web.json"
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.asyncio
async def test_two_changes_at_once_keep_both() -> None:
    """Each change reads the file, then writes it; neither may write over the other."""
    storage = McpTokenStorage("web")
    token = OAuthToken(
        access_token="a1",  # noqa: S106 -- A test's token, no secret.
        token_type="Bearer",  # noqa: S106 -- The scheme's name, no secret.
    )
    client = OAuthClientInformationFull(
        client_id="c1",
        redirect_uris=[AnyUrl("http://127.0.0.1:1/callback")],
    )
    await asyncio.gather(storage.set_tokens(token), storage.set_client_info(client))
    assert await storage.get_tokens() == token
    assert await storage.get_client_info() == client


@pytest.mark.asyncio
async def test_a_change_waits_for_another_task_holding_the_file() -> None:
    """Only the task holding the lock may write without taking it; any other waits."""
    storage = McpTokenStorage("web")
    token = OAuthToken(
        access_token="a1",  # noqa: S106 -- A test's token, no secret.
        token_type="Bearer",  # noqa: S106 -- The scheme's name, no secret.
    )
    async with storage.locked():
        writing = asyncio.create_task(storage.set_tokens(token))
        await asyncio.sleep(0.05)
        assert not writing.done()
        async with storage.locked():
            assert await storage.get_tokens() is None
    await writing
    assert await storage.get_tokens() == token


@pytest.mark.asyncio
async def test_sessions_open_at_once_renew_once_and_share_the_renewal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two sessions, as two processes hold them, outlive one token between them.

    The authority hands out a new refresh token with each renewal and refuses
    the old one, as Atlassian's does; a session renewing with the token it read
    at its start, after another session renewed it, is refused.
    """
    authority = _Authority(lifetime_sec=1)
    async with _protected(authority) as url:
        server = McpServer(name="web", parameters=StreamableHttpParameters(url=url))
        await _sign_in(server, monkeypatch=monkeypatch)
        async with McpSession([server]) as first, McpSession([server]) as second:
            await asyncio.sleep(1.1)
            for session in (first, second):
                (greet,) = session.tools()
                result = await greet.call(
                    ToolContext(cwd=config_dir()),
                    {"name": "Ada"},
                )
                assert result.text() == "hello Ada"
    assert authority.renewed == 1


@pytest.mark.asyncio
async def test_a_server_needing_a_sign_in_is_skipped_until_one_is_made(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with _protected() as url:
        server = McpServer(name="web", parameters=StreamableHttpParameters(url=url))
        with caplog.at_level(logging.WARNING):
            async with McpSession([server]) as session:
                assert session.tools() == []
        assert "run `sagent mcp login web`" in caplog.text
        browser = _Browser()
        monkeypatch.setattr(mcp_auth, "open_browser", browser.open)
        shown: list[str] = []
        await login(server, show=shown.append)
        assert len(browser.opened) == 1
        assert browser.opened[0] in shown[0]
        async with McpSession([server]) as session:
            (greet,) = session.tools()
            result = await greet.call(ToolContext(cwd=config_dir()), {"name": "Ada"})
    assert greet.name == "mcp__web__greet"
    assert result.text() == "hello Ada"


@pytest.mark.asyncio
async def test_a_session_with_no_sign_in_registers_no_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    authority = _Authority()
    async with _protected(authority) as url:
        server = McpServer(name="web", parameters=StreamableHttpParameters(url=url))
        with caplog.at_level(logging.WARNING):
            async with McpSession([server]) as session:
                assert session.tools() == []
    assert authority.clients == {}
    assert not McpTokenStorage(server.key).path.exists()
    assert "run `sagent mcp login web`" in caplog.text


@pytest.mark.asyncio
async def test_an_expired_token_is_renewed_by_a_later_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _Authority()
    async with _protected(authority) as url:
        server = McpServer(name="web", parameters=StreamableHttpParameters(url=url))
        await _sign_in(server, monkeypatch=monkeypatch)
        _expire(authority, server)
        async with McpSession([server]) as session:
            assert [t.name for t in session.tools()] == ["mcp__web__greet"]
    assert authority.renewed == 1


@pytest.mark.asyncio
async def test_a_sign_in_the_server_has_revoked_asks_for_another(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    authority = _Authority()
    async with _protected(authority) as url:
        server = McpServer(name="web", parameters=StreamableHttpParameters(url=url))
        await _sign_in(server, monkeypatch=monkeypatch)
        authority.tokens.clear()
        authority.refresh_tokens.clear()
        with caplog.at_level(logging.WARNING):
            async with McpSession([server]) as session:
                assert session.tools() == []
    assert "run `sagent mcp login web`" in caplog.text
    assert len(authority.clients) == 1


@pytest.mark.asyncio
async def test_signing_in_again_keeps_the_token_renewable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _Authority()
    async with _protected(authority) as url:
        server = McpServer(name="web", parameters=StreamableHttpParameters(url=url))
        await _sign_in(server, monkeypatch=monkeypatch)
        await _sign_in(server, monkeypatch=monkeypatch)
        _expire(authority, server)
        async with McpSession([server]) as session:
            assert [t.name for t in session.tools()] == ["mcp__web__greet"]
    assert authority.renewed == 1


@pytest.mark.asyncio
async def test_an_abandoned_second_sign_in_keeps_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _Authority()
    async with _protected(authority) as url:
        server = McpServer(name="web", parameters=StreamableHttpParameters(url=url))
        await _sign_in(server, monkeypatch=monkeypatch)
        kept = McpTokenStorage(server.key).path.read_text()
        monkeypatch.setattr("builtins.input", _no_answer)
        with pytest.raises(EOFError):
            await login(server, show=list[str]().append, headless=True)
        assert McpTokenStorage(server.key).path.read_text() == kept
        _expire(authority, server)
        async with McpSession([server]) as session:
            assert [t.name for t in session.tools()] == ["mcp__web__greet"]


@pytest.mark.asyncio
async def test_sagent_registers_as_a_public_client_that_may_renew(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _Authority()
    async with _protected(authority) as url:
        server = McpServer(name="web", parameters=StreamableHttpParameters(url=url))
        paste = _Paste()
        monkeypatch.setattr("builtins.input", paste)
        await login(server, show=paste.shown.append, headless=True)
    (client,) = authority.clients.values()
    assert client.client_name == "sagent"
    assert client.grant_types == ["authorization_code", "refresh_token"]
    assert client.response_types == ["code"]
    assert client.token_endpoint_auth_method == "none"  # noqa: S105 -- A method's name.
    (redirect,) = client.redirect_uris or []
    assert re.fullmatch(r"http://127\.0\.0\.1:\d+/callback", str(redirect))


@pytest.mark.asyncio
async def test_a_headless_sign_in_takes_the_pasted_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _protected() as url:
        server = McpServer(name="web", parameters=StreamableHttpParameters(url=url))
        paste = _Paste()
        monkeypatch.setattr(mcp_auth, "open_browser", _never)
        monkeypatch.setattr("builtins.input", paste)
        await login(server, show=paste.shown.append, headless=True)
        assert paste.shown[-1] == (
            "Open it in a browser anywhere, then paste the address it was sent"
            " to, though it fails to load:\n"
        )
        async with McpSession([server]) as session:
            assert [t.name for t in session.tools()] == ["mcp__web__greet"]


@pytest.mark.asyncio
async def test_a_pasted_address_of_another_sign_in_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _protected() as url:
        server = McpServer(name="web", parameters=StreamableHttpParameters(url=url))
        paste = _Paste(fixed="http://127.0.0.1:1/callback?code=c&state=other")
        monkeypatch.setattr("builtins.input", paste)
        with pytest.raises(AuthError, match="belongs to another sign-in"):
            await login(server, show=paste.shown.append, headless=True)


@pytest.mark.asyncio
async def test_a_pasted_address_with_no_code_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _protected() as url:
        server = McpServer(name="web", parameters=StreamableHttpParameters(url=url))
        paste = _Paste(drop_code=True)
        monkeypatch.setattr("builtins.input", paste)
        with pytest.raises(AuthError, match="holds no sign-in code"):
            await login(server, show=paste.shown.append, headless=True)


@pytest.mark.asyncio
async def test_nothing_pasted_says_it_holds_no_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _protected() as url:
        server = McpServer(name="web", parameters=StreamableHttpParameters(url=url))
        paste = _Paste(fixed=" ")
        monkeypatch.setattr("builtins.input", paste)
        with pytest.raises(AuthError, match="holds no sign-in code"):
            await login(server, show=paste.shown.append, headless=True)


@pytest.mark.asyncio
async def test_a_session_never_asks_for_the_code_it_cannot_get() -> None:
    needed = mcp_auth._SignInNeeded(name="web")
    with pytest.raises(AuthError, match="MCP server 'web' needs a sign-in"):
        await needed.callback()


@pytest.mark.asyncio
async def test_a_browser_that_never_comes_back_gives_way_to_a_paste(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _protected() as url:
        server = McpServer(name="web", parameters=StreamableHttpParameters(url=url))
        paste = _Paste()
        monkeypatch.setattr(mcp_auth, "open_browser", _never)
        monkeypatch.setattr("builtins.input", paste)
        await login(server, show=paste.shown.append, wait_sec=0.05)
        assert "No browser came back" in paste.shown[-1]
        assert await McpTokenStorage(server.key).get_tokens() is not None


@pytest.mark.asyncio
async def test_only_a_remote_server_signs_in() -> None:
    local = McpServer(name="git", parameters=StdioServerParameters(command="x"))
    with pytest.raises(ValueError, match="MCP server 'git' runs locally"):
        await login(local, show=list[str]().append)


def _never(url: str) -> None:
    del url


def _no_answer(prompt: str) -> str:
    del prompt
    raise EOFError


async def _sign_in(server: McpServer, *, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sign in to ``server`` headless, pasting where its sign-in page lands."""
    paste = _Paste()
    monkeypatch.setattr("builtins.input", paste)
    await login(server, show=paste.shown.append, headless=True)


def _expire(authority: _Authority, server: McpServer) -> None:
    """Expire every access token, at the authority and in the file sagent keeps."""
    authority.tokens.clear()
    path = McpTokenStorage(server.key).path
    kept = json_loads(path.read_text(), dict[str, object])
    path.write_text(json_dumps({**kept, "expires_at": time.time() - 1}))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Paste:
    """The user at ``input``: pastes where the sign-in page shown lands, or ``fixed``."""

    shown: list[str] = field(default_factory=list[str])
    fixed: str = ""
    drop_code: bool = False
    """Paste the address the page lands on without its ``code``."""

    def __call__(self, prompt: str) -> str:
        del prompt
        if self.fixed:
            return self.fixed
        (url,) = (w for w in self.shown[0].split() if w.startswith("http"))
        with httpx2.Client() as http:
            landed = http.get(url).headers["location"]
        if self.drop_code:
            query = parse_qs(urlparse(landed).query)
            return (
                urlparse(landed)
                ._replace(query=urlencode({"state": query["state"][0]}))
                .geturl()
            )
        return landed


@dataclass(slots=True, kw_only=True)
class _Browser:
    """Visits what it is asked to open, following redirects to the sign-in's own."""

    opened: list[str] = field(default_factory=list[str])
    visits: set[asyncio.Task[None]] = field(default_factory=set[asyncio.Task[None]])

    def open(self, url: str) -> None:
        self.opened.append(url)
        task = asyncio.get_running_loop().create_task(self._visit(url))
        self.visits.add(task)
        task.add_done_callback(self.visits.discard)

    async def _visit(self, url: str) -> None:
        async with httpx2.AsyncClient(follow_redirects=True) as http:
            await http.get(url)


@dataclass(slots=True, kw_only=True)
class _Authority:
    """An OAuth server that approves every request at once, all in memory."""

    clients: dict[str, OAuthClientInformationFull] = field(
        default_factory=dict[str, OAuthClientInformationFull],
    )
    codes: dict[str, AuthorizationCode] = field(
        default_factory=dict[str, AuthorizationCode],
    )
    tokens: dict[str, AccessToken] = field(default_factory=dict[str, AccessToken])
    refresh_tokens: dict[str, RefreshToken] = field(
        default_factory=dict[str, RefreshToken],
    )
    renewed: int = 0
    """How many refresh tokens were exchanged."""

    lifetime_sec: int = 3600
    """How long the access token a sign-in issues is good for.

    A renewed access token, and every refresh token, is good for an hour: a short
    lifetime then makes the sign-in's token expire, while two sessions that share
    one renewal both outlive it however slowly the second one calls.
    """

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        assert client_info.client_id is not None
        self.clients[client_info.client_id] = client_info

    async def authorize(
        self,
        client: OAuthClientInformationFull,
        params: AuthorizationParams,
    ) -> str:
        code = secrets.token_hex(8)
        assert client.client_id is not None
        self.codes[code] = AuthorizationCode(
            code=code,
            scopes=params.scopes or [],
            expires_at=time.time() + 300,
            client_id=client.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
        )
        return construct_redirect_uri(
            str(params.redirect_uri),
            code=code,
            state=params.state,
        )

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> AuthorizationCode | None:
        del client
        return self.codes.get(authorization_code)

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        del self.codes[authorization_code.code]
        return self._issue(
            client,
            scopes=authorization_code.scopes,
            resource=authorization_code.resource,
            lifetime_sec=self.lifetime_sec,
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        return self.tokens.get(token)

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> RefreshToken | None:
        del client
        return self.refresh_tokens.get(refresh_token)

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        del self.refresh_tokens[refresh_token.token]
        self.renewed += 1
        return self._issue(
            client,
            scopes=scopes or refresh_token.scopes,
            resource=refresh_token.resource,
            lifetime_sec=3600,
        )

    def _issue(
        self,
        client: OAuthClientInformationFull,
        *,
        scopes: list[str],
        resource: str | None,
        lifetime_sec: int,
    ) -> OAuthToken:
        """Issue an access token good for ``lifetime_sec``, and a refresh token."""
        assert client.client_id is not None
        access, refresh = secrets.token_hex(8), secrets.token_hex(8)
        now = int(time.time())
        self.tokens[access] = AccessToken(
            token=access,
            client_id=client.client_id,
            scopes=scopes,
            expires_at=now + lifetime_sec,
            resource=resource,
        )
        self.refresh_tokens[refresh] = RefreshToken(
            token=refresh,
            client_id=client.client_id,
            scopes=scopes,
            expires_at=now + 3600,
            resource=resource,
        )
        return OAuthToken(
            access_token=access,
            token_type="Bearer",  # noqa: S106 -- The scheme's name, no secret.
            expires_in=lifetime_sec,
            refresh_token=refresh,
        )

    async def exchange_identity_assertion(
        self,
        client: OAuthClientInformationFull,
        params: IdentityAssertionParams,
    ) -> OAuthToken:
        del client, params
        raise NotImplementedError

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        del token


class _Serving(uvicorn.Server):
    """A uvicorn server that says when it is listening."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.listening = asyncio.Event()

    @override
    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets)
        self.listening.set()


@asynccontextmanager
async def _protected(authority: _Authority | None = None) -> AsyncGenerator[str]:
    """Serve an MCP server that needs a sign-in from ``authority``; yield its URL."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        # Typeshed types an address as ``Any``; an IPv4 one is ``(host, port)``.
        _, port = cast(tuple[str, int], probe.getsockname())
    base = f"http://127.0.0.1:{port}"
    app = MCPServer(
        "web",
        auth_server_provider=authority if authority is not None else _Authority(),
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(base),
            resource_server_url=AnyHttpUrl(f"{base}/mcp"),
            validate_token_resource=True,
            client_registration_options=ClientRegistrationOptions(enabled=True),
        ),
    )

    @app.tool()
    def greet(name: str) -> str:
        """Say hello."""
        return f"hello {name}"

    server = _Serving(
        uvicorn.Config(
            app.streamable_http_app(),
            host="127.0.0.1",
            port=port,
            log_level="critical",
            lifespan="on",
        ),
    )
    serving = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(10):
            await server.listening.wait()
        yield f"{base}/mcp"
    finally:
        server.should_exit = True
        await serving


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
