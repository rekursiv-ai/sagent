"""Tests for the Claude subscription login, against a fake OAuth server."""

from __future__ import annotations

from asyncio.subprocess import Process
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast, override
from unittest.mock import AsyncMock, Mock, call
from urllib.parse import parse_qs, urlparse

import asyncio
import base64
import getpass
import hashlib
import re
import stat

from treekle import ReadError, from_plain, json_dumps, json_loads

import httpx2
import pytest

from sagent.lib.posix.child import Child
from sagent.model.exceptions import AuthError
from sagent.providers.anthropic import login
from sagent.providers.anthropic.login import ClaudeLogin
from sagent.providers.resources import SharedResources
from sagent.providers.testing import refuse_browsers


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


@pytest.fixture(autouse=True)
def no_real_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test that opens the user's browser, as a sign-in not headless would."""
    refuse_browsers(monkeypatch, login)


_GRANT_URL = "https://platform.claude.com/v1/oauth/token"
_PROFILE_URL = "https://api.anthropic.com/api/oauth/profile"
_NOW = 1_800_000_000.0
_REVOKED = r"^the Claude login has expired or been revoked; sign in again$"


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Stop the login's clock at ``_NOW``; a test moves it by changing ``[0]``."""
    now = [_NOW]
    monkeypatch.setattr(login, "time", SimpleNamespace(time=lambda: now[0]))
    return now


@pytest.fixture
def authorization(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Supply a browser code at the authorization boundary."""
    authorized = AsyncMock(return_value=("the-code", "http://localhost:4321/callback"))
    monkeypatch.setattr(login, "_browser_code", authorized)
    return authorized


def _store(
    path: Path,
    *,
    access: str,
    expires_in_sec: float,
    **document: object,
) -> None:
    oauth = {
        "accessToken": access,
        "refreshToken": f"rt-{access}",
        "expiresAt": int((_NOW + expires_in_sec) * 1000),
        "subscriptionType": "max",
    }
    path.write_text(json_dumps({"claudeAiOauth": oauth, **document}))


def _stored(path: Path) -> dict[str, object]:
    return json_loads(path.read_text(), dict[str, object])


def _oauth(path: Path) -> dict[str, object]:
    return from_plain(_stored(path)["claudeAiOauth"], dict[str, object])


def _granted(access: str, *, expires_in_sec: int = 28_800) -> httpx2.Response:
    grant = {
        "access_token": access,
        "refresh_token": f"rt-{access}",
        "expires_in": expires_in_sec,
    }
    return httpx2.Response(200, content=json_dumps(grant).encode())


def _body(request: httpx2.Request) -> dict[str, object]:
    return json_loads(request.content, dict[str, object])


@pytest.mark.asyncio
async def test_a_stored_token_far_from_expiry_needs_no_server(tmp_path: Path) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="at-1", expires_in_sec=3600.0)
    async with _Served(_Server()) as shared:
        assert await ClaudeLogin(path, resources=shared).token() == "at-1"


@pytest.mark.asyncio
async def test_a_token_five_minutes_from_expiry_is_still_good(tmp_path: Path) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="at-1", expires_in_sec=300.0)
    async with _Served(_Server()) as shared:
        assert await ClaudeLogin(path, resources=shared).token() == "at-1"


@pytest.mark.asyncio
async def test_the_default_login_is_the_one_claude_keeps_in_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off macOS, where the Keychain holds it, ``claude`` keeps it in a file."""
    monkeypatch.setattr(login, "platform", SimpleNamespace(system=lambda: "Linux"))
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".claude").mkdir()
    _store(
        tmp_path / ".claude" / ".credentials.json",
        access="at-1",
        expires_in_sec=600.0,
    )
    async with _Served(_Server()) as shared:
        assert await ClaudeLogin(resources=shared).token() == "at-1"


@pytest.mark.asyncio
async def test_a_stored_login_without_a_token_is_renewed(tmp_path: Path) -> None:
    """Its expiry is no promise of a token to hand out."""
    path = tmp_path / ".credentials.json"
    oauth = {"refreshToken": "rt-1", "expiresAt": int((_NOW + 3600) * 1000)}
    path.write_text(json_dumps({"claudeAiOauth": oauth}))
    server = _Server(replies={_GRANT_URL: [_granted("at-2")]})
    async with _Served(server) as shared:
        assert await ClaudeLogin(path, resources=shared).token() == "at-2"
    assert _body(server.requests[0])["refresh_token"] == oauth["refreshToken"]


@pytest.mark.asyncio
async def test_a_credentials_file_that_is_not_an_object_is_an_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / ".credentials.json"
    path.write_text("[]")
    async with _Served(_Server()) as shared:
        with pytest.raises(ReadError):
            await ClaudeLogin(path, resources=shared).token()


@pytest.mark.asyncio
async def test_a_token_about_to_expire_is_renewed_and_stored(tmp_path: Path) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="at-1", expires_in_sec=299.0)
    server = _Server(replies={_GRANT_URL: [_granted("at-2")]})
    async with _Served(server) as shared:
        assert await ClaudeLogin(path, resources=shared).token() == "at-2"
    (sent,) = server.requests
    assert _body(sent) == {
        "grant_type": "refresh_token",
        "refresh_token": "rt-at-1",
        "scope": "user:profile user:inference user:sessions:claude_code"
        " user:mcp_servers user:file_upload",
        "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
    }
    oauth = _oauth(path)
    assert oauth["accessToken"] == "at-2"
    assert oauth["refreshToken"] == "rt-at-2"
    assert oauth["expiresAt"] == int((_NOW + 28_800) * 1000)
    # Every field ``claude`` keeps beside the token survives.
    assert oauth["subscriptionType"] == "max"


@pytest.mark.asyncio
async def test_a_token_renewed_by_another_process_is_adopted(
    tmp_path: Path,
    clock: list[float],
) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="at-1", expires_in_sec=3600.0)
    async with _Served(_Server()) as shared:
        claude_login = ClaudeLogin(path, resources=shared)
        assert await claude_login.token() == "at-1"
        clock[0] += 3500.0
        _store(path, access="at-2", expires_in_sec=7200.0)
        assert await claude_login.token() == "at-2"


@pytest.mark.asyncio
async def test_concurrent_callers_renew_once(tmp_path: Path) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="at-1", expires_in_sec=0.0)
    server = _Server(replies={_GRANT_URL: [_granted("at-2")]})
    async with _Served(server) as shared:
        claude_login = ClaudeLogin(path, resources=shared)
        tokens = await asyncio.gather(claude_login.token(), claude_login.token())
    assert tokens == ["at-2", "at-2"]
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_renew_replaces_a_rejected_token_even_when_fresh(tmp_path: Path) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="at-1", expires_in_sec=3600.0)
    server = _Server(replies={_GRANT_URL: [_granted("at-2")]})
    async with _Served(server) as shared:
        claude_login = ClaudeLogin(path, resources=shared)
        assert await claude_login.token() == "at-1"
        await claude_login.renew()
        assert await claude_login.token() == "at-2"
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_renew_takes_a_token_another_process_already_rotated(
    tmp_path: Path,
) -> None:
    """A refresh token works once; refreshing it again would revoke the login."""
    path = tmp_path / ".credentials.json"
    _store(path, access="at-1", expires_in_sec=3600.0)
    async with _Served(_Server()) as shared:
        claude_login = ClaudeLogin(path, resources=shared)
        assert await claude_login.token() == "at-1"
        _store(path, access="at-2", expires_in_sec=3600.0)
        await claude_login.renew()
        assert await claude_login.token() == "at-2"


@pytest.mark.parametrize("status", [400, 401])
@pytest.mark.asyncio
async def test_a_refused_refresh_asks_to_sign_in_again(
    tmp_path: Path,
    status: int,
) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="at-1", expires_in_sec=0.0)
    server = _Server(replies={_GRANT_URL: [httpx2.Response(status)]})
    async with _Served(server) as shared:
        with pytest.raises(AuthError, match=_REVOKED):
            await ClaudeLogin(path, resources=shared).token()
    assert _oauth(path)["accessToken"] == "at-1"


@pytest.mark.asyncio
async def test_a_grant_that_is_not_an_object_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="at-1", expires_in_sec=0.0)
    server = _Server(replies={_GRANT_URL: [httpx2.Response(200, content=b"[]")]})
    async with _Served(server) as shared:
        with pytest.raises(ReadError):
            await ClaudeLogin(path, resources=shared).token()


@pytest.mark.asyncio
async def test_a_failing_token_server_is_not_a_revoked_login(tmp_path: Path) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="at-1", expires_in_sec=0.0)
    server = _Server(replies={_GRANT_URL: [httpx2.Response(503)]})
    async with _Served(server) as shared:
        with pytest.raises(httpx2.HTTPStatusError):
            await ClaudeLogin(path, resources=shared).token()


@pytest.mark.asyncio
async def test_no_stored_login_asks_to_sign_in(tmp_path: Path) -> None:
    server = _Server(replies={_GRANT_URL: [httpx2.Response(400)]})
    async with _Served(server) as shared:
        claude_login = ClaudeLogin(tmp_path / "missing.json", resources=shared)
        with pytest.raises(AuthError):
            await claude_login.token()
    assert _body(server.requests[0])["refresh_token"] == ""


@pytest.mark.asyncio
async def test_renew_refreshes_a_rotated_token_that_has_expired_too(
    tmp_path: Path,
) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="at-1", expires_in_sec=3600.0)
    server = _Server(replies={_GRANT_URL: [_granted("at-3")]})
    async with _Served(server) as shared:
        claude_login = ClaudeLogin(path, resources=shared)
        assert await claude_login.token() == "at-1"
        _store(path, access="at-2", expires_in_sec=0.0)
        rotated = _oauth(path)["refreshToken"]
        await claude_login.renew()
        assert len(server.requests) == 1
        assert await claude_login.token() == "at-3"
    assert _body(server.requests[0])["refresh_token"] == rotated


@pytest.mark.asyncio
async def test_saving_keeps_the_other_logins_claude_keeps(tmp_path: Path) -> None:
    """``claude`` keeps its MCP servers' logins in the same file."""
    path = tmp_path / ".credentials.json"
    mcp = {"github|1": {"accessToken": "gh"}}
    _store(path, access="at-1", expires_in_sec=0.0, mcpOAuth=mcp)
    server = _Server(replies={_GRANT_URL: [_granted("at-2")]})
    async with _Served(server) as shared:
        await ClaudeLogin(path, resources=shared).token()
    assert _stored(path)["mcpOAuth"] == mcp


@pytest.mark.asyncio
async def test_the_credentials_file_is_never_readable_by_others(tmp_path: Path) -> None:
    path = tmp_path / "new" / ".credentials.json"
    path.parent.mkdir()
    _store(path, access="at-1", expires_in_sec=0.0)
    path.chmod(0o644)
    # A staging file left behind by an older writer, readable by anyone.
    staged = path.with_suffix(".tmp")
    staged.write_text("")
    staged.chmod(0o644)
    server = _Server(replies={_GRANT_URL: [_granted("at-2")]})
    async with _Served(server) as shared:
        await ClaudeLogin(path, resources=shared).token()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # Staged under a fresh name, the save never writes into the leftover.
    assert staged.read_text() == ""


@pytest.mark.asyncio
async def test_a_first_save_makes_a_private_directory(
    tmp_path: Path,
    authorization: AsyncMock,
) -> None:
    path = tmp_path / "home" / ".claude" / ".credentials.json"
    server = _Server(
        replies={_GRANT_URL: [_granted("at-1")], _PROFILE_URL: [httpx2.Response(500)]},
    )
    async with _Served(server) as shared:
        await ClaudeLogin(path, resources=shared).sign_in()
    authorization.assert_awaited_once()
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    lock = path.with_suffix(".lock")
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_refresh_waits_for_lock_admission_before_exchanging_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="at-1", expires_in_sec=0.0)
    requested, admitted = asyncio.Event(), asyncio.Event()

    async def enter() -> None:
        requested.set()
        await admitted.wait()

    entered = AsyncMock(side_effect=enter)
    exited = AsyncMock(return_value=False)
    locked = Mock(return_value=Mock(__aenter__=entered, __aexit__=exited))
    monkeypatch.setattr(login, "file_lock", locked)
    server = _Server(replies={_GRANT_URL: [_granted("at-2")]})
    async with _Served(server) as shared:
        waiting = asyncio.create_task(requested.wait())
        renewing = asyncio.create_task(ClaudeLogin(path, resources=shared).token())
        try:
            done, _ = await asyncio.wait(
                (waiting, renewing),
                return_when=asyncio.FIRST_COMPLETED,
            )
            assert waiting in done
            locked.assert_called_once_with(path.with_suffix(".lock"))
            assert not renewing.done()
            assert server.requests == []
            exited.assert_not_awaited()
            admitted.set()
            assert await renewing == "at-2"
        finally:
            waiting.cancel()
            renewing.cancel()
            await asyncio.gather(waiting, renewing, return_exceptions=True)
    entered.assert_awaited_once_with()
    exited.assert_awaited_once_with(None, None, None)
    assert len(server.requests) == 1
    assert _oauth(path)["accessToken"] == "at-2"


@pytest.mark.asyncio
async def test_sign_in_exchanges_the_authorized_code_and_stores_its_plan(
    tmp_path: Path,
    authorization: AsyncMock,
) -> None:
    path = tmp_path / ".credentials.json"
    profile = {"organization": {"organization_type": "max", "rate_limit_tier": "t20"}}
    server = _Server(
        replies={
            _GRANT_URL: [_granted("at-1")],
            _PROFILE_URL: [httpx2.Response(200, content=json_dumps(profile).encode())],
        },
    )
    shown: list[str] = []
    async with _Served(server) as shared:
        await ClaudeLogin(path, resources=shared).sign_in(show=shown.append)
    authorization.assert_awaited_once()
    assert authorization.await_args is not None
    challenge, state = cast(tuple[str, str], authorization.await_args.args)
    authorization.assert_awaited_once_with(challenge, state, show=shown.append)
    # 32 and 64 random bytes, base64url without padding.
    assert len(state) == 43
    grant, me = server.requests
    assert len(str(_body(grant)["code_verifier"])) == 86
    assert grant.extensions["timeout"] == _timeout(15.0)
    assert me.extensions["timeout"] == _timeout(10.0)
    sent = _body(grant)
    assert sent["grant_type"] == "authorization_code"
    assert sent["code"] == "the-code"
    assert sent["state"] == state
    assert sent["redirect_uri"] == "http://localhost:4321/callback"
    verifier = sent["code_verifier"]
    assert isinstance(verifier, str)
    digest = hashlib.sha256(verifier.encode()).digest()
    assert base64.urlsafe_b64encode(digest).rstrip(b"=").decode() == challenge
    assert me.headers["authorization"] == "Bearer at-1"
    assert me.headers["anthropic-beta"] == "oauth-2025-04-20"
    assert _oauth(path) == {
        "accessToken": "at-1",
        "refreshToken": "rt-at-1",
        "expiresAt": int((_NOW + 28_800) * 1000),
        "subscriptionType": "max",
        "rateLimitTier": "t20",
        "scopes": [
            "user:profile",
            "user:inference",
            "user:sessions:claude_code",
            "user:mcp_servers",
            "user:file_upload",
        ],
    }


@pytest.mark.asyncio
async def test_browser_authorization_owns_its_loopback_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its callback listens on loopback alone: the code must not reach the network."""
    socket = Mock(getsockname=Mock(return_value=("127.0.0.1", 4321)))
    entered, exited = AsyncMock(), AsyncMock(return_value=False)
    server = Mock(
        spec=asyncio.Server,
        sockets=[socket],
        __aenter__=entered,
        __aexit__=exited,
    )
    start = AsyncMock(return_value=server)
    waited = AsyncMock(return_value="the-code")
    opened, shown = Mock(), Mock()
    monkeypatch.setattr(asyncio, "start_server", start)
    monkeypatch.setattr(asyncio, "wait_for", waited)
    monkeypatch.setattr(login, "open_browser", opened)
    assert await login._browser_code("challenge", "state", show=shown) == (
        "the-code",
        "http://localhost:4321/callback",
    )
    start.assert_awaited_once()
    assert start.await_args is not None
    callback, host = cast(tuple[object, str], start.await_args.args)
    assert callable(callback)
    assert host == "127.0.0.1"
    opened.assert_called_once()
    assert opened.call_args is not None
    url = cast(str, opened.call_args.args[0])
    assert url.startswith("https://claude.com/cai/oauth/authorize?")
    assert parse_qs(urlparse(url).query) == {
        "code": ["true"],
        "client_id": ["9d1c250a-e61b-44d9-88ed-5944d1962f5e"],
        "response_type": ["code"],
        "redirect_uri": ["http://localhost:4321/callback"],
        "scope": [
            (
                "user:profile user:inference user:sessions:claude_code"
                " user:mcp_servers user:file_upload"
            ),
        ],
        "code_challenge": ["challenge"],
        "code_challenge_method": ["S256"],
        "state": ["state"],
    }
    shown.assert_called_once_with(f"Sign in to Claude in your browser:\n\n  {url}\n")
    waited.assert_awaited_once()
    assert waited.await_args is not None
    assert isinstance(waited.await_args.args[0], asyncio.Future)
    assert waited.await_args.kwargs == {"timeout": 300.0}
    entered.assert_awaited_once_with()
    exited.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_a_profile_that_is_not_an_object_is_an_error(
    tmp_path: Path,
    authorization: AsyncMock,
) -> None:
    profile = httpx2.Response(200, content=b"[]")
    server = _Server(replies={_GRANT_URL: [_granted("at-1")], _PROFILE_URL: [profile]})
    async with _Served(server) as shared:
        with pytest.raises(ReadError):
            await ClaudeLogin(
                tmp_path / ".credentials.json",
                resources=shared,
            ).sign_in()
    authorization.assert_awaited_once()


@pytest.mark.asyncio
async def test_browser_timeout_closes_its_listener_and_preserves_callback_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, exited = AsyncMock(), AsyncMock(return_value=False)
    server = Mock(
        spec=asyncio.Server,
        sockets=[Mock(getsockname=Mock(return_value=("127.0.0.1", 4321)))],
        __aenter__=entered,
        __aexit__=exited,
    )
    start = AsyncMock(return_value=server)
    waited = AsyncMock(side_effect=TimeoutError)
    redirected = AsyncMock()
    monkeypatch.setattr(asyncio, "start_server", start)
    monkeypatch.setattr(asyncio, "wait_for", waited)
    monkeypatch.setattr(login, "open_browser", Mock())
    monkeypatch.setattr(login, "redirected", redirected)
    assert await login._browser_code("challenge", "state", show=Mock()) is None
    assert start.await_args is not None
    callback = cast(
        "Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]",
        start.await_args.args[0],
    )
    reader, writer = Mock(spec=asyncio.StreamReader), Mock(spec=asyncio.StreamWriter)
    await callback(reader, writer)
    assert waited.await_args is not None
    redirected.assert_awaited_once_with(
        reader,
        writer,
        path="/callback",
        state="state",
        code=waited.await_args.args[0],
    )
    entered.assert_awaited_once_with()
    exited.assert_awaited_once_with(None, None, None)


@pytest.mark.parametrize(
    "profile",
    [httpx2.Response(403), httpx2.Response(200, content=b"{}")],
    ids=["refused", "no-organization"],
)
@pytest.mark.asyncio
async def test_a_plan_the_profile_cannot_give_is_left_unsaid(
    tmp_path: Path,
    profile: httpx2.Response,
    authorization: AsyncMock,
) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="old", expires_in_sec=0.0)
    server = _Server(replies={_GRANT_URL: [_granted("at-1")], _PROFILE_URL: [profile]})
    async with _Served(server) as shared:
        claude_login = ClaudeLogin(path, resources=shared)
        await claude_login.sign_in()
        assert await claude_login.token() == "at-1"
    authorization.assert_awaited_once()
    oauth = _oauth(path)
    assert oauth["subscriptionType"] == "max"  # Kept from before, not erased.
    assert "rateLimitTier" not in oauth


@pytest.mark.asyncio
async def test_a_plan_field_the_profile_omits_keeps_its_stored_value(
    tmp_path: Path,
    authorization: AsyncMock,
) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="old", expires_in_sec=0.0)
    path.write_text(
        json_dumps(
            {"claudeAiOauth": {**_oauth(path), "rateLimitTier": "t5"}},
        ),
    )
    profile = {"organization": {"organization_type": "pro"}}
    server = _Server(
        replies={
            _GRANT_URL: [_granted("at-1")],
            _PROFILE_URL: [httpx2.Response(200, content=json_dumps(profile).encode())],
        },
    )
    async with _Served(server) as shared:
        await ClaudeLogin(path, resources=shared).sign_in()
    authorization.assert_awaited_once()
    oauth = _oauth(path)
    assert (oauth["subscriptionType"], oauth["rateLimitTier"]) == ("pro", "t5")


@pytest.mark.asyncio
async def test_a_pasted_code_must_carry_this_sign_ins_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pasted = _Typist(answers=["the-code#abc  \n"])
    monkeypatch.setattr(login, "read_login_input", pasted)
    shown: list[str] = []
    code, redirect = await login._pasted_code("challenge", "abc", show=shown.append)
    assert (code, redirect) == (
        "the-code",
        "https://platform.claude.com/oauth/code/callback",
    )
    assert pasted.prompts == ["Code: "]
    (url,) = from_plain(re.findall(r"https://\S+", shown[0]), list[str])
    query = parse_qs(urlparse(url).query)
    assert query["redirect_uri"] == ["https://platform.claude.com/oauth/code/callback"]
    assert (query["code_challenge"], query["state"]) == (["challenge"], ["abc"])
    pasted.answers.append("the-code#other")
    with pytest.raises(
        AuthError,
        match=r"^The pasted address belongs to another sign-in\.$",
    ):
        await login._pasted_code("challenge", "abc", show=shown.append)


@pytest.mark.asyncio
async def test_sign_in_falls_back_to_a_pasted_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorization: AsyncMock,
) -> None:
    """Over ssh no browser reaches back, so the user pastes the code instead."""
    authorization.return_value = None
    pasted = AsyncMock(
        return_value=("pasted", "https://platform.claude.com/oauth/code/callback"),
    )
    monkeypatch.setattr(login, "_pasted_code", pasted)
    path = tmp_path / ".credentials.json"
    server = _Server(
        replies={_GRANT_URL: [_granted("at-1")], _PROFILE_URL: [httpx2.Response(500)]},
    )
    shown: list[str] = []
    async with _Served(server) as shared:
        await ClaudeLogin(path, resources=shared).sign_in(show=shown.append)
    authorization.assert_awaited_once()
    assert authorization.await_args is not None
    challenge, state = cast(tuple[str, str], authorization.await_args.args)
    pasted.assert_awaited_once_with(challenge, state, show=shown.append)
    sent = _body(server.requests[0])
    assert sent["code"] == "pasted"
    assert sent["state"] == state
    assert sent["redirect_uri"] == "https://platform.claude.com/oauth/code/callback"


@pytest.mark.asyncio
async def test_headless_sign_in_asks_for_the_code_without_the_browser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authorization: AsyncMock,
) -> None:
    path = tmp_path / ".credentials.json"
    pasted = AsyncMock(
        return_value=("pasted", "https://platform.claude.com/oauth/code/callback"),
    )
    monkeypatch.setattr(login, "_pasted_code", pasted)
    shown: list[str] = []
    server = _Server(
        replies={_GRANT_URL: [_granted("at-1")], _PROFILE_URL: [httpx2.Response(500)]},
    )
    async with _Served(server) as shared:
        await ClaudeLogin(path, resources=shared).sign_in(
            headless=True,
            show=shown.append,
        )
    authorization.assert_not_awaited()
    sent = _body(server.requests[0])
    assert sent["code"] == "pasted"
    verifier = sent["code_verifier"]
    assert isinstance(verifier, str)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    pasted.assert_awaited_once_with(
        challenge.rstrip(b"=").decode(),
        sent["state"],
        show=shown.append,
    )
    assert sent["redirect_uri"] == "https://platform.claude.com/oauth/code/callback"
    assert _oauth(path)["accessToken"] == "at-1"


@pytest.mark.asyncio
async def test_the_keychain_reads_only_claudes_oauth_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(getpass, "getuser", lambda: "user")
    oauth = {"accessToken": "at-1", "refreshToken": "rt-at-1", "expiresAt": 0}
    security = AsyncMock(
        return_value=json_dumps(
            {"claudeAiOauth": oauth, "mcpOAuth": {"github|1": {"accessToken": "gh"}}},
        ),
    )
    monkeypatch.setattr(login, "_security", security)
    keychain = login._Keychain()
    assert keychain.lock == tmp_path / ".claude" / ".credentials.lock"
    assert await keychain.read() == oauth
    security.assert_awaited_once_with(
        "find-generic-password",
        "-a",
        "user",
        "-s",
        "Claude Code-credentials",
        "-w",
    )


def test_the_default_file_named_on_macos_is_the_keychain_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller deriving every account's file from the default names it too."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(login, "platform", SimpleNamespace(system=lambda: "Darwin"))
    keychain = Mock(spec=login._Keychain)
    factory = Mock(return_value=keychain)
    monkeypatch.setattr(login, "_Keychain", factory)
    resources = Mock(spec=SharedResources)
    for path in (None, ClaudeLogin.default_path()):
        assert ClaudeLogin(path, resources=resources)._store is keychain
    assert factory.call_args_list == [call(), call()]


@pytest.mark.asyncio
async def test_a_keychain_write_keeps_other_logins_and_secrets_off_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any local user can read a process's arguments while it runs."""
    monkeypatch.setattr(getpass, "getuser", lambda: "user")
    mcp = {"github|1": {"accessToken": "gh"}}
    security = AsyncMock(
        return_value=json_dumps(
            {"claudeAiOauth": {"accessToken": "old"}, "mcpOAuth": mcp},
        ),
    )
    monkeypatch.setattr(login, "_security", security)
    oauth = {"accessToken": "at-2", "refreshToken": "rt-at-2"}
    await login._Keychain().write(oauth)
    secret = json_dumps({"claudeAiOauth": oauth, "mcpOAuth": mcp}).encode().hex()
    assert security.await_args_list == [
        call(
            "find-generic-password",
            "-a",
            "user",
            "-s",
            "Claude Code-credentials",
            "-w",
        ),
        call(
            "-i",
            commands=(
                'add-generic-password -U -a "user" -s "Claude Code-credentials"'
                f" -X {secret}\n"
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_an_empty_keychain_holds_no_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    security = AsyncMock(return_value="")
    monkeypatch.setattr(login, "_security", security)
    assert await login._Keychain().read() == {}
    security.assert_awaited_once()


@pytest.mark.parametrize("outcome", ["success", "failed", "cancelled"])
@pytest.mark.asyncio
async def test_security_owns_command_streams_and_child_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    error = asyncio.CancelledError("cancelled") if outcome == "cancelled" else None
    communicate = AsyncMock(return_value=(b" secret \n", b""), side_effect=error)
    process = Mock(
        spec=Process,
        returncode=1 if outcome == "failed" else 0,
        communicate=communicate,
    )
    enter, leave = AsyncMock(), AsyncMock(return_value=False)
    child = AsyncMock(
        spec=Child,
        process=process,
        __aenter__=enter,
        __aexit__=leave,
    )
    enter.return_value = child
    start = AsyncMock(return_value=child)
    monkeypatch.setattr(Child, "start", start)

    if error is not None:
        with pytest.raises(asyncio.CancelledError) as raised:
            await login._security("find", "--flag", commands="private")
        assert raised.value is error
        leave.assert_awaited_once()
        assert leave.await_args is not None
        assert leave.await_args.args[0] is asyncio.CancelledError
        assert leave.await_args.args[1] is error
    else:
        assert await login._security("find", "--flag", commands="private") == (
            "" if outcome == "failed" else "secret"
        )
        leave.assert_awaited_once_with(None, None, None)

    start.assert_awaited_once_with(
        "/usr/bin/security",
        "find",
        "--flag",
        grace_sec=0.0,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    enter.assert_awaited_once_with()
    communicate.assert_awaited_once_with(b"private")


@pytest.mark.asyncio
async def test_a_named_login_on_a_mac_is_its_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(login, "platform", SimpleNamespace(system=lambda: "Darwin"))
    keychain = Mock(spec=login._Keychain)
    monkeypatch.setattr(login, "_Keychain", keychain)
    path = tmp_path / ".credentials-work.json"
    _store(path, access="at-work", expires_in_sec=3600.0)
    async with _Served(_Server()) as shared:
        claude_login = ClaudeLogin(path, resources=shared)
        keychain.assert_not_called()
        assert await claude_login.token() == "at-work"


@dataclass(slots=True, kw_only=True)
class _Server:
    """Answers each URL with its next scripted reply, and keeps the requests."""

    replies: dict[str, list[httpx2.Response]] = field(
        default_factory=dict[str, list[httpx2.Response]],
    )
    requests: list[httpx2.Request] = field(default_factory=list[httpx2.Request])

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.replies[str(request.url)].pop(0)


@dataclass(slots=True, kw_only=True)
class _Typist:
    """Stands in for ``input``: types each of ``answers`` in turn."""

    answers: list[str] = field(default_factory=list[str])
    prompts: list[str] = field(default_factory=list[str])

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0)


class _Served(SharedResources):
    """Shared resources whose HTTP client answers from ``handler``, never the network."""

    def __init__(self, handler: Callable[[httpx2.Request], httpx2.Response]) -> None:
        super().__init__()
        self._client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))

    @property
    @override
    def http(self) -> httpx2.AsyncClient:
        return self._client

    @override
    async def aclose(self) -> None:
        await super().aclose()
        await self._client.aclose()


def _timeout(seconds: float) -> dict[str, float]:
    return dict.fromkeys(("connect", "read", "write", "pool"), seconds)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
