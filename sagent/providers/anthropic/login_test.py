"""Tests for the Claude subscription login, against a fake OAuth server."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, NoReturn, override
from urllib.parse import parse_qs, urlparse

import asyncio
import base64
import fcntl
import getpass
import hashlib
import os
import re
import stat

from treekle import ReadError, from_plain, json_dumps, json_loads

import httpx2
import pytest

from sagent.model.exceptions import AuthError
from sagent.providers.anthropic import login
from sagent.providers.anthropic.login import ClaudeLogin
from sagent.providers.resources import SharedResources


if TYPE_CHECKING:
    from asyncio.subprocess import Process
    from collections.abc import Awaitable, Callable


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


# ---- token -------------------------------------------------------------------


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


# ---- the credentials file ----------------------------------------------------


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
    assert not staged.exists()


@pytest.mark.asyncio
async def test_a_first_save_makes_a_private_directory(tmp_path: Path) -> None:
    path = tmp_path / "home" / ".claude" / ".credentials.json"
    server = _Server(
        replies={_GRANT_URL: [_granted("at-1")], _PROFILE_URL: [httpx2.Response(500)]},
    )
    async with _Served(server) as shared:
        await _sign_in(ClaudeLogin(path, resources=shared), browser=_Browser())
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    lock = path.with_suffix(".lock")
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_another_process_holding_the_lock_is_waited_for(tmp_path: Path) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="at-1", expires_in_sec=0.0)
    server = _Server(replies={_GRANT_URL: [_granted("at-2")]})
    fd = os.open(path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        async with _Served(server) as shared:
            renewing = asyncio.ensure_future(
                ClaudeLogin(path, resources=shared).token(),
            )
            await asyncio.sleep(0.05)
            assert not renewing.done()
            assert server.requests == []
            fcntl.flock(fd, fcntl.LOCK_UN)
            assert await renewing == "at-2"
    finally:
        os.close(fd)


# ---- signing in --------------------------------------------------------------


@pytest.mark.asyncio
async def test_sign_in_takes_the_code_the_browser_brings_back(tmp_path: Path) -> None:
    path = tmp_path / ".credentials.json"
    profile = {"organization": {"organization_type": "max", "rate_limit_tier": "t20"}}
    server = _Server(
        replies={
            _GRANT_URL: [_granted("at-1")],
            _PROFILE_URL: [httpx2.Response(200, content=json_dumps(profile).encode())],
        },
    )
    browser = _Browser()
    shown: list[str] = []
    async with _Served(server) as shared:
        await _sign_in(ClaudeLogin(path, resources=shared), browser=browser, show=shown)
    (url,) = browser.opened
    assert url in shown[0]
    assert url.startswith("https://claude.com/cai/oauth/authorize?")
    query = {key: values[0] for key, values in parse_qs(urlparse(url).query).items()}
    redirect = query.pop("redirect_uri")
    assert re.fullmatch(r"http://localhost:\d+/callback", redirect)
    challenge, state = query.pop("code_challenge"), query.pop("state")
    assert query == {
        "code": "true",
        "client_id": "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
        "response_type": "code",
        "scope": "user:profile user:inference user:sessions:claude_code"
        " user:mcp_servers user:file_upload",
        "code_challenge_method": "S256",
    }
    query |= {"redirect_uri": redirect, "code_challenge": challenge, "state": state}
    # 32 and 64 random bytes, base64url without padding.
    assert len(state) == 43
    grant, me = server.requests
    assert len(str(_body(grant)["code_verifier"])) == 86
    assert grant.extensions["timeout"] == _timeout(15.0)
    assert me.extensions["timeout"] == _timeout(10.0)
    sent = _body(grant)
    assert sent["grant_type"] == "authorization_code"
    assert sent["code"] == "the-code"
    assert sent["state"] == query["state"]
    assert sent["redirect_uri"] == query["redirect_uri"]
    verifier = sent["code_verifier"]
    assert isinstance(verifier, str)
    digest = hashlib.sha256(verifier.encode()).digest()
    assert (
        base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        == query["code_challenge"]
    )
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
async def test_the_redirect_takes_only_this_sign_ins_callback(tmp_path: Path) -> None:
    path = tmp_path / ".credentials.json"
    server = _Server(
        replies={_GRANT_URL: [_granted("at-1")], _PROFILE_URL: [httpx2.Response(500)]},
    )
    browser = _Browser(
        visits=(
            "GET /callback?code=forged&state=someone-else HTTP/1.1",
            "GET /favicon.ico?code=forged&state={state} HTTP/1.1",
            "GET /callback?state={state} HTTP/1.1",
            "GET /\udcff?code=forged&state={state} HTTP/1.1",
            "GARBAGE",
            "GET /callback?code=the-code&state={state} HTTP/1.1",
        ),
    )
    async with _Served(server) as shared:
        await _sign_in(ClaudeLogin(path, resources=shared), browser=browser)
    assert browser.statuses == [400, 400, 400, 400, 400, 200]
    failed = "Sign-in failed; see the terminal."
    assert browser.replies[0] == (
        "HTTP/1.1 400 Bad Request\r\ncontent-type: text/plain\r\n"
        f"content-length: {len(failed)}\r\nconnection: close\r\n\r\n{failed}"
    )
    signed_in = "Signed in; you can close this tab."
    assert browser.replies[-1] == (
        "HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\n"
        f"content-length: {len(signed_in)}\r\nconnection: close\r\n\r\n{signed_in}"
    )
    assert _body(server.requests[0])["code"] == "the-code"


@pytest.mark.parametrize(
    "profile",
    [httpx2.Response(403), httpx2.Response(200, content=b"{}")],
    ids=["refused", "no-organization"],
)
@pytest.mark.asyncio
async def test_a_plan_the_profile_cannot_give_is_left_unsaid(
    tmp_path: Path,
    profile: httpx2.Response,
) -> None:
    path = tmp_path / ".credentials.json"
    _store(path, access="old", expires_in_sec=0.0)
    server = _Server(replies={_GRANT_URL: [_granted("at-1")], _PROFILE_URL: [profile]})
    async with _Served(server) as shared:
        claude_login = ClaudeLogin(path, resources=shared)
        await _sign_in(claude_login, browser=_Browser())
        assert await claude_login.token() == "at-1"
    oauth = _oauth(path)
    assert oauth["subscriptionType"] == "max"  # Kept from before, not erased.
    assert "rateLimitTier" not in oauth


@pytest.mark.asyncio
async def test_a_plan_field_the_profile_omits_keeps_its_stored_value(
    tmp_path: Path,
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
        await _sign_in(ClaudeLogin(path, resources=shared), browser=_Browser())
    oauth = _oauth(path)
    assert (oauth["subscriptionType"], oauth["rateLimitTier"]) == ("pro", "t5")


@pytest.mark.asyncio
async def test_a_pasted_code_must_carry_this_sign_ins_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pasted = _Typist(answers=["the-code#abc  \n"])
    monkeypatch.setattr("builtins.input", pasted)
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
        match=r"^the pasted code belongs to another sign-in$",
    ):
        await login._pasted_code("challenge", "abc", show=shown.append)


@pytest.mark.asyncio
async def test_sign_in_falls_back_to_a_pasted_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over ssh no browser reaches back, so the user pastes the code instead."""
    path = tmp_path / ".credentials.json"
    browser = _Browser(visits=())
    waited = _Expired()
    monkeypatch.setattr(asyncio, "wait_for", waited)
    monkeypatch.setattr("builtins.input", _Reader(browser=browser))
    server = _Server(
        replies={_GRANT_URL: [_granted("at-1")], _PROFILE_URL: [httpx2.Response(500)]},
    )
    shown: list[str] = []
    async with _Served(server) as shared:
        await _sign_in(ClaudeLogin(path, resources=shared), browser=browser, show=shown)
    assert [text.split(":")[0] for text in shown] == [
        "Sign in to Claude in your browser",
        "Open this, sign in, and paste the code it shows",
    ]
    assert waited.timeouts == [300.0]  # Five minutes for the browser.
    challenges = [
        parse_qs(urlparse(url).query)["code_challenge"]
        for url in (browser.opened[0], *re.findall(r"https://\S+", shown[1]))
    ]
    assert challenges[0] == challenges[1]
    sent = _body(server.requests[0])
    assert sent["code"] == "pasted"
    assert sent["redirect_uri"] == "https://platform.claude.com/oauth/code/callback"


@pytest.mark.asyncio
async def test_headless_sign_in_asks_for_the_code_without_the_browser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / ".credentials.json"
    browser = _Browser(visits=())
    waited = _Expired()
    monkeypatch.setattr(asyncio, "wait_for", waited)
    shown: list[str] = []
    monkeypatch.setattr("builtins.input", _Reader(browser=browser, shown=shown))
    server = _Server(
        replies={_GRANT_URL: [_granted("at-1")], _PROFILE_URL: [httpx2.Response(500)]},
    )
    async with _Served(server) as shared:
        await _sign_in(
            ClaudeLogin(path, resources=shared),
            browser=browser,
            show=shown,
            headless=True,
        )
    assert [text.split(":")[0] for text in shown] == [
        "Open this, sign in, and paste the code it shows",
    ]
    assert browser.opened == []
    assert waited.timeouts == []
    assert _body(server.requests[0])["code"] == "pasted"


@pytest.mark.asyncio
async def test_the_macos_keychain_holds_the_login_claude_keeps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keychain = _Keychain.installed(tmp_path, monkeypatch)
    mcp = {"github|1": {"accessToken": "gh"}}
    oauth = {"accessToken": "at-1", "refreshToken": "rt-at-1", "expiresAt": 0}
    keychain.entry.write_text(json_dumps({"claudeAiOauth": oauth, "mcpOAuth": mcp}))
    server = _Server(replies={_GRANT_URL: [_granted("at-2")]})
    async with _Served(server) as shared:
        assert await ClaudeLogin(resources=shared).token() == "at-2"
    stored = _stored(keychain.entry)
    assert stored["mcpOAuth"] == mcp
    renewed = from_plain(stored["claudeAiOauth"], dict[str, object])
    assert renewed["accessToken"] == "at-2"
    # The renewal spent the refresh token the Keychain held.
    assert _body(server.requests[0])["refresh_token"] == oauth["refreshToken"]
    assert (tmp_path / ".claude" / ".credentials.lock").exists()


@pytest.mark.asyncio
async def test_a_keychain_write_keeps_the_login_out_of_every_command_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any local user can read a process's arguments while it runs."""
    keychain = _Keychain.installed(tmp_path, monkeypatch)
    oauth = {"accessToken": "at-1", "refreshToken": "rt-at-1", "expiresAt": 0}
    keychain.entry.write_text(json_dumps({"claudeAiOauth": oauth}))
    server = _Server(replies={_GRANT_URL: [_granted("at-2")]})
    async with _Served(server) as shared:
        await ClaudeLogin(resources=shared).token()
    leaked = [call for call in keychain.calls() if any("at-2" in arg for arg in call)]
    assert leaked == []
    assert _oauth(keychain.entry)["refreshToken"] == "rt-at-2"


@pytest.mark.asyncio
async def test_an_empty_keychain_holds_no_login(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    _Keychain.installed(tmp_path, monkeypatch)
    server = _Server(replies={_GRANT_URL: [httpx2.Response(400)]})
    async with _Served(server) as shared:
        with pytest.raises(AuthError, match=_REVOKED):
            await ClaudeLogin(resources=shared).token()
    assert _body(server.requests[0])["refresh_token"] == ""
    # The miss ``security`` reports is ours to handle, not the terminal's.
    assert capfd.readouterr().err == ""


@pytest.mark.asyncio
async def test_a_keychain_read_cancelled_midway_ends_security(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``security`` that hangs, on a locked Keychain, is not left behind."""
    script = tmp_path / "security"
    script.write_text(f'#!/bin/sh\necho $$ > "{tmp_path}/pid"\nexec sleep 60\n')
    script.chmod(0o700)
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        partial(_security_at, stub=str(script), real=asyncio.create_subprocess_exec),
    )
    reading = asyncio.create_task(login._security(*login._find_entry(), "-w"))
    async with asyncio.timeout(5):
        while True:
            pid = _text(tmp_path / "pid")
            if pid:
                break
            await asyncio.sleep(0.01)
    reading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(reading, timeout=5)
    async with asyncio.timeout(5):
        while True:
            if not _text(Path(f"/proc/{pid}/stat")):
                break
            await asyncio.sleep(0.01)


def _text(path: Path) -> str:
    """Return ``path``'s text, stripped; empty while it does not exist."""
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return ""


@pytest.mark.asyncio
async def test_a_named_login_on_a_mac_is_its_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keychain = _Keychain.installed(tmp_path, monkeypatch)
    path = tmp_path / ".credentials-work.json"
    _store(path, access="at-work", expires_in_sec=3600.0)
    async with _Served(_Server()) as shared:
        assert await ClaudeLogin(path, resources=shared).token() == "at-work"
    assert not (keychain.dir / "calls").exists()


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

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Reader:
    """Stands in for ``input``: the user pastes the code the sign-in page shows."""

    browser: _Browser
    shown: list[str] = field(default_factory=list[str])

    def __call__(self, prompt: str) -> str:
        del prompt
        link = self.browser.opened[0] if self.browser.opened else self.shown[0]
        state = parse_qs(urlparse(link.split()[-1]).query)["state"][0]
        return f"pasted#{state}"


# Answers as ``security`` does, for the one entry ``claude`` keeps, and refuses
# any other command line: ``find-generic-password -w`` prints the entry, and
# ``-i`` reads ``add-generic-password ... -X <hex>`` lines on stdin. Logs each
# call's arguments, and each line it reads.
_SECURITY = """#!/bin/sh
dir=$(dirname "$0")
printf '%s\\0' "$@" >> "$dir/calls"
printf '\\n' >> "$dir/calls"
if [ "$*" = "find-generic-password -a {user} -s Claude Code-credentials -w" ]; then
  if [ -e "$dir/entry" ]; then cat "$dir/entry"; exit 0; fi
  echo "security: The specified item could not be found in the keychain." >&2
  exit 44
fi
[ "$*" = "-i" ] || exit 2
while IFS= read -r line; do
  printf '%s\\n' "$line" >> "$dir/said"
  case "$line" in
    'add-generic-password -U -a "{user}" -s "Claude Code-credentials" -X '*)
      printf '%s' "${{line##* }}" | xxd -r -p > "$dir/entry" ;;
    *) exit 2 ;;
  esac
done
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Keychain:
    """A stub ``security`` standing in for the macOS Keychain, with one entry."""

    dir: Path

    @classmethod
    def installed(cls, home: Path, monkeypatch: pytest.MonkeyPatch) -> _Keychain:
        """Make this a Mac whose ``/usr/bin/security`` is the stub, homed at ``home``."""
        keychain = cls(dir=home / "keychain")
        keychain.dir.mkdir()
        script = keychain.dir / "security"
        script.write_text(_SECURITY.format(user=getpass.getuser()))
        script.chmod(0o700)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(login, "platform", SimpleNamespace(system=lambda: "Darwin"))
        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            partial(
                _security_at,
                stub=str(script),
                real=asyncio.create_subprocess_exec,
            ),
        )
        return keychain

    @property
    def entry(self) -> Path:
        return self.dir / "entry"

    def calls(self) -> list[list[str]]:
        lines = (self.dir / "calls").read_text().splitlines()
        return [line.split("\0")[:-1] for line in lines]


async def _security_at(
    program: str,
    *args: str,
    stub: str,
    real: Callable[..., Awaitable[Process]],
    **kwargs: object,
) -> Process:
    """Run the stub wherever the code runs ``/usr/bin/security``."""
    return await real(
        stub if program == "/usr/bin/security" else program,
        *args,
        **kwargs,
    )


@dataclass(slots=True, kw_only=True)
class _Expired:
    """Stands in for ``asyncio.wait_for``: every wait runs out, and is noted."""

    timeouts: list[float] = field(default_factory=list[float])

    async def __call__(self, awaitable: Awaitable[object], **limits: float) -> NoReturn:
        del awaitable
        self.timeouts.append(limits["timeout"])
        raise TimeoutError


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


@dataclass(slots=True, kw_only=True)
class _Browser:
    r"""Opens the sign-in page, and sends the redirect each request line of ``visits``.

    A line's ``{state}`` is the sign-in's; ``\\udcff`` sends the byte 0xff.
    """

    visits: tuple[str, ...] = ("GET /callback?code=the-code&state={state} HTTP/1.1",)
    opened: list[str] = field(default_factory=list[str])
    replies: list[str] = field(default_factory=list[str])
    tasks: list[asyncio.Task[None]] = field(default_factory=list["asyncio.Task[None]"])

    @property
    def statuses(self) -> list[int]:
        return [int(reply.split()[1]) for reply in self.replies]

    def open(self, url: str) -> bool:
        self.opened.append(url)
        self.tasks.append(asyncio.get_running_loop().create_task(self._visit(url)))
        return True

    async def _visit(self, url: str) -> None:
        query = parse_qs(urlparse(url).query)
        port = urlparse(query["redirect_uri"][0]).port
        for visit in self.visits:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            line = visit.format(state=query["state"][0])
            writer.write(
                f"{line}\r\nhost: localhost\r\n\r\n".encode(errors="surrogateescape"),
            )
            await writer.drain()
            self.replies.append((await reader.read()).decode())
            writer.close()
            await writer.wait_closed()


def _timeout(seconds: float) -> dict[str, float]:
    return dict.fromkeys(("connect", "read", "write", "pool"), seconds)


async def _sign_in(
    claude_login: ClaudeLogin,
    *,
    browser: _Browser,
    show: list[str] | None = None,
    headless: bool = False,
) -> None:
    shown = [] if show is None else show
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(login, "webbrowser", browser)
        await claude_login.sign_in(headless=headless, show=shown.append)
    await asyncio.gather(*browser.tasks)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
