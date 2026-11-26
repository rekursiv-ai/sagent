"""Tests for the ChatGPT login: reading, renewing, lending, and signing in, offline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast, override
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlparse

import asyncio
import base64
import hashlib
import stat
import time

from treekle import from_plain, json_dumps, json_loads

import httpx2
import pytest

from sagent.lib.files.owner_only import write_private
from sagent.model.exceptions import AuthError
from sagent.providers.openai import login
from sagent.providers.openai.lease import LeasedToken
from sagent.providers.openai.login import (
    CodexAccount,
    CodexLogin,
    CodexTokens,
    DeviceCode,
    codex_account,
    poll_device_code,
    refresh_grant,
    request_device_code,
    revoke_refresh_token,
)
from sagent.providers.resources import SharedResources
from sagent.providers.testing import refuse_browsers


if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.fixture(autouse=True)
def no_real_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test that opens the user's browser, as a sign-in not headless would."""
    refuse_browsers(monkeypatch, login)


_CLIENT = "app_EMoamEEZ73f0CkXaXp7hrann"
"""The OAuth client ``codex`` signs in as."""


def _jwt(claims: Mapping[str, object]) -> str:
    """Return an unsigned JWT carrying ``claims``."""
    return f"{_part({'alg': 'none'})}.{_part(claims)}.sig"


def _part(value: object) -> str:
    return base64.urlsafe_b64encode(json_dumps(value).encode()).rstrip(b"=").decode()


def _token(name: str, *, expires_in_sec: float = 3_600.0, account: str = "") -> str:
    claims: dict[str, object] = {"sub": name, "exp": time.time() + expires_in_sec}
    if account:
        claims["https://api.openai.com/auth"] = {"chatgpt_account_id": account}
    return _jwt(claims)


def _store(path: Path, tokens: Mapping[str, object]) -> None:
    _write(path, json_dumps({"auth_mode": "chatgpt", "keep": "me", "tokens": tokens}))


def _write(path: Path, text: str) -> None:
    path.write_text(text)


def _stored(path: Path) -> dict[str, object]:
    return json_loads(path.read_text(), dict[str, object])


def _tokens(path: Path) -> object:
    return _stored(path)["tokens"]


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _json(request: httpx2.Request) -> dict[str, object]:
    return json_loads(request.content, dict[str, object])


def _form(request: httpx2.Request) -> dict[str, str]:
    return {
        key: values[0] for key, values in parse_qs(request.content.decode()).items()
    }


def _reply(body: Mapping[str, object], status: int = 200) -> httpx2.Response:
    return httpx2.Response(status, content=json_dumps(body).encode())


def _refresh(token: str) -> dict[str, object]:
    return {"grant_type": "refresh_token", "refresh_token": token, "client_id": _CLIENT}


@pytest.fixture
def auth(tmp_path: Path) -> Path:
    return tmp_path / "auth.json"


def test_the_login_is_the_one_codex_keeps_unless_named(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resources = SharedResources()
    monkeypatch.delenv("CODEX_HOME", raising=False)
    assert (
        CodexLogin(resources=resources).path == Path("~/.codex/auth.json").expanduser()
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    assert CodexLogin(resources=resources).path == tmp_path / "home" / "auth.json"
    named = tmp_path / "auth-work.json"
    assert CodexLogin(named, resources=resources).path == named


@pytest.mark.asyncio
async def test_token_is_the_stored_one_while_it_is_fresh(auth: Path) -> None:
    access = _token("a")
    _store(auth, {"access_token": access, "refresh_token": "r", "account_id": "acct"})
    server = _Server(replies=[])
    async with _Served(server) as shared:
        codex = CodexLogin(auth, resources=shared)
        assert await codex.token() == (access, "acct")
        assert await codex.token() == (access, "acct")
    assert server.requests == []


@pytest.mark.asyncio
async def test_a_token_about_to_expire_is_renewed_and_stored(auth: Path) -> None:
    # Due within the five minutes it is renewed ahead of.
    old = _token("old", expires_in_sec=299.0)
    _store(
        auth,
        {
            "access_token": old,
            "refresh_token": "r1",
            "account_id": "acct-1",
            "id_token": "i1",
            "keep": "too",
        },
    )
    fresh = _token("new", account="acct-2")
    server = _Server(replies=[_reply({"access_token": fresh, "refresh_token": "r2"})])
    async with _Served(server) as shared:
        codex = CodexLogin(auth, resources=shared)
        assert await codex.token() == (fresh, "acct-2")
        assert await codex.token() == (fresh, "acct-2")
    (sent,) = server.requests
    assert str(sent.url) == "https://auth.openai.com/oauth/token"
    assert _json(sent) == _refresh("r1")
    stored = _stored(auth)
    # Every field ``codex`` keeps beside the tokens survives, and what the
    # refresh left out (the ID token) is kept.
    assert stored["keep"] == "me"
    assert stored["auth_mode"] == "chatgpt"
    assert isinstance(stored["last_refresh"], str)
    assert stored["last_refresh"].endswith("Z")
    assert _tokens(auth) == {
        "access_token": fresh,
        "refresh_token": "r2",
        "account_id": "acct-2",
        "id_token": "i1",
        "keep": "too",
    }
    assert _mode(auth) == 0o600
    assert (auth.parent / "auth.json.lock").exists()


@pytest.mark.asyncio
async def test_a_token_with_five_minutes_left_is_kept(
    auth: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    access = _jwt({"exp": now + 300.0})
    _store(auth, {"access_token": access, "refresh_token": "r", "account_id": "acct"})
    monkeypatch.setattr(time, "time", lambda: now)
    server = _Server(replies=[])
    async with _Served(server) as shared:
        assert await CodexLogin(auth, resources=shared).token() == (access, "acct")
    assert server.requests == []


@pytest.mark.asyncio
async def test_a_stored_login_without_a_token_is_renewed(auth: Path) -> None:
    _store(auth, {"refresh_token": "r1", "account_id": "acct"})
    fresh = _token("fresh")
    server = _Server(replies=[_reply({"access_token": fresh, "refresh_token": "r2"})])
    async with _Served(server) as shared:
        assert await CodexLogin(auth, resources=shared).token() == (fresh, "acct")
    assert _json(server.requests[0]) == _refresh("r1")


@pytest.mark.asyncio
async def test_a_refresh_that_names_no_new_refresh_token_keeps_the_old(
    auth: Path,
) -> None:
    old = _token("old", expires_in_sec=0.0)
    _store(auth, {"access_token": old, "refresh_token": "r1", "account_id": "acct"})
    fresh = _token("new")
    server = _Server(replies=[_reply({"access_token": fresh, "id_token": "i2"})])
    async with _Served(server) as shared:
        assert await CodexLogin(auth, resources=shared).token() == (fresh, "acct")
    assert _tokens(auth) == {
        "access_token": fresh,
        "refresh_token": "r1",
        "account_id": "acct",
        "id_token": "i2",
    }


@pytest.mark.asyncio
async def test_renew_adopts_a_token_another_process_rotated(auth: Path) -> None:
    first = _token("first")
    _store(auth, {"access_token": first, "refresh_token": "r1", "account_id": "acct"})
    third = _token("third")
    server = _Server(replies=[_reply({"access_token": third, "refresh_token": "r3"})])
    async with _Served(server) as shared:
        codex = CodexLogin(auth, resources=shared)
        assert await codex.token() == (first, "acct")
        second = _token("second")
        _store(
            auth,
            {"access_token": second, "refresh_token": "r2", "account_id": "acct"},
        )
        await codex.renew()
        assert await codex.token() == (second, "acct")
        assert server.requests == []
        # Rejected with nothing newer on disk, it rotates.
        await codex.renew()
        assert await codex.token() == (third, "acct")
    (sent,) = server.requests
    assert _json(sent) == _refresh("r2")


@pytest.mark.asyncio
async def test_renew_waits_for_lock_admission_before_exchanging_tokens(
    auth: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store(
        auth,
        {"access_token": _token("a"), "refresh_token": "r1", "account_id": "x"},
    )
    requested, admitted = asyncio.Event(), asyncio.Event()

    async def enter() -> None:
        requested.set()
        await admitted.wait()

    entered = AsyncMock(side_effect=enter)
    exited = AsyncMock(return_value=False)
    locked = Mock(return_value=Mock(__aenter__=entered, __aexit__=exited))
    monkeypatch.setattr(login, "file_lock", locked)
    fresh = _token("fresh")
    server = _Server(replies=[_reply({"access_token": fresh, "refresh_token": "r2"})])
    async with _Served(server) as shared:
        codex = CodexLogin(auth, resources=shared)
        await codex.token()
        renewing = asyncio.create_task(codex.renew())
        try:
            await requested.wait()
            locked.assert_called_once_with(auth.with_suffix(".json.lock"))
            assert not renewing.done()
            assert server.requests == []
            exited.assert_not_awaited()
            admitted.set()
            await renewing
            assert await codex.token() == (fresh, "x")
        finally:
            renewing.cancel()
            await asyncio.gather(renewing, return_exceptions=True)
    entered.assert_awaited_once_with()
    exited.assert_awaited_once_with(None, None, None)
    assert len(server.requests) == 1
    assert _tokens(auth) == {
        "access_token": fresh,
        "refresh_token": "r2",
        "account_id": "x",
    }


@pytest.mark.asyncio
async def test_a_refused_refresh_says_to_sign_in_again(auth: Path) -> None:
    old = _token("old", expires_in_sec=0.0)
    _store(auth, {"access_token": old, "refresh_token": "r1", "account_id": "a"})
    server = _Server(
        replies=[
            _reply({"error": "invalid_grant"}, status=400),
            _reply({"error": "revoked"}, status=401),
            _reply({"error": "down"}, status=503),
        ],
    )
    async with _Served(server) as shared:
        codex = CodexLogin(auth, resources=shared)
        for _ in range(2):
            with pytest.raises(AuthError, match="sign in again"):
                await codex.token()
        # The issuer failing is not the login failing.
        with pytest.raises(httpx2.HTTPStatusError):
            await codex.token()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "holds no ChatGPT login"),
        ("not json", "holds no ChatGPT login"),
        ('{"tokens": {"access_token": "a"}}', "holds no ChatGPT login"),
        ("[1]", "holds no ChatGPT login"),
        ('{"auth_mode": "apikey"}', "holds an API key"),
        ('{"auth_mode": "api_key"}', "holds an API key"),
        ('{"OPENAI_API_KEY": "sk"}', "holds an API key"),
    ],
)
@pytest.mark.asyncio
async def test_a_file_without_a_chatgpt_login_says_so(
    auth: Path,
    text: str,
    message: str,
) -> None:
    if text:
        _write(auth, text)
    async with SharedResources() as shared:
        with pytest.raises(AuthError, match=message):
            await CodexLogin(auth, resources=shared).token()


@pytest.mark.asyncio
async def test_a_token_without_an_expiry_lives_as_long_as_its_grant(auth: Path) -> None:
    _store(
        auth,
        {
            "access_token": _jwt({"sub": "old"}),
            "refresh_token": "r1",
            "account_id": "a",
        },
    )
    plain = _jwt({"sub": "plain"})
    server = _Server(
        replies=[
            _reply({"access_token": plain, "expires_in": 3_600}),
            _reply({"access_token": plain}),
        ],
    )
    async with _Served(server) as shared:
        codex = CodexLogin(auth, resources=shared)
        assert await codex.token() == (plain, "a")
        assert await codex.token() == (plain, "a")
        assert len(server.requests) == 1
        with pytest.raises(
            AuthError,
            match=r"^The ChatGPT login was issued with no expiry; sign in again\.$",
        ):
            await codex.renew()


@pytest.mark.parametrize(
    "token",
    [
        "a.bm90IGpzb24.c",  # A payload that is not JSON.
        f"a.{_part([1])}.c",  # One that is no object.
        _jwt({"exp": "tomorrow"}),
        "opaque",
    ],
)
@pytest.mark.asyncio
async def test_a_token_whose_expiry_cannot_be_read_is_renewed(
    auth: Path,
    token: str,
) -> None:
    _store(auth, {"access_token": token, "refresh_token": "r1", "account_id": "a"})
    fresh = _token("fresh")
    server = _Server(replies=[_reply({"access_token": fresh, "refresh_token": "r2"})])
    async with _Served(server) as shared:
        assert await CodexLogin(auth, resources=shared).token() == (fresh, "a")


def test_a_stored_login_is_private_even_over_a_leftover_staging_file(
    auth: Path,
) -> None:
    staged = auth.with_suffix(".tmp")
    staged.write_text("old")
    staged.chmod(0o644)
    write_private(auth, "{}")
    assert _mode(auth) == 0o600
    assert auth.read_text() == "{}"
    assert staged.read_text() == "old"


@pytest.mark.asyncio
async def test_lend_hands_out_the_token_held_and_renews_one_refused(auth: Path) -> None:
    held = _token("held", account="acct")
    _store(auth, {"access_token": held, "refresh_token": "r1", "account_id": "acct"})
    fresh = _token("fresh", account="acct")
    server = _Server(replies=[_reply({"access_token": fresh, "refresh_token": "r2"})])
    async with _Served(server) as shared:
        codex = CodexLogin(auth, resources=shared)
        lent = await codex.lend(rejected="")
        assert (lent.access_token, lent.account_id) == (held, "acct")
        # A token the lender no longer holds was replaced already.
        assert (await codex.lend(rejected="stale")).access_token == held
        assert server.requests == []
        renewed = await codex.lend(rejected=held)
        assert renewed == LeasedToken(
            access_token=fresh,
            account_id="acct",
            expires_at=renewed.expires_at,
        )
        assert renewed.expires_at > time.time() + 3_000.0
    assert len(server.requests) == 1


_LINK = "Sign in to ChatGPT in your browser:\n\n  {}\n"


@pytest.mark.asyncio
async def test_sign_in_exchanges_the_authorization_code_and_stores_the_login(
    auth: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = AsyncMock(return_value="the-code")
    monkeypatch.setattr(login, "_code", authorization)
    shown: list[str] = []
    show = shown.append
    access = _token("signed-in", account="acct-9")
    server = _Server(
        replies=[
            _reply({"access_token": access, "refresh_token": "r", "id_token": "i"}),
        ],
    )
    async with _Served(server) as shared:
        codex = CodexLogin(auth, resources=shared)
        await codex.sign_in(show=show)
        assert await codex.token() == (access, "acct-9")
    authorization.assert_awaited_once()
    url = cast(str, authorization.await_args_list[0].args[0])
    assert authorization.await_args_list[0].args == (url,)
    assert shown == []
    (sent,) = server.requests
    assert str(sent.url) == "https://auth.openai.com/oauth/token"
    assert set(from_plain(sent.extensions["timeout"], dict[str, float]).values()) == {
        15.0,
    }
    form = _form(sent)
    verifier = form.pop("code_verifier")
    # 64 and 32 random bytes, base64url without padding.
    assert len(verifier) == 86
    assert form == {
        "grant_type": "authorization_code",
        "code": "the-code",
        "redirect_uri": "http://localhost:1455/auth/callback",
        "client_id": _CLIENT,
    }
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    parsed = urlparse(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://auth.openai.com/oauth/authorize"
    )
    params = {key: values[0] for key, values in parse_qs(parsed.query).items()}
    assert params == {
        "client_id": _CLIENT,
        "response_type": "code",
        "redirect_uri": "http://localhost:1455/auth/callback",
        "scope": "openid profile email offline_access api.connectors.read api.connectors.invoke",
        "code_challenge": expected.rstrip(b"=").decode(),
        "code_challenge_method": "S256",
        "state": params["state"],
        "codex_cli_simplified_flow": "true",
        "id_token_add_organizations": "true",
        "originator": "codex_cli_rs",
    }
    assert len(params["state"]) == 43
    authorization.assert_awaited_once_with(
        url,
        state=params["state"],
        show=show,
        headless=False,
    )
    assert _tokens(auth) == {
        "access_token": access,
        "refresh_token": "r",
        "id_token": "i",
        "account_id": "acct-9",
    }
    assert _stored(auth)["auth_mode"] == "chatgpt"
    assert (auth.parent / "auth.json.lock").exists()


@pytest.mark.asyncio
async def test_refused_authorization_neither_exchanges_nor_stores_tokens(
    auth: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refused = AuthError("wrong state")
    authorization = AsyncMock(side_effect=refused)
    monkeypatch.setattr(login, "_code", authorization)
    server = _Server(replies=[])
    async with _Served(server) as shared:
        with pytest.raises(AuthError) as raised:
            await CodexLogin(auth, resources=shared).sign_in(show=lambda _: None)
    assert raised.value is refused
    authorization.assert_awaited_once()
    assert server.requests == []
    assert not await asyncio.to_thread(auth.exists)


@pytest.mark.asyncio
async def test_the_browser_redirects_back_with_the_code() -> None:
    entered, exited = AsyncMock(), AsyncMock(return_value=False)
    server = Mock(spec=asyncio.Server, __aenter__=entered, __aexit__=exited)
    listening = AsyncMock(return_value=server)
    waiting = AsyncMock(return_value="c1")
    pasted = AsyncMock()
    redirected = AsyncMock()
    opened = Mock()
    shown: list[str] = []
    with (
        patch.object(asyncio, "start_server", listening),
        patch.object(asyncio, "wait_for", waiting),
        patch.object(login, "open_browser", opened),
        patch.object(login, "read_pasted_code", pasted),
        patch.object(login, "redirected", redirected),
    ):
        code = await login._code(
            "https://sign.in",
            state="s1",
            show=shown.append,
            headless=False,
            port=4321,
        )
        listening.assert_awaited_once()
        assert listening.await_args_list[0].args[1:] == ("127.0.0.1", 4321)
        assert listening.await_args_list[0].kwargs == {}
        callback = cast(
            Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]],
            listening.await_args_list[0].args[0],
        )
        reader, writer = asyncio.StreamReader(), Mock(spec=asyncio.StreamWriter)
        await callback(reader, writer)
    assert code == "c1"
    opened.assert_called_once_with("https://sign.in")
    assert shown == [_LINK.format("https://sign.in")]
    waiting.assert_awaited_once()
    pending = cast(asyncio.Future[str], waiting.await_args_list[0].args[0])
    waiting.assert_awaited_once_with(pending, timeout=300.0)
    redirected.assert_awaited_once_with(
        reader,
        writer,
        path="/auth/callback",
        state="s1",
        code=pending,
    )
    pasted.assert_not_awaited()
    entered.assert_awaited_once_with()
    exited.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("headless", [False, True])
async def test_a_browser_that_never_redirects_gives_way_to_a_paste(
    headless: bool,
) -> None:
    entered, exited = AsyncMock(), AsyncMock(return_value=False)
    server = Mock(spec=asyncio.Server, __aenter__=entered, __aexit__=exited)
    listening = AsyncMock(return_value=server)
    waiting = AsyncMock(side_effect=TimeoutError)
    pasted = AsyncMock(return_value="c")
    opened = Mock()
    shown: list[str] = []
    show = shown.append
    with (
        patch.object(asyncio, "start_server", listening),
        patch.object(asyncio, "wait_for", waiting),
        patch.object(login, "open_browser", opened),
        patch.object(login, "read_pasted_code", pasted),
    ):
        code = await login._code(
            "https://sign.in",
            state="s",
            show=show,
            headless=headless,
        )
    assert code == "c"
    assert shown == [_LINK.format("https://sign.in")]
    listening.assert_awaited_once()
    assert listening.await_args_list[0].args[1:] == ("127.0.0.1", 1455)
    pasted.assert_awaited_once_with(state="s", show=show, browser_tried=not headless)
    if headless:
        opened.assert_not_called()
        waiting.assert_not_awaited()
    else:
        opened.assert_called_once_with("https://sign.in")
        waiting.assert_awaited_once()
        pending = cast(asyncio.Future[str], waiting.await_args_list[0].args[0])
        waiting.assert_awaited_once_with(pending, timeout=300.0)
    entered.assert_awaited_once_with()
    exited.assert_awaited_once_with(None, None, None)


@pytest.mark.asyncio
async def test_a_browser_sign_in_gives_way_when_its_port_is_taken() -> None:
    listening = AsyncMock(side_effect=OSError("port taken"))
    pasted = AsyncMock(return_value="c")
    opened = Mock()
    shown: list[str] = []
    show = shown.append
    with (
        patch.object(asyncio, "start_server", listening),
        patch.object(login, "open_browser", opened),
        patch.object(login, "read_pasted_code", pasted),
    ):
        code = await login._code("https://x", state="s", show=show, headless=False)
    assert code == "c"
    assert shown == [_LINK.format("https://x")]
    listening.assert_awaited_once()
    assert listening.await_args_list[0].args[1:] == ("127.0.0.1", 1455)
    pasted.assert_awaited_once_with(state="s", show=show, browser_tried=False)
    opened.assert_not_called()


@pytest.mark.asyncio
async def test_sign_in_device_polls_until_the_code_is_approved(
    auth: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = AsyncMock()
    monkeypatch.setattr(login, "_code", authorization)
    waits: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(delay_sec: float) -> None:
        waits.append(delay_sec)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    limits: list[float | None] = []
    real_timeout = asyncio.timeout

    def timeout(delay: float | None) -> asyncio.Timeout:
        limits.append(delay)
        return real_timeout(delay)

    monkeypatch.setattr(asyncio, "timeout", timeout)
    access = _token("device", account="acct-d")
    server = _Server(
        replies=[
            _reply(
                {"usercode": "ABCD-1234", "device_auth_id": "dev-1", "interval": "3"},
            ),
            _reply({}, status=403),
            _reply({}, status=404),
            _reply({"authorization_code": "auth-code", "code_verifier": "verifier"}),
            _reply({"access_token": access, "refresh_token": "r"}),
        ],
    )
    shown: list[str] = []
    async with _Served(server) as shared:
        codex = CodexLogin(auth, resources=shared)
        await codex.sign_in(headless=True, show=shown.append)
        assert await codex.token() == (access, "acct-d")
    assert shown == [
        "Open https://auth.openai.com/codex/device and enter the code ABCD-1234.\n",
    ]
    assert waits == [3.0, 3.0, 3.0]
    authorization.assert_not_awaited()
    # Fifteen minutes to approve it.
    assert limits == [900.0]
    assert (auth.parent / "auth.json.lock").exists()
    issue, *polls, exchange = server.requests
    assert str(issue.url) == "https://auth.openai.com/api/accounts/deviceauth/usercode"
    assert _json(issue) == {"client_id": _CLIENT}
    assert [str(poll.url) for poll in polls] == [
        "https://auth.openai.com/api/accounts/deviceauth/token",
    ] * 3
    assert [_json(poll) for poll in polls] == [
        {"device_auth_id": "dev-1", "user_code": "ABCD-1234"},
    ] * 3
    assert _form(exchange) == {
        "grant_type": "authorization_code",
        "code": "auth-code",
        "redirect_uri": "https://auth.openai.com/deviceauth/callback",
        "client_id": _CLIENT,
        "code_verifier": "verifier",
    }


@pytest.mark.asyncio
async def test_sign_in_device_polls_no_faster_than_once_a_second(
    auth: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(delay_sec: float) -> None:
        waits.append(delay_sec)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    server = _Server(
        replies=[
            _reply({"user_code": "WXYZ", "device_auth_id": "dev-2"}),
            _reply({"authorization_code": "c", "code_verifier": "v"}),
            _reply({"access_token": _token("d"), "refresh_token": "r"}),
        ],
    )
    async with _Served(server) as shared:
        await CodexLogin(auth, resources=shared).sign_in_device(show=list[str]().append)
    # No interval given waits five seconds; one under a second waits one.
    assert waits == [5.0]
    server = _Server(
        replies=[
            _reply({"user_code": "W", "device_auth_id": "d", "interval": 0}),
            _reply({}, status=500),
        ],
    )
    waits.clear()
    async with _Served(server) as shared:
        with pytest.raises(httpx2.HTTPStatusError):
            await CodexLogin(auth, resources=shared).sign_in_device(
                show=list[str]().append,
            )
    assert waits == [1.0]


@pytest.mark.asyncio
async def test_sign_out_revokes_the_refresh_token(auth: Path) -> None:
    _store(
        auth,
        {"access_token": _token("a"), "refresh_token": "r-out", "account_id": "a"},
    )
    server = _Server(replies=[_reply({})])
    async with _Served(server) as shared:
        await CodexLogin(auth, resources=shared).sign_out()
    (sent,) = server.requests
    assert str(sent.url) == "https://auth.openai.com/oauth/revoke"
    assert _json(sent) == {
        "token": "r-out",
        "token_type_hint": "refresh_token",
        "client_id": _CLIENT,
    }


@pytest.mark.parametrize(
    ("identity", "access", "account"),
    [
        (
            _jwt(
                {
                    "email": "me@id.test",
                    "https://api.openai.com/auth": {"chatgpt_plan_type": "pro"},
                },
            ),
            _jwt({"https://api.openai.com/profile": {"email": "me@access.test"}}),
            CodexAccount(plan="pro", email="me@id.test"),
        ),
        (
            "not-a-jwt",
            _jwt({"https://api.openai.com/profile": {"email": "me@access.test"}}),
            CodexAccount(plan="", email="me@access.test"),
        ),
        # Claims of the wrong shape say nothing.
        (
            _jwt({"https://api.openai.com/auth": "pro", "email": 7}),
            _jwt({"https://api.openai.com/auth": {"chatgpt_plan_type": 3}}),
            CodexAccount(plan="", email=""),
        ),
    ],
)
@pytest.mark.asyncio
async def test_the_account_is_what_the_tokens_say_the_id_token_first(
    auth: Path,
    identity: str,
    access: str,
    account: CodexAccount,
) -> None:
    _store(auth, {"access_token": access, "refresh_token": "r", "id_token": identity})
    async with SharedResources() as shared:
        assert await CodexLogin(auth, resources=shared).account() == account


_ISSUER = "https://issuer.test"


def _http(server: _Server) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(server))


@pytest.mark.asyncio
async def test_a_device_sign_in_runs_one_step_per_call() -> None:
    access = _token("device", account="acct-d")
    server = _Server(
        replies=[
            _reply({"usercode": "ABCD", "device_auth_id": "dev-1", "interval": "3"}),
            _reply({}, status=403),
            _reply({"authorization_code": "auth-code", "code_verifier": "verifier"}),
            _reply({"access_token": access, "refresh_token": "r", "id_token": "i"}),
        ],
    )
    async with _http(server) as http:
        code = await request_device_code(http, issuer=_ISSUER)
        pending = await poll_device_code(http, code, issuer=_ISSUER)
        approved = await poll_device_code(http, code, issuer=_ISSUER)

    assert code == DeviceCode(
        verification_url=f"{_ISSUER}/codex/device",
        user_code="ABCD",
        device_auth_id="dev-1",
        interval_sec=3.0,
    )
    assert pending is None
    assert approved is not None
    assert (approved.access_token, approved.refresh_token, approved.id_token) == (
        access,
        "r",
        "i",
    )
    assert approved.account_id == "acct-d"
    assert approved.expires_at == pytest.approx(time.time() + 3_600.0, abs=60)
    assert (
        _form(server.requests[-1])["redirect_uri"] == f"{_ISSUER}/deviceauth/callback"
    )
    assert [str(r.url) for r in server.requests[1:]] == [
        f"{_ISSUER}/api/accounts/deviceauth/token",
        f"{_ISSUER}/api/accounts/deviceauth/token",
        f"{_ISSUER}/oauth/token",
    ]
    assert {
        timeout
        for r in server.requests
        for timeout in from_plain(r.extensions["timeout"], dict[str, float]).values()
    } == {15.0}


@pytest.mark.asyncio
async def test_an_approval_that_is_no_object_is_refused() -> None:
    server = _Server(
        replies=[
            _reply({"user_code": "W", "device_auth_id": "d"}),
            httpx2.Response(200, content=b'["auth-code"]'),
        ],
    )
    async with _http(server) as http:
        code = await request_device_code(http, issuer=_ISSUER)
        with pytest.raises(ValueError, match="dict"):
            _ = await poll_device_code(http, code, issuer=_ISSUER)


@pytest.mark.asyncio
async def test_an_approved_code_issuing_tokens_alone_keeps_nothing_else() -> None:
    access = _jwt({"exp": time.time() + 600.0})
    server = _Server(
        replies=[
            _reply({"authorization_code": "c", "code_verifier": "v"}),
            _reply({"access_token": access}),
        ],
    )
    code = DeviceCode(
        verification_url="",
        user_code="W",
        device_auth_id="d",
        interval_sec=1.0,
    )
    async with _http(server) as http:
        tokens = await poll_device_code(http, code, issuer=_ISSUER)

    assert tokens is not None
    assert tokens == CodexTokens(
        access_token=access,
        refresh_token="",
        account_id="",
        id_token="",
        expires_at=tokens.expires_at,
    )


@pytest.mark.asyncio
async def test_a_device_code_that_is_no_object_is_refused() -> None:
    server = _Server(replies=[httpx2.Response(200, content=b"[]")])
    async with _http(server) as http:
        with pytest.raises(ValueError, match="dict"):
            _ = await request_device_code(http, issuer=_ISSUER)


@pytest.mark.asyncio
async def test_a_device_code_with_no_interval_is_polled_every_five_seconds() -> None:
    server = _Server(replies=[_reply({"user_code": "W", "device_auth_id": "d"})])
    async with _http(server) as http:
        code = await request_device_code(http, issuer=_ISSUER)

    assert code.interval_sec == 5.0


@pytest.mark.asyncio
async def test_a_refresh_keeps_what_its_answer_leaves_out() -> None:
    fresh, old, identity = _token("fresh"), _token("r-old"), _token("id-old")
    server = _Server(replies=[_reply({"access_token": fresh})])
    async with _http(server) as http:
        tokens = await refresh_grant(
            http,
            old,
            account_id="acct",
            id_token=identity,
            issuer=_ISSUER,
        )

    assert tokens == CodexTokens(
        access_token=fresh,
        refresh_token=old,
        account_id="acct",
        id_token=identity,
        expires_at=tokens.expires_at,
    )
    (sent,) = server.requests
    assert str(sent.url) == f"{_ISSUER}/oauth/token"
    assert _json(sent) == _refresh(old)
    assert set(from_plain(sent.extensions["timeout"], dict[str, float]).values()) == {
        15.0,
    }


@pytest.mark.asyncio
async def test_a_refresh_with_nothing_to_keep_leaves_it_empty() -> None:
    # A token without an expiry lives as long as the grant says, even a second.
    plain, held = _jwt({"sub": "plain"}), _token("r")
    server = _Server(replies=[_reply({"access_token": plain, "expires_in": 1})])
    async with _http(server) as http:
        tokens = await refresh_grant(http, held, issuer=_ISSUER)

    assert tokens == CodexTokens(
        access_token=plain,
        refresh_token=held,
        account_id="",
        id_token="",
        expires_at=tokens.expires_at,
    )
    assert tokens.expires_at == pytest.approx(time.time() + 1.0, abs=5)


@pytest.mark.asyncio
async def test_a_grant_that_is_no_object_is_refused() -> None:
    server = _Server(replies=[httpx2.Response(200, content=b"[]")])
    async with _http(server) as http:
        with pytest.raises(ValueError, match="dict"):
            _ = await refresh_grant(http, "r", issuer=_ISSUER)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401])
async def test_a_refused_refresh_is_an_auth_error(status: int) -> None:
    server = _Server(replies=[_reply({}, status=status)])
    async with _http(server) as http:
        with pytest.raises(
            AuthError,
            match=r"^The ChatGPT login has expired or been revoked; sign in again\.$",
        ):
            _ = await refresh_grant(http, "r", account_id="a", issuer=_ISSUER)


@pytest.mark.asyncio
async def test_a_refresh_token_is_revoked_at_the_issuer() -> None:
    out = _token("r-out")
    server = _Server(replies=[_reply({})])
    async with _http(server) as http:
        await revoke_refresh_token(http, out, issuer=_ISSUER)

    (sent,) = server.requests
    assert str(sent.url) == f"{_ISSUER}/oauth/revoke"
    assert _json(sent)["token"] == out
    assert set(from_plain(sent.extensions["timeout"], dict[str, float]).values()) == {
        10.0,
    }


def test_the_account_of_tokens_reads_the_id_token_first() -> None:
    tokens = CodexTokens(
        access_token=_jwt({"https://api.openai.com/profile": {"email": "a@x.test"}}),
        refresh_token=_token("r"),
        account_id="acct",
        id_token=_jwt(
            {
                "email": "i@x.test",
                "https://api.openai.com/auth": {"chatgpt_plan_type": "pro"},
            },
        ),
        expires_at=0.0,
    )

    assert codex_account(tokens) == CodexAccount(plan="pro", email="i@x.test")


@dataclass(slots=True, kw_only=True)
class _Server:
    """Answers each request with the next scripted reply, and keeps the requests."""

    replies: list[httpx2.Response]
    requests: list[httpx2.Request] = field(default_factory=list[httpx2.Request])

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.replies.pop(0)


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


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
