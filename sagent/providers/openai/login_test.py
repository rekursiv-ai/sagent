"""Tests for the ChatGPT login: reading, renewing, lending, and signing in, offline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, override
from urllib.parse import parse_qs, urlparse

import asyncio
import base64
import builtins
import fcntl
import hashlib
import os
import socket
import stat
import time
import webbrowser

from treekle import json_dumps, json_loads

import httpx2
import pytest

from sagent.model.exceptions import AuthError
from sagent.providers.openai import login
from sagent.providers.openai.lease import LeasedToken
from sagent.providers.openai.login import (
    CodexAccount,
    CodexLogin,
    CodexTokens,
    DeviceCode,
    account_of,
    poll_device_code,
    refresh_grant,
    request_device_code,
    revoke_refresh_token,
)
from sagent.providers.resources import SharedResources


if TYPE_CHECKING:
    from collections.abc import Mapping


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


# ---- where the login lives ---------------------------------------------------


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


# ---- the token ---------------------------------------------------------------


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
    }
    assert _mode(auth) == 0o600
    assert (auth.parent / "auth.json.lock").exists()


@pytest.mark.asyncio
async def test_a_token_with_more_than_five_minutes_left_is_kept(auth: Path) -> None:
    access = _token("a", expires_in_sec=310.0)
    _store(auth, {"access_token": access, "refresh_token": "r", "account_id": "acct"})
    server = _Server(replies=[])
    async with _Served(server) as shared:
        assert await CodexLogin(auth, resources=shared).token() == (access, "acct")
    assert server.requests == []


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
async def test_renew_waits_for_another_process_holding_the_login(auth: Path) -> None:
    _store(
        auth,
        {"access_token": _token("a"), "refresh_token": "r1", "account_id": "x"},
    )
    fresh = _token("fresh")
    server = _Server(replies=[_reply({"access_token": fresh, "refresh_token": "r2"})])
    async with _Served(server) as shared:
        codex = CodexLogin(auth, resources=shared)
        await codex.token()
        held = _hold(auth.with_suffix(".json.lock"))
        renewing = asyncio.create_task(codex.renew())
        await asyncio.sleep(0.05)
        assert not renewing.done()
        assert server.requests == []
        os.close(held)
        await renewing
        assert await codex.token() == (fresh, "x")
    assert len(server.requests) == 1


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
        ('{"auth_mode": "apikey", "OPENAI_API_KEY": "sk"}', "holds an API key"),
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
        with pytest.raises(AuthError, match="no expiry"):
            await codex.renew()


@pytest.mark.parametrize(
    "token",
    [
        "a.bm90IGpzb24.c",  # A payload that is not JSON.
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
    login._write_private(auth, "{}")
    assert _mode(auth) == 0o600
    assert auth.read_text() == "{}"


# ---- lending -----------------------------------------------------------------


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


# ---- signing in --------------------------------------------------------------


@dataclass(slots=True, kw_only=True)
class _Paster:
    """Answers the paste prompt with the redirect for the sign-in it was shown."""

    pasted: Callable[[str, str], str]
    shown: list[str] = field(default_factory=list[str])

    def show(self, text: str) -> None:
        self.shown.append(text)

    def input(self, prompt: str = "") -> str:
        del prompt
        query = parse_qs(urlparse(_shown_url(self.shown)).query)
        return self.pasted(query["state"][0], query["code_challenge"][0])


def _shown_url(shown: list[str]) -> str:
    return next(
        line.strip()
        for text in shown
        for line in text.splitlines()
        if "authorize?" in line
    )


def _stale(state: str, challenge: str) -> str:
    del state, challenge
    return "the-code#another-state"


def _hashed(state: str, challenge: str) -> str:
    del challenge
    return f"the-code#{state}"


def _no_code(state: str, challenge: str) -> str:
    del challenge
    return f"http://localhost:1455/auth/callback?state={state}"


async def _no_browser(url: str, *, state: str, show: Callable[[str], object]) -> str:
    del url, state, show
    return ""


@pytest.mark.asyncio
async def test_sign_in_exchanges_a_pasted_redirect_and_stores_the_login(
    auth: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenges: list[str] = []

    def pasted(state: str, challenge: str) -> str:
        challenges.append(challenge)
        return f"http://localhost:1455/auth/callback?code=the-code&state={state}"

    paster = _Paster(pasted=pasted)
    monkeypatch.setattr(login, "_browser_code", _no_browser)
    monkeypatch.setattr(builtins, "input", paster.input)
    access = _token("signed-in", account="acct-9")
    server = _Server(
        replies=[
            _reply({"access_token": access, "refresh_token": "r", "id_token": "i"}),
        ],
    )
    async with _Served(server) as shared:
        codex = CodexLogin(auth, resources=shared)
        await codex.sign_in(show=paster.show)
        assert await codex.token() == (access, "acct-9")
    (sent,) = server.requests
    form = _form(sent)
    verifier = form.pop("code_verifier")
    assert form == {
        "grant_type": "authorization_code",
        "code": "the-code",
        "redirect_uri": "http://localhost:1455/auth/callback",
        "client_id": _CLIENT,
    }
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    assert challenges == [expected.rstrip(b"=").decode()]
    parsed = urlparse(_shown_url(paster.shown))
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://auth.openai.com/oauth/authorize"
    )
    params = {key: values[0] for key, values in parse_qs(parsed.query).items()}
    assert params == {
        "client_id": _CLIENT,
        "response_type": "code",
        "redirect_uri": "http://localhost:1455/auth/callback",
        "scope": "openid profile email offline_access api.connectors.read api.connectors.invoke",
        "code_challenge": challenges[0],
        "code_challenge_method": "S256",
        "state": params["state"],
        "codex_cli_simplified_flow": "true",
        "id_token_add_organizations": "true",
        "originator": "codex_cli_rs",
    }
    assert _tokens(auth) == {
        "access_token": access,
        "refresh_token": "r",
        "id_token": "i",
        "account_id": "acct-9",
    }


@pytest.mark.parametrize("pasted", [_stale, _no_code])
@pytest.mark.asyncio
async def test_a_pasted_code_must_belong_to_this_sign_in(
    auth: Path,
    monkeypatch: pytest.MonkeyPatch,
    pasted: Callable[[str, str], str],
) -> None:
    monkeypatch.setattr(login, "_browser_code", _no_browser)
    paster = _Paster(pasted=pasted)
    monkeypatch.setattr(builtins, "input", paster.input)
    server = _Server(replies=[])
    async with _Served(server) as shared:
        with pytest.raises(AuthError, match="another sign-in"):
            await CodexLogin(auth, resources=shared).sign_in(show=paster.show)
    assert server.requests == []


@pytest.mark.asyncio
async def test_a_pasted_code_and_state_alone_are_enough(
    auth: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(login, "_browser_code", _no_browser)
    paster = _Paster(pasted=_hashed)
    monkeypatch.setattr(builtins, "input", paster.input)
    server = _Server(
        replies=[_reply({"access_token": _token("x"), "refresh_token": "r"})],
    )
    async with _Served(server) as shared:
        await CodexLogin(auth, resources=shared).sign_in(show=paster.show)
    assert _form(server.requests[0])["code"] == "the-code"


@pytest.mark.asyncio
async def test_sign_in_takes_the_browsers_code_without_a_paste(
    auth: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def browser(url: str, *, state: str, show: Callable[[str], object]) -> str:
        del url, state, show
        return "from-browser"

    def no_paste(prompt: str = "") -> str:
        raise AssertionError(prompt)

    monkeypatch.setattr(login, "_browser_code", browser)
    monkeypatch.setattr(builtins, "input", no_paste)
    server = _Server(
        replies=[_reply({"access_token": _token("x"), "refresh_token": "r"})],
    )
    async with _Served(server) as shared:
        await CodexLogin(auth, resources=shared).sign_in(show=list[str]().append)
    assert _form(server.requests[0])["code"] == "from-browser"


@pytest.mark.asyncio
async def test_a_headless_sign_in_opens_no_browser_and_takes_the_paste(
    auth: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def browser(url: str, *, state: str, show: Callable[[str], object]) -> str:
        raise AssertionError((url, state, show))

    paster = _Paster(pasted=_hashed)
    monkeypatch.setattr(login, "_browser_code", browser)
    monkeypatch.setattr(builtins, "input", paster.input)
    server = _Server(
        replies=[_reply({"access_token": _token("x"), "refresh_token": "r"})],
    )
    async with _Served(server) as shared:
        await CodexLogin(auth, resources=shared).sign_in(
            headless=True,
            show=paster.show,
        )
    assert _form(server.requests[0])["code"] == "the-code"


@pytest.mark.asyncio
async def test_the_browser_redirects_back_with_the_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = _free_port()
    opened: list[str] = []
    visits: list[asyncio.Task[int]] = []

    def open_in_browser(url: str) -> bool:
        opened.append(url)
        loop = asyncio.get_running_loop()
        visits.extend(
            loop.create_task(_get(port, target))
            for target in (
                "/auth/callback?code=c1&state=wrong",
                "/auth/callback?code=c1&state=s1",
            )
        )
        return True

    monkeypatch.setattr(webbrowser, "open", open_in_browser)
    shown: list[str] = []
    code = await login._browser_code(
        "https://sign.in",
        state="s1",
        show=shown.append,
        port=port,
    )
    assert code == "c1"
    assert opened == ["https://sign.in"]
    assert shown == ["Sign in to ChatGPT in your browser:\n\n  https://sign.in\n"]
    assert [await visit for visit in visits] == [400, 200]


@pytest.mark.asyncio
async def test_a_browser_that_never_redirects_gives_way_to_a_paste(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(webbrowser, "open", _nowhere)
    code = await login._browser_code(
        "https://sign.in",
        state="s",
        show=list[str]().append,
        port=_free_port(),
        timeout_sec=0.01,
    )
    assert code == ""


@pytest.mark.asyncio
async def test_the_redirect_is_answered_only_on_its_path_with_its_state() -> None:
    code = asyncio.Future[str]()
    server = await asyncio.start_server(
        lambda reader, writer: login._redirected(reader, writer, state="s1", code=code),
        "127.0.0.1",
        0,
    )
    port = _port(server.sockets[0])
    async with server:
        assert await _get(port, "/auth/callback?code=c1&state=wrong") == 400
        assert await _get(port, "/elsewhere?code=c1&state=s1") == 400
        assert await _get(port, "/auth/callback?state=s1") == 400
        assert await _raw(port, b"garbage\r\n") == 400
        assert not code.done()
        assert await _get(port, "/auth/callback?code=c1&state=s1") == 200
        # A second redirect after the first is answered and changes nothing.
        assert await _get(port, "/auth/callback?code=c2&state=s1") == 200
    assert code.result() == "c1"


@pytest.mark.asyncio
async def test_a_browser_sign_in_gives_way_when_its_port_is_taken() -> None:
    shown: list[str] = []
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = _port(taken)
        assert (
            await login._browser_code(
                "https://x",
                state="s",
                show=shown.append,
                port=port,
            )
            == ""
        )
    assert shown == []


@pytest.mark.asyncio
async def test_sign_in_device_polls_until_the_code_is_approved(
    auth: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(delay_sec: float) -> None:
        waits.append(delay_sec)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)
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
        await codex.sign_in_device(show=shown.append)
        assert await codex.token() == (access, "acct-d")
    assert shown == [
        "Open https://auth.openai.com/codex/device and enter the code ABCD-1234.\n",
    ]
    assert waits == [3.0, 3.0, 3.0]
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


# ---- the account -------------------------------------------------------------


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


# ---- the steps, for a server that runs them across its own requests ----------

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
    assert _json(server.requests[0]) == {**_refresh(old)}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401])
async def test_a_refused_refresh_is_an_auth_error(status: int) -> None:
    server = _Server(replies=[_reply({}, status=status)])
    async with _http(server) as http:
        with pytest.raises(AuthError, match="sign in again"):
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

    assert account_of(tokens) == CodexAccount(plan="pro", email="i@x.test")


def _nowhere(url: str) -> bool:
    """Open ``url`` in a browser that never comes back."""
    del url
    return True


def _hold(lock: Path) -> int:
    """Take the login's lock, as another process renewing it would."""
    held = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(held, fcntl.LOCK_EX)
    return held


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return _port(probe)


def _port(bound: _Bound) -> int:
    return bound.getsockname()[1]


class _Bound(Protocol):
    """An IPv4 socket, whose address typeshed leaves untyped."""

    def getsockname(self) -> tuple[str, int]: ...


async def _get(port: int, target: str) -> int:
    """Return the status a GET of ``target`` on ``port`` answers with."""
    return await _raw(
        port,
        f"GET {target} HTTP/1.1\r\nhost: localhost\r\n\r\n".encode(),
    )


async def _raw(port: int, request: bytes) -> int:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request)
    await writer.drain()
    status = int((await reader.readline()).split()[1])
    writer.close()
    await writer.wait_closed()
    return status


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
