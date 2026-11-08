"""Tests for a login's file: its lock, and the files each named account keeps."""

from __future__ import annotations

from pathlib import Path
from typing import cast, override
from unittest.mock import AsyncMock, Mock, patch

import asyncio
import fcntl
import logging
import os
import stat
import subprocess
import time
import webbrowser

from treekle import json_dumps, json_loads

import pytest

from sagent.lib.userdirs import state_dir
from sagent.model.exceptions import AuthError
from sagent.providers import login
from sagent.providers.login import (
    Logins,
    Spent,
    check_account,
    file_lock,
    headless_wanted,
    open_browser,
    pasted_code,
    read_pasted_code,
    redirected,
)
from sagent.providers.testing import first_live, window


@pytest.mark.asyncio
async def test_cancelled_input_propagates_and_the_next_read_uses_a_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = asyncio.CancelledError()
    first = AsyncMock(side_effect=cancelled)
    second = AsyncMock(return_value="4/code")
    make = Mock(side_effect=[Mock(prompt_async=first), Mock(prompt_async=second)])
    monkeypatch.setattr(login, "PromptSession", make)
    with pytest.raises(asyncio.CancelledError) as raised:
        await login.read_login_input("Code: ")
    assert raised.value is cancelled
    assert await login.read_login_input("Retry: ") == "4/code"
    assert make.call_args_list == [(), ()]
    first.assert_awaited_once_with("Code: ")
    second.assert_awaited_once_with("Retry: ")


@pytest.mark.asyncio
async def test_closed_sign_in_input_raises_eof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eof = EOFError()
    reading = AsyncMock(side_effect=eof)
    make = Mock(return_value=Mock(prompt_async=reading))
    monkeypatch.setattr(login, "PromptSession", make)
    with pytest.raises(EOFError) as raised:
        await login.read_login_input("Code: ")
    assert raised.value is eof
    make.assert_called_once_with()
    reading.assert_awaited_once_with("Code: ")


@pytest.mark.asyncio
@pytest.mark.parametrize("headless", [True, False])
async def test_login_flow_runs_only_the_selected_authorization(
    headless: bool,
) -> None:
    called: list[str] = []

    async def browser() -> str:
        called.append("browser")
        return "browser-code"

    async def remote() -> str:
        called.append("remote")
        return "remote-code"

    code = await login.login_flow(headless=headless, browser=browser, remote=remote)
    selected = "remote" if headless else "browser"
    assert (code, called) == (f"{selected}-code", [selected])


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/callback", "/oauth2callback", "/auth/callback"])
async def test_the_redirect_is_answered_only_on_its_path_with_its_state(
    path: str,
) -> None:
    code = asyncio.Future[str]()
    for request, accepted in (
        (f"GET {path}?code=c1&state=wrong HTTP/1.1\r\n", False),
        ("GET /elsewhere?code=c1&state=s1 HTTP/1.1\r\n", False),
        (f"GET {path}?state=s1 HTTP/1.1\r\n", False),
        ("garbage\r\n", False),
        (f"GET {path}?state=s1&code=\r\n", False),
        (f"GET {path}?code=c1&state=s1\r\n", True),
        (f"GET {path}?code=c2&state=s1 HTTP/1.1\r\n", True),
    ):
        reader = asyncio.StreamReader()
        reader.feed_data(request.encode())
        reader.feed_eof()
        written, drained, closed = Mock(), AsyncMock(), Mock()
        writer = Mock(
            spec=asyncio.StreamWriter,
            write=written,
            drain=drained,
            close=closed,
        )
        await redirected(reader, writer, path=path, state="s1", code=code)
        written.assert_called_once()
        assert cast(bytes, written.call_args_list[0].args[0]).startswith(
            b"HTTP/1.1 200 OK\r\n" if accepted else b"HTTP/1.1 400 Bad Request\r\n",
        )
        drained.assert_awaited_once_with()
        closed.assert_called_once_with()
        assert code.done() is accepted
        if accepted:
            assert code.result() == "c1"


@pytest.mark.asyncio
async def test_the_redirect_tells_the_browser_whether_the_sign_in_took() -> None:
    """A request line that is not UTF-8 is read as near as it decodes, and refused."""
    code = asyncio.Future[str]()
    failed = "Sign-in failed; see the terminal."
    signed_in = "Signed in; you can close this tab."
    for request, response in (
        (
            b"GET /cb\xff?code=c&state=s1 HTTP/1.1\r\n\r\n",
            (
                "HTTP/1.1 400 Bad Request\r\ncontent-type: text/plain\r\n"
                f"content-length: {len(failed)}\r\nconnection: close\r\n\r\n{failed}"
            ),
        ),
        (
            b"GET /cb?code=c&state=s1 HTTP/1.1\r\n\r\n",
            (
                "HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\n"
                f"content-length: {len(signed_in)}\r\nconnection: close\r\n\r\n{signed_in}"
            ),
        ),
    ):
        reader = asyncio.StreamReader()
        reader.feed_data(request)
        reader.feed_eof()
        written, drained, closed = Mock(), AsyncMock(), Mock()
        writer = Mock(
            spec=asyncio.StreamWriter,
            write=written,
            drain=drained,
            close=closed,
        )
        await redirected(reader, writer, path="/cb", state="s1", code=code)
        written.assert_called_once_with(response.encode())
        drained.assert_awaited_once_with()
        closed.assert_called_once_with()
    assert code.result() == "c"


_ANOTHER = r"^The pasted address belongs to another sign-in\.$"
_NO_CODE = r"^The pasted address holds no sign-in code\.$"


@pytest.mark.parametrize(
    ("pasted", "code"),
    [
        ("http://127.0.0.1:1455/oauth2callback?code=c-1&state=s", "c-1"),
        ("c-2#s", "c-2"),
        # Pasted without its scheme, the address still holds the code.
        ("127.0.0.1:1455/oauth2callback?code=c-3&state=s", "c-3"),
    ],
)
def test_a_pasted_address_or_code_yields_its_code(pasted: str, code: str) -> None:
    assert pasted_code(pasted, state="s") == code


@pytest.mark.parametrize(
    ("pasted", "message"),
    [
        ("http://127.0.0.1:1455/oauth2callback?code=c&state=other", _ANOTHER),
        ("http://127.0.0.1:1455/oauth2callback?code=c", _ANOTHER),
        ("c#other", _ANOTHER),
        ("c#x#s", _ANOTHER),
        ("c", _ANOTHER),
        ("http://127.0.0.1:1455/oauth2callback?state=s", _NO_CODE),
        ("http://127.0.0.1:1455/oauth2callback?code=&state=s", _NO_CODE),
        ("#s", _NO_CODE),
        ("", _NO_CODE),
    ],
)
def test_a_pasted_address_must_be_this_sign_ins(pasted: str, message: str) -> None:
    with pytest.raises(AuthError, match=message):
        pasted_code(pasted, state="s")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pasted", "code"),
    [
        (" 4/code ", "4/code"),
        ("https://host/cb?code=4%2Fcode&state=s", "4/code"),
        ("4/code#s", "4/code"),
        ("Xcode", "Xcode"),
    ],
)
async def test_pkce_accepts_a_hosted_code_or_a_state_checked_redirect(
    monkeypatch: pytest.MonkeyPatch,
    pasted: str,
    code: str,
) -> None:
    async def typed(prompt: str) -> str:
        assert prompt == "Code: "
        return pasted

    monkeypatch.setattr(login, "read_login_input", typed)
    shown: list[str] = []
    assert (
        await read_pasted_code(
            state="s",
            show=shown.append,
            browser_tried=False,
            pkce=True,
        )
        == code
    )
    assert shown == [
        "Open the link in your browser, then paste the authorization code or full redirect URL:\n",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pasted",
    ["", "https://host/cb?state=s", "code#wrong", "https://host/cb?code=c&state=wrong"],
)
async def test_pkce_still_refuses_missing_codes_and_mismatched_states(
    monkeypatch: pytest.MonkeyPatch,
    pasted: str,
) -> None:
    async def typed(prompt: str) -> str:
        assert prompt == "Code: "
        return pasted

    monkeypatch.setattr(login, "read_login_input", typed)
    shown: list[str] = []
    with pytest.raises(AuthError):
        await read_pasted_code(
            state="s",
            show=shown.append,
            browser_tried=False,
            pkce=True,
        )


@pytest.mark.parametrize(
    "browser",
    [webbrowser.GenericBrowser, webbrowser.BackgroundBrowser],
)
def test_a_browser_command_opens_the_url_and_never_writes_to_the_terminal(
    browser: type[webbrowser.GenericBrowser],
) -> None:
    command = browser(["browser", "--url=%s", "--fixed"])
    url = "https://example.com/sign-in?x=';exit 1"
    with (
        patch.object(webbrowser, "get", return_value=command),
        patch.object(
            subprocess,
            "run",
        ) as run,
    ):
        open_browser(url)
    run.assert_called_once_with(
        ["/bin/sh", "-c", '"$@" &', "sh", "browser", f"--url={url}", "--fixed"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        check=False,
    )


def test_a_browser_command_that_cannot_start_raises_nothing() -> None:
    command = webbrowser.GenericBrowser(["browser", "%s"])
    with (
        patch.object(webbrowser, "get", return_value=command),
        patch.object(
            subprocess,
            "run",
            side_effect=FileNotFoundError("/bin/sh"),
        ) as run,
    ):
        open_browser("https://example.com")
    run.assert_called_once()


def test_a_browser_that_is_no_command_opens_the_url_its_own_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []

    class _Remote(webbrowser.BaseBrowser):
        @override
        def open(self, url: str, new: int = 0, autoraise: bool = True) -> bool:
            opened.append(url)
            return True

    monkeypatch.setattr(webbrowser, "get", lambda: _Remote("remote"))
    open_browser("https://example.com")
    assert opened == ["https://example.com"]


def test_no_browser_opens_nothing_and_raises_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def none() -> webbrowser.BaseBrowser:
        raise webbrowser.Error("could not locate runnable browser")

    monkeypatch.setattr(webbrowser, "get", none)
    open_browser("https://example.com")


@pytest.mark.asyncio
async def test_a_contended_lock_is_retried_before_admission(tmp_path: Path) -> None:
    lock = tmp_path / "auth.json.lock"
    opened = AsyncMock(return_value=17)
    flock = Mock(side_effect=[BlockingIOError(), None])
    sleeping = AsyncMock()
    closed = Mock()
    with (
        patch.object(asyncio, "to_thread", opened),
        patch.object(fcntl, "flock", flock),
        patch.object(asyncio, "sleep", sleeping),
        patch.object(os, "close", closed),
    ):
        async with file_lock(lock):
            assert flock.call_count == 2
            closed.assert_not_called()
    opened.assert_awaited_once_with(login._open, lock)
    assert flock.call_args_list == [((17, fcntl.LOCK_EX | fcntl.LOCK_NB),)] * 2
    sleeping.assert_awaited_once_with(0.01)
    closed.assert_called_once_with(17)


@pytest.mark.asyncio
async def test_the_lock_file_and_its_directory_are_made_private(tmp_path: Path) -> None:
    lock = tmp_path / "new" / "auth.json.lock"

    async with file_lock(lock):
        pass

    assert lock.stat().st_mode & 0o777 == 0o600
    assert lock.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.asyncio
async def test_a_waiter_cancelled_closes_its_lock(tmp_path: Path) -> None:
    lock = tmp_path / "auth.json.lock"
    cancelled = asyncio.CancelledError()
    opened = AsyncMock(return_value=17)
    flock = Mock(side_effect=BlockingIOError)
    sleeping = AsyncMock(side_effect=cancelled)
    closed = Mock()
    with (
        patch.object(asyncio, "to_thread", opened),
        patch.object(fcntl, "flock", flock),
        patch.object(asyncio, "sleep", sleeping),
        patch.object(os, "close", closed),
        pytest.raises(asyncio.CancelledError) as raised,
    ):
        async with file_lock(lock):
            pytest.fail("A contended lock admitted its cancelled waiter.")
    assert raised.value is cancelled
    opened.assert_awaited_once_with(login._open, lock)
    flock.assert_called_once_with(17, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sleeping.assert_awaited_once_with(0.01)
    closed.assert_called_once_with(17)


def test_an_account_s_file_sits_beside_the_default_one() -> None:
    logins = Logins(Path("/logins/auth.json"))
    assert logins.path("work") == Path("/logins/auth-work.json")
    assert logins.path(None) == logins.path("default") == Path("/logins/auth.json")


@pytest.mark.parametrize(
    ("environ", "platform", "wanted"),
    [
        ({"DISPLAY": ":0"}, "linux", False),
        ({"WAYLAND_DISPLAY": "wayland-0"}, "linux", False),
        ({}, "linux", True),
        ({"DISPLAY": ":0", "SSH_CONNECTION": "1 2 3 4"}, "linux", True),
        ({}, "darwin", False),
        ({"SSH_TTY": "/dev/pts/1"}, "darwin", True),
        ({"SSH_CLIENT": "1 2 3"}, "darwin", True),
    ],
    ids=["x11", "wayland", "no-display", "ssh", "mac", "mac-ssh-tty", "mac-ssh"],
)
def test_headless_is_wanted_over_ssh_or_where_no_screen_shows_a_browser(
    environ: dict[str, str],
    platform: str,
    wanted: bool,
) -> None:
    assert headless_wanted(environ, platform=platform) is wanted


@pytest.mark.parametrize("given", [True, False])
def test_what_the_user_says_of_headless_wins(given: bool) -> None:
    assert headless_wanted({}, platform="linux", given=given) is given
    assert headless_wanted({"SSH_TTY": "x"}, platform="darwin", given=given) is given


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("browser_tried", "asked"),
    [
        (
            False,
            (
                "Open it in a browser anywhere, then paste the address it was sent to,"
                " though it fails to load:\n"
            ),
        ),
        (
            True,
            (
                "No browser came back. Paste the address it was sent to, though it"
                " fails to load:\n"
            ),
        ),
    ],
)
async def test_a_pasted_address_is_asked_for_and_its_code_read(
    monkeypatch: pytest.MonkeyPatch,
    browser_tried: bool,
    asked: str,
) -> None:
    prompts: list[str] = []

    async def typed(prompt: str) -> str:
        prompts.append(prompt)
        return " http://127.0.0.1:1/cb?code=c1&state=s1 "

    monkeypatch.setattr(login, "read_login_input", typed)
    shown: list[str] = []
    code = await read_pasted_code(
        state="s1",
        show=shown.append,
        browser_tried=browser_tried,
    )
    assert (code, shown, prompts) == ("c1", [asked], ["Address: "])


@pytest.mark.parametrize("account", ["../x", "a/b", "-x", "a b", "work.x", "a..b", ""])
def test_an_account_that_is_no_name_is_refused(account: str) -> None:
    with pytest.raises(ValueError, match="is not a name"):
        check_account(account)
    with pytest.raises(ValueError, match="is not a name"):
        Logins(Path("/logins/auth.json")).path(account)


@pytest.mark.parametrize("account", ["Work", "w2", "a_b-C"])
def test_an_account_of_letters_digits_and_separators_is_a_name(account: str) -> None:
    check_account(account)


def _save(logins: Logins, *accounts: str) -> None:
    """Write a login file for each of ``accounts``, ``default`` included."""
    logins.default.parent.mkdir(parents=True, exist_ok=True)
    for account in accounts:
        logins.path(account).write_text("{}")


def test_the_saved_accounts_are_the_files_beside_the_default_one(
    tmp_path: Path,
) -> None:
    logins = Logins(tmp_path / "auth.json")
    assert logins.accounts() == ()
    _save(logins, "default", "work", "home")
    (tmp_path / "auth-.json").write_text("{}")
    (tmp_path / "auth-x.txt").write_text("{}")
    assert logins.accounts() == ("home", "work")
    assert logins.order() == ("default", "home", "work")


@pytest.mark.asyncio
async def test_accounts_come_in_the_order_they_were_added(tmp_path: Path) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "zed", "home", "work")
    await logins.added("zed")
    await logins.added("work")
    await logins.added("zed")
    await logins.added(None)
    assert logins.accounts() == ("zed", "work", "home")
    # Another holder of the same default reads the same ledger.
    assert Logins(tmp_path / "auth.json").order() == ("default", "zed", "work", "home")


@pytest.mark.asyncio
async def test_a_login_signed_out_is_passed_over_until_added_again(
    tmp_path: Path,
) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work")
    await logins.sign_out(None)
    await logins.sign_out(None)
    await logins.sign_out("work")
    assert logins.order() == ()
    await logins.added(None)
    assert logins.order() == ("default",)


@pytest.mark.asyncio
async def test_a_precedence_orders_the_saved_accounts_it_names(tmp_path: Path) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "home", "work")
    await logins.prefer(("work", "gone", "default"))
    assert logins.precedence() == ("work", "gone", "default")
    assert logins.order() == ("work", "default")
    await logins.prefer(None)
    assert logins.precedence() is None
    assert logins.order() == ("default", "home", "work")
    with pytest.raises(ValueError, match="is not a name"):
        await logins.prefer(("../x",))


@pytest.mark.asyncio
async def test_two_families_keep_their_own_ledgers(tmp_path: Path) -> None:
    claude = Logins(tmp_path / "claude" / "auth.json")
    codex = Logins(tmp_path / "codex" / "auth.json")
    _save(claude, "default", "work")
    _save(codex, "default", "work")
    await claude.prefer(("work",))
    assert codex.order() == ("default", "work")


@pytest.mark.parametrize(
    ("window", "model", "covered"),
    [
        ("5-hour", "sonnet-5.5", True),
        ("weekly", "opus-5.5", True),
        ("weekly Opus", "opus-5.5", True),
        ("weekly Opus", "sonnet-5.5", False),
        ("", "opus-5.5", True),
        ("", "sonnet-5.5", False),
        # A second word that names no family of the model spending it is no family.
        ("weekly Oauth_apps", "opus-5.5", True),
        ("weekly Oauth_apps", "sonnet-5.5", True),
    ],
)
def test_a_window_covers_every_model_or_only_its_family(
    window: str,
    model: str,
    covered: bool,
) -> None:
    spent = Spent(window=window, model="opus-5.5", until=10.0)
    assert spent.covers(model) is covered


def test_a_family_is_every_word_after_the_span() -> None:
    """``weekly Opus Fast`` names the family ``opus fast``, not ``fast``."""
    spent = Spent(window="weekly Opus Fast", model="fast-1", until=10.0)
    assert spent.covers("opus-5.5")


@pytest.mark.asyncio
async def test_of_an_accounts_windows_the_one_resetting_last_keeps_it(
    tmp_path: Path,
) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default")
    now = time.time()
    weekly = Spent(window="weekly", model="m", until=now + 200.0)
    await logins.spend(None, Spent(window="5-hour", model="m", until=now + 100.0))
    await logins.spend(None, weekly)
    await logins.spend(None, Spent(window="", model="m", until=now + 50.0))
    assert window(logins, "default", "m", now=now) == weekly


@pytest.mark.asyncio
async def test_clearing_spent_quota_keeps_other_models_and_accounts(
    tmp_path: Path,
) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work")
    until = time.time() + 432_000.0
    for account in (None, "work"):
        await logins.spend(
            account,
            Spent(window="weekly Opus", model="opus-5.5", until=until),
        )
    await logins.spend(None, Spent(window="", model="sonnet-5.5", until=until))
    await logins.clear_spent(None, model="opus-5.5")
    assert window(logins, "default", "opus-5.5", now=time.time()) is None
    assert window(logins, "default", "sonnet-5.5", now=time.time()) is not None
    assert window(logins, "work", "opus-5.5", now=time.time()) is not None
    await logins.clear_spent("absent", model="opus-5.5")
    assert window(logins, "work", "opus-5.5", now=time.time()) is not None


@pytest.mark.asyncio
async def test_a_window_resetting_now_is_spent_no_longer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default")
    monkeypatch.setattr(time, "time", lambda: 1_000.0)
    await logins.spend(None, Spent(window="5-hour", model="m", until=1_000.0))
    kept = json_loads(_ledger().read_text(), dict[str, dict[str, object]])
    assert "spent" not in kept[str(logins.default)]


@pytest.mark.asyncio
async def test_a_sign_out_is_kept_through_the_last_moment_of_its_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passed over until a day has gone by, and kept on the ledger until then."""
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work", "home")
    now = [1_000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    os.utime(logins.path("work"), (1_000.0, 1_000.0))
    await logins.sign_out("work")
    # Its file written at the sign-out, not since: still signed out.
    assert [a for a, _ in logins.ranked("m", now=1_000.0)] == ["default", "home"]
    end = 1_000.0 + logins.signed_out_sec
    assert [a for a, _ in logins.ranked("m", now=end)] == ["default", "home"]
    now[0] = end
    await logins.sign_out("home")
    kept = json_loads(_ledger().read_text(), dict[str, dict[str, dict[str, object]]])
    assert sorted(kept[str(logins.default)]["signed_out_at"]) == ["home", "work"]


@pytest.mark.asyncio
async def test_signing_in_as_the_default_adds_no_account(tmp_path: Path) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work")
    await logins.added(None)
    await logins.added("default")
    await logins.added("work")
    kept = json_loads(_ledger().read_text(), dict[str, dict[str, object]])
    assert kept[str(logins.default)]["added"] == ["work"]


@pytest.mark.asyncio
async def test_a_ledger_that_is_no_object_reads_as_empty_and_says_so(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work")
    _ledger().write_text("[1]")
    with caplog.at_level(logging.WARNING, logger=login.__name__):
        assert logins.order() == ("default", "work")
    assert caplog.messages == ["the logins ledger does not read; starting it afresh"]


@pytest.mark.asyncio
async def test_the_first_live_account_skips_each_spent_until_it_resets(
    tmp_path: Path,
) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "home", "work")
    now = time.time()
    five_hour = Spent(window="5-hour", model="opus-5.5", until=now + 100.0)
    await logins.spend(None, five_hour)
    await logins.spend(
        "home",
        Spent(window="weekly Opus", model="opus-5.5", until=now + 50.0),
    )
    assert first_live(logins, "opus-5.5", now=now) == "work"
    assert first_live(logins, "sonnet-5.5", now=now) == "home"
    assert first_live(logins, "opus-5.5", now=now + 60.0) == "home"
    assert first_live(logins, "opus-5.5", now=now + 100.0) == "default"
    assert window(logins, "default", "opus-5.5", now=now) == five_hour
    assert window(logins, "default", "opus-5.5", now=now + 100.0) is None


@pytest.mark.asyncio
async def test_an_account_put_first_ranks_ahead_of_the_saved_order(
    tmp_path: Path,
) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "home", "work")
    now = time.time()
    spent = Spent(window="5-hour", model="opus-5.5", until=now + 100.0)
    await logins.spend("work", spent)
    assert logins.ranked("opus-5.5", now=now, first="work") == (
        ("work", spent),
        ("default", None),
        ("home", None),
    )


@pytest.mark.asyncio
async def test_no_account_is_live_when_every_one_is_spent(tmp_path: Path) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work")
    now = time.time()
    for account in ("default", "work"):
        await logins.spend(account, Spent(window="5-hour", model="m", until=now + 99))
    assert first_live(logins, "m", now=now) is None


@pytest.mark.asyncio
async def test_the_ledger_is_readable_by_its_owner_alone(tmp_path: Path) -> None:
    """It names the user's accounts and when each ran out."""
    _ledger().write_text("{}")
    _ledger().chmod(0o664)
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default")
    await logins.spend(
        "default",
        Spent(window="5-hour", model="m", until=time.time() + 9),
    )
    assert stat.S_IMODE(_ledger().stat().st_mode) == 0o600


def _ledger() -> Path:
    ledger = state_dir() / "rekursiv-ai" / "sagent" / "logins.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    return ledger


@pytest.mark.asyncio
async def test_a_field_another_version_wrote_is_read_past_and_kept(
    tmp_path: Path,
) -> None:
    """Sessions of two versions share one file; neither may break or erase the other's."""
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work")
    key = str(logins.default)
    _ledger().write_text(
        json_dumps({key: {"precedence": ["work"], "newer": "x"}, "other": {"a": 1}}),
    )
    assert logins.order() == ("work",)
    await logins.added("work")
    assert logins.order() == ("work",)
    kept = json_loads(_ledger().read_text(), dict[str, dict[str, object]])
    assert kept[key]["newer"] == "x"
    assert kept["other"] == {"a": 1}


@pytest.mark.asyncio
async def test_an_entry_that_does_not_read_is_dropped_and_the_rest_kept(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed mark must not stop a sign-in from being noted, and says so."""
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work")
    until = time.time() + 60
    good = {"window": "5-hour", "model": "m", "until": until}
    _ledger().write_text(
        json_dumps(
            {
                str(logins.default): {
                    "spent": {"default": [good, {"window": 5}], "work": "bad"},
                },
            },
        ),
    )
    with caplog.at_level(logging.WARNING, logger=login.__name__):
        assert window(logins, "default", "m", now=until - 1) is not None
    assert caplog.messages == [
        "passing over a ledger entry that does not read: {'window': 5}",
        "passing over a ledger entry that does not read: 'bad'",
    ]
    await logins.added("work")
    assert window(logins, "default", "m", now=until - 1) is not None
    assert window(logins, "work", "m", now=until - 1) is None


@pytest.mark.asyncio
async def test_a_ledger_that_is_no_json_reads_as_empty_and_is_rewritten(
    tmp_path: Path,
) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work")
    _ledger().write_text("{")
    assert logins.order() == ("default", "work")
    await logins.added("work")
    assert json_loads(_ledger().read_text(), dict[str, object])


@pytest.mark.asyncio
async def test_a_login_signed_out_comes_back_once_its_file_is_written_again(
    tmp_path: Path,
) -> None:
    """Signed in again outside sagent, as by ``claude /login``, it is used again."""
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work")
    await logins.sign_out("work")
    assert logins.order() == ("default",)
    later = time.time() + 5
    os.utime(logins.path("work"), (later, later))
    assert logins.order() == ("default", "work")


@pytest.mark.asyncio
async def test_a_login_signed_out_comes_back_after_a_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A login with no file to watch, as macOS keeps the default in its Keychain."""
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "work")
    await logins.sign_out(None)
    assert logins.order() == ("work",)
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now + 86_400 + 1)
    assert logins.order() == ("default", "work")


@pytest.mark.asyncio
async def test_a_sign_out_past_its_day_is_dropped_on_the_next_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work", "home")
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now - 2 * 86_400)
    await logins.sign_out("work")
    monkeypatch.setattr(time, "time", lambda: now)
    await logins.sign_out("home")
    kept = json_loads(_ledger().read_text(), dict[str, dict[str, dict[str, object]]])
    assert list(kept[str(logins.default)]["signed_out_at"]) == ["home"]


@pytest.mark.asyncio
async def test_ranked_judges_a_sign_out_at_the_time_it_is_given(tmp_path: Path) -> None:
    """One ``now`` answers both what is spent and who is signed out."""
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work")
    await logins.sign_out("work")
    now = time.time()
    assert [a for a, _ in logins.ranked("m", now=now)] == ["default"]
    later = now + logins.signed_out_sec + 1
    assert [a for a, _ in logins.ranked("m", now=later)] == ["default", "work"]


@pytest.mark.parametrize("asked", ["order", "ranked"])
def test_one_answer_reads_the_ledger_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    asked: str,
) -> None:
    """Read once per account, the answer cost a read for each, and they could disagree."""
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work", "home")
    reads: list[str] = []
    raw = login._raw

    def counted(text: str) -> dict[str, dict[str, object]]:
        reads.append(text)
        return raw(text)

    monkeypatch.setattr(login, "_raw", counted)
    if asked == "order":
        logins.order()
    else:
        logins.ranked("m", now=time.time())
    assert len(reads) == 1


@pytest.mark.asyncio
async def test_a_mark_past_its_reset_is_dropped_on_the_next_write(
    tmp_path: Path,
) -> None:
    logins = Logins(tmp_path / "auth.json")
    _save(logins, "default", "work")
    await logins.spend("work", Spent(window="5-hour", model="m", until=1.0))
    await logins.spend("default", Spent(window="5-hour", model="m", until=9e12))
    ledger = state_dir() / "rekursiv-ai" / "sagent" / "logins.json"
    kept = json_loads(ledger.read_text(), dict[str, dict[str, dict[str, object]]])
    assert list(kept[str(logins.default)]["spent"]) == ["default"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
