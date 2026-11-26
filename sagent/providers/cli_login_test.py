"""Tests for signing in through a vendor CLI's own sign-in, at sagent's prompt."""

from __future__ import annotations

from pathlib import Path

import asyncio
import os
import signal

import pytest

from sagent.model.exceptions import AuthError
from sagent.providers import cli_login
from sagent.providers.cli_login import cli_sign_in
from sagent.providers.testing import refuse_browsers


_URL = "https://sign.in/oauth?a=1&state=s"
_GRACE_SEC = 0.05


@pytest.fixture(autouse=True)
def no_real_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a test that opens the user's browser without meaning to."""
    refuse_browsers(monkeypatch, cli_login)


@pytest.mark.asyncio
@pytest.mark.parametrize("browser", [True, False])
async def test_the_code_typed_at_sagent_s_prompt_reaches_the_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    browser: bool,
) -> None:
    cli = _cli(
        tmp_path,
        "echo 'Opening browser to sign in...'",
        f'echo "If the browser didn\'t open, visit: {_URL}"',
        "printf 'Paste code here if prompted > '",
        "read code",
        'echo "$code $GREETING" > "$0.out"',
    )
    prompts: list[str] = []
    opened: list[str] = []

    async def typed(prompt: str) -> str:
        prompts.append(prompt)
        return "the-code"

    monkeypatch.setattr(cli_login, "read_login_input", typed)
    if browser:
        monkeypatch.setattr(cli_login, "open_browser", opened.append)
    shown: list[str] = []
    await cli_sign_in(
        cli,
        env={"PATH": os.environ["PATH"], "GREETING": "hi"},
        browser=browser,
        show=shown.append,
        grace_sec=_GRACE_SEC,
    )
    assert _read(f"{cli}.out") == "the-code hi\n"
    assert shown == [
        f"Sign in at this address, then paste the code it shows:\n\n  {_URL}\n",
    ]
    assert prompts == ["Code: "]
    assert opened == ([_URL] if browser else [])


@pytest.mark.asyncio
async def test_a_cli_that_finishes_the_sign_in_itself_ends_the_prompt_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """As ``claude`` does when the browser comes back to its own loopback server."""
    cli = _cli(tmp_path, f"echo 'visit: {_URL}'", "exit 0")
    released = asyncio.Event()

    async def waiting(prompt: str) -> str:
        del prompt
        try:
            await asyncio.Event().wait()
        finally:
            # A prompt that takes some turns of the loop to give the terminal back.
            for _ in range(20):
                await asyncio.sleep(0)
            released.set()
        return ""

    monkeypatch.setattr(cli_login, "read_login_input", waiting)
    await cli_sign_in(cli, browser=False, show=_unseen, grace_sec=_GRACE_SEC)
    assert released.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("said", "message"),
    [
        (
            "echo \"Couldn't log in: That code wasn't accepted.\" >&2",
            r"^cli sign-in failed: Couldn't log in: That code wasn't accepted\.$",
        ),
        (r"printf 'bad \377 code\n' >&2", "^cli sign-in failed: bad \ufffd code$"),
    ],
    ids=["stderr", "not-utf8"],
)
async def test_a_refused_code_raises_what_the_cli_said_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    said: str,
    message: str,
) -> None:
    cli = _cli(
        tmp_path,
        f"echo '  {_URL}'",
        "printf 'Paste code here if prompted > '",
        "read code",
        said,
        "exit 1",
    )
    monkeypatch.setattr(cli_login, "read_login_input", _typed)
    with pytest.raises(AuthError, match=message):
        await cli_sign_in(cli, browser=False, show=_unseen, grace_sec=_GRACE_SEC)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("said", "message"),
    [
        (
            "echo 'zeroth'; echo 'first'; echo 'no network'",
            r"^cli sign-in failed: no network$",
        ),
        (":", r"^cli sign-in failed \(exit 3\)\.$"),
    ],
    ids=["its-last-words", "silent"],
)
async def test_a_cli_that_exits_before_any_address_raises(
    tmp_path: Path,
    said: str,
    message: str,
) -> None:
    shown: list[str] = []
    with pytest.raises(AuthError, match=message):
        await cli_sign_in(
            _cli(tmp_path, said, "exit 3"),
            browser=False,
            show=shown.append,
            grace_sec=_GRACE_SEC,
        )
    assert shown == []


@pytest.mark.asyncio
async def test_a_failure_is_raised_though_a_browser_it_started_holds_its_pipes(
    tmp_path: Path,
) -> None:
    """``claude`` starts the browser, which may keep its output open past its exit."""
    cli = _cli(
        tmp_path,
        'sleep 30 & echo $! > "$0.pid"',
        "echo 'no network' >&2",
        "exit 3",
    )
    try:
        with pytest.raises(AuthError, match=r"^cli sign-in failed: no network$"):
            await cli_sign_in(cli, browser=False, show=_unseen, grace_sec=_GRACE_SEC)
    finally:
        os.kill(int(_read(f"{cli}.pid")), signal.SIGKILL)


@pytest.mark.asyncio
async def test_a_cli_that_signs_in_without_an_address_succeeds(tmp_path: Path) -> None:
    shown: list[str] = []
    await cli_sign_in(
        _cli(tmp_path, "echo 'Already signed in.'"),
        browser=False,
        show=shown.append,
        grace_sec=_GRACE_SEC,
    )
    assert shown == []


@pytest.mark.asyncio
async def test_no_answer_at_the_prompt_stops_the_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli(
        tmp_path,
        'echo $$ > "$0.pid"',
        f"echo '{_URL}'",
        "read code",
        "sleep 30",
    )

    async def closed(prompt: str) -> str:
        del prompt
        raise EOFError

    monkeypatch.setattr(cli_login, "read_login_input", closed)
    with pytest.raises(EOFError):
        await cli_sign_in(cli, browser=False, show=_unseen, grace_sec=_GRACE_SEC)
    _assert_gone(int(_read(f"{cli}.pid")))


@pytest.mark.asyncio
async def test_a_cancelled_sign_in_stops_the_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli(tmp_path, 'echo $$ > "$0.pid"', f"echo '{_URL}'", "read code")
    asked = asyncio.Event()

    async def waiting(prompt: str) -> str:
        del prompt
        asked.set()
        await asyncio.Event().wait()
        return ""

    monkeypatch.setattr(cli_login, "read_login_input", waiting)
    running = asyncio.create_task(
        cli_sign_in(cli, browser=False, show=_unseen, grace_sec=_GRACE_SEC),
    )
    prompted = asyncio.create_task(asked.wait())
    await asyncio.wait((running, prompted), return_when=asyncio.FIRST_COMPLETED)
    assert prompted.done()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    _assert_gone(int(_read(f"{cli}.pid")))


def _cli(tmp_path: Path, *lines: str) -> str:
    """Return a CLI named ``cli`` that runs ``lines`` in the shell."""
    path = tmp_path / "cli"
    path.write_text("\n".join(("#!/bin/sh", *lines)) + "\n")
    path.chmod(0o700)
    return str(path)


def _read(path: str) -> str:
    return Path(path).read_text()


def _unseen(text: str) -> None:
    del text


async def _typed(prompt: str) -> str:
    del prompt
    return "the-code"


def _assert_gone(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
