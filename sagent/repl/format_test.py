"""Tests for ``repl.format``: terminal-formatting helpers."""

from __future__ import annotations

from typing import cast, override

import io

from rich.console import Console

import pytest

from sagent.repl.format import (
    format_count,
    format_elapsed,
    print_user_bar,
    set_terminal_title,
)


class _TtyBuf(io.StringIO):
    """``StringIO`` that reports itself as a TTY for OSC-title tests."""

    @override
    def isatty(self) -> bool:
        return True


def _capture_console(width: int = 80) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return (
        Console(
            file=buf,
            width=width,
            force_terminal=False,
            color_system=None,
            highlight=False,
        ),
        buf,
    )


def test_print_user_bar_one_line() -> None:
    con, buf = _capture_console()
    print_user_bar(con, "hello")
    out = buf.getvalue()
    assert "> hello" in out


def test_print_user_bar_multi_line() -> None:
    con, buf = _capture_console()
    print_user_bar(con, "line 1\nline 2")
    out = buf.getvalue()
    assert "> line 1" in out
    assert "  line 2" in out


def test_print_user_bar_narrow_terminal_path() -> None:
    # The ``width <= 2`` path uses a single ``console.print(Text(...))``.
    # Verify by mocking a console with an invalid width attribute -- the
    # ``int(out.width)`` cast raises and the function falls into the
    # ``width = 0`` early-exit branch.
    captured: list[str] = []

    class Stub:
        width: object = "not-an-int"

        def print(
            self,
            text: object,
            style: str = "",
        ) -> None:
            del style
            captured.append(str(text))

    print_user_bar(cast("Console", Stub()), "hi")
    assert any("> hi" in line for line in captured)


def test_print_user_bar_wraps_long_lines() -> None:
    con, buf = _capture_console(width=10)
    print_user_bar(con, "abcdefghijklmnop")
    out = buf.getvalue()
    # Output split across multiple lines.
    assert out.count("\n") >= 2


def test_set_terminal_title_noop_when_not_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Capture stderr writes; isatty returns False by default for StringIO.
    sio = io.StringIO()
    monkeypatch.setattr("sys.stderr", sio)
    set_terminal_title("title")
    assert sio.getvalue() == ""


def test_set_terminal_title_writes_on_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sio = _TtyBuf()
    monkeypatch.setattr("sys.stderr", sio)
    set_terminal_title("hello world")
    assert "\x1b]0;hello world\x07" in sio.getvalue()


def test_set_terminal_title_truncates_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sio = _TtyBuf()
    monkeypatch.setattr("sys.stderr", sio)
    set_terminal_title("x" * 200, max_len=20)
    written = sio.getvalue()
    # The OSC-0 escape contains a truncated 20-char title.
    assert "\x1b]0;" in written
    # 19 x's plus ellipsis = 20 chars.
    assert "xxxxxxxxxxxxxxxxxxx…" in written


def test_set_terminal_title_replaces_newlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sio = _TtyBuf()
    monkeypatch.setattr("sys.stderr", sio)
    set_terminal_title("first\nsecond")
    assert "\nfirst" not in sio.getvalue()
    assert "first second" in sio.getvalue()


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "0s"),
        (0.4, "0s"),
        (12.0, "12s"),
        (83.0, "1m 23s"),
        (3_600.0, "1h 0m 0s"),
        (3_904.0, "1h 5m 4s"),
        (86_400.0, "1d 0h 0m 0s"),
        (90_125.0, "1d 1h 2m 5s"),
    ],
)
def test_format_elapsed(seconds: float, expected: str) -> None:
    assert format_elapsed(seconds) == expected


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "0"),
        (412, "412"),
        (9_999, "9999"),
        (10_000, "10K"),
        (12_345, "12K"),
        (999_499, "999K"),
        (999_500, "1.0M"),
        (1_800_000, "1.8M"),
    ],
)
def test_format_count(n: int, expected: str) -> None:
    assert format_count(n) == expected


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
