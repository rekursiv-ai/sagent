"""Tests for the Rich scrollback: one left edge, bars, rows, and their colors."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import io
import re
import sys

from rich.console import Console
from rich.style import Style

import pytest

from sagent.runtime import types
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tui.console import ConsolePrinter, ConsoleStyle, Title


if TYPE_CHECKING:
    from sagent.tui.types import Printer


class _Screen:
    """A printer over an in-memory console, read back plain or with its codes."""

    def __init__(self, *, width: int = 40, color: bool = False, **options: int) -> None:
        self.out = io.StringIO()
        console = (
            Console(
                file=self.out,
                width=width,
                color_system="truecolor",
                force_terminal=True,
            )
            if color
            else Console(file=self.out, width=width, color_system=None, highlight=False)
        )
        self.printer = ConsolePrinter(
            console,
            max_label_lines=options.get("max_label_lines", 12),
        )

    def rows(self) -> list[str]:
        return [row.rstrip() for row in self.out.getvalue().split("\n")[:-1]]

    def raw(self) -> list[str]:
        return self.out.getvalue().split("\n")[:-1]


class _Tty(io.StringIO):
    @override
    def isatty(self) -> bool:
        return True


class _BrokenTty(_Tty):
    @override
    def write(self, s: str) -> int:
        raise OSError("gone")


def _codes(row: str) -> list[str]:
    return [str(m.group(1)) for m in re.finditer(r"\x1b\[([0-9;]*)m", row)]


def _visible(row: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", row)


def test_it_is_a_printer_that_shows_thinking_by_default() -> None:
    printer: Printer = ConsolePrinter(Console(file=io.StringIO()))
    assert printer.show_thinking is True
    assert (
        ConsolePrinter(Console(file=io.StringIO()), show_thinking=False).show_thinking
        is False
    )


def test_a_line_is_written_as_it_is() -> None:
    screen = _Screen(color=True)
    screen.printer.line("[bold]not markup[/bold] 123")
    (row,) = screen.raw()
    assert row == "[bold]not markup[/bold] 123"


def test_a_note_is_dim_one_row_per_line() -> None:
    screen = _Screen(color=True)
    screen.printer.note("a\nb")
    screen.printer.note("")
    assert [_visible(r) for r in screen.raw()] == ["a", "b", ""]
    assert all(_codes(r)[0] == "2" for r in screen.raw()[:2])


def test_a_user_bar_fills_the_width_with_a_prompt_on_its_first_row() -> None:
    screen = _Screen(width=12, color=True)
    screen.printer.user("hello\nworld wide web\n")
    raw = screen.raw()
    assert [_visible(r) for r in raw] == [
        "",
        "> hello     ",
        "  world wide",
        "   web      ",
    ]
    assert all("48;2;55;55;55" in _codes(r)[0] for r in raw[1:])
    assert all(_codes(r)[0].startswith("37;") for r in raw[1:])


def test_a_user_bar_stops_at_80_in_a_wide_window() -> None:
    screen = _Screen(width=200, color=True)
    screen.printer.user("hi\n")
    assert [len(_visible(r)) for r in screen.raw()] == [0, 80]


def _fills(screen: _Screen) -> list[int]:
    """Return each filled row's width, in order, runs of one width merged."""
    widths = [len(_visible(r)) for r in screen.raw() if len(_visible(r)) >= 80]
    return [w for i, w in enumerate(widths) if i == 0 or widths[i - 1] != w]


def test_a_turn_s_fills_widen_to_the_widest_so_far_and_never_narrow() -> None:
    screen = _Screen(width=200, color=True)
    screen.printer.markdown("```\nx = 1\n```")
    screen.printer.markdown(f"```\n{'y' * 90}\n```")
    screen.printer.diff("@@ -1 +1 @@\n-a\n+b\n", path="t.py")
    screen.printer.markdown(f"```\n{'w' * 95}\n```")
    screen.printer.markdown("```\nz = 2\n```")
    assert _fills(screen) == [80, 92, 97]


def test_a_user_message_narrows_the_next_turn_s_fills_again() -> None:
    screen = _Screen(width=200, color=True)
    screen.printer.markdown(f"```\n{'y' * 90}\n```")
    screen.printer.user("next\n")
    screen.printer.diff("@@ -1 +1 @@\n-a\n+b\n", path="t.py")
    screen.printer.markdown("```\nz = 2\n```")
    assert _fills(screen) == [92, 80]


def test_an_empty_user_message_still_shows_its_bar() -> None:
    screen = _Screen(width=6)
    screen.printer.user("")
    screen.printer.user("a\n\nb")
    assert screen.rows() == ["", ">", "", "> a", "", "  b"]


def test_a_user_bar_on_a_tiny_console_is_one_unwrapped_row() -> None:
    screen = _Screen(width=2, color=True)
    screen.printer.user("hi")
    raw = screen.raw()
    assert "".join(_visible(r) for r in raw) == "> hi"
    assert "48;2;55;55;55" in _codes(raw[1])[0]


def test_a_wide_character_wraps_by_cells() -> None:
    screen = _Screen(width=6)
    screen.printer.user("日本語です")
    assert screen.rows() == ["", "> 日本", "  語で", "  す"]


def test_another_agent_s_words_are_a_bar_naming_sender_and_recipient() -> None:
    screen = _Screen(width=24, color=True)
    screen.printer.agent("w", "Agent", "done\nall")
    screen.printer.agent("w", "user", "pong")
    screen.printer.agent("w", None, "check it")
    raw = screen.raw()
    assert [_visible(r) for r in raw] == [
        "",
        "w→Agent: done           ",
        "  all                   ",
        "",
        "w→user: pong            ",
        "",
        "w→me: check it          ",
    ]
    to_agent, to_user = _codes(raw[1])[0], _codes(raw[4])[0]
    assert "48;2;55;55;55" not in to_agent
    assert to_agent != to_user


def test_markdown_follows_a_blank_line_without_one_of_its_own() -> None:
    screen = _Screen()
    screen.printer.markdown("- one\n- two")
    assert screen.rows() == ["", " • one", " • two"]


def test_thinking_is_headed_and_indented() -> None:
    screen = _Screen(color=True)
    screen.printer.thinking("why\nbecause", opens=True, closes=True)
    raw = screen.raw()
    assert [_visible(r) for r in raw] == ["∴ Thinking", "  why", "  because"]
    assert _codes(raw[0])[0] == "2;3"
    assert _codes(raw[1])[0] == "2"


def test_streamed_thinking_rows_join_into_the_whole_block() -> None:
    whole, streamed = _Screen(), _Screen()
    whole.printer.thinking("a\n\nb\nc", opens=True, closes=True)
    streamed.printer.thinking("", opens=True, closes=False)
    streamed.printer.thinking("a\n\n", opens=False, closes=False)
    streamed.printer.thinking("b\nc", opens=False, closes=True)
    assert streamed.raw() == whole.raw()
    assert whole.rows() == ["∴ Thinking", "  a", "", "  b", "  c"]


def test_a_call_header_is_indented_and_its_input_follows_the_mark() -> None:
    screen = _Screen(color=True)
    screen.printer.call("Bash list files\nls -la\ncd /", display=ToolDisplay())
    raw = screen.raw()
    assert [_visible(r) for r in raw] == [
        "  Bash list files",
        "  ⎿  ls -la",
        "     cd /",
    ]
    assert _codes(raw[0])[0] == "2"
    assert all("2" in _codes(r)[-2].split(";") for r in raw[1:])


def test_a_call_with_no_input_is_its_header_alone() -> None:
    screen = _Screen()
    screen.printer.call("Read a.py", display=ToolDisplay())
    screen.printer.call("", display=ToolDisplay())
    assert screen.rows() == ["  Read a.py", ""]


def test_a_long_header_wraps_at_the_indent() -> None:
    screen = _Screen(width=12)
    screen.printer.call("abcdefghijklmnop", display=ToolDisplay())
    assert screen.rows() == ["  abcdefghij", "  klmnop"]


def test_a_wrapped_header_hangs_under_its_argument() -> None:
    screen = _Screen(width=20)
    screen.printer.call("Grep " + "y" * 30, display=ToolDisplay())
    assert screen.rows() == [
        "  Grep " + "y" * 13,
        "       " + "y" * 13,
        "       " + "y" * 4,
    ]


def test_the_header_is_capped_and_says_how_many_rows_it_hid() -> None:
    screen = _Screen(width=24, max_label_lines=3)
    screen.printer.call("a" * 110, display=ToolDisplay())
    assert screen.rows() == ["  " + "a" * 22, "  " + "a" * 22, "  ... (3 more lines)"]


def test_a_header_at_the_cap_is_kept_whole() -> None:
    screen = _Screen(width=12, max_label_lines=2)
    screen.printer.call("a" * 20, display=ToolDisplay())
    assert screen.rows() == ["  aaaaaaaaaa", "  aaaaaaaaaa"]


def test_the_input_rows_follow_the_display_and_its_language() -> None:
    display = ToolDisplay(
        command=DisplaySpec(lines=(1, 1)),
        command_language="bash",
    )
    plain = _Screen()
    plain.printer.call("Bash\na\nb\nc", display=display)
    assert plain.rows() == ["  Bash", "  ⎿  a", "     ⋯ 1 lines ⋯", "     c"]
    colored = _Screen(color=True)
    colored.printer.call("Bash\necho hi", display=display)
    unhighlighted = _Screen(color=True)
    unhighlighted.printer.call("Bash\necho hi", display=ToolDisplay())
    assert colored.raw() != unhighlighted.raw()
    assert [_visible(r) for r in colored.raw()] == ["  Bash", "  ⎿  echo hi"]


def test_input_rows_wrap_at_the_mark() -> None:
    screen = _Screen(width=10)
    screen.printer.call("H\nabcdefghij", display=ToolDisplay())
    assert screen.rows() == ["  H", "  ⎿  abcde", "     fghij"]


def test_output_rows_are_dim_at_the_output_indent() -> None:
    screen = _Screen(width=10, color=True)
    screen.printer.output("one\ntwo three", display=ToolDisplay(output=DisplaySpec()))
    raw = screen.raw()
    assert [_visible(r) for r in raw] == ["     one", "     two t", "     hree"]
    assert all(_codes(r)[0] == "2" for r in raw)


def test_output_the_display_hides_prints_nothing() -> None:
    screen = _Screen()
    screen.printer.output("secret", display=ToolDisplay())
    assert screen.rows() == []


def test_a_diff_renders_with_its_counts() -> None:
    screen = _Screen()
    screen.printer.diff("@@ -1 +1 @@\n-a\n+b", path="x.py")
    assert screen.rows() == [
        "  ⎿  Added 1 lines, removed 1 lines",
        "  1 a",
        "  1 b",
    ]


def test_an_error_is_red_with_a_mark_on_its_first_row() -> None:
    screen = _Screen(color=True)
    screen.printer.error("failed\ndetail\n")
    raw = screen.raw()
    assert [_visible(r) for r in raw] == ["     ✗ failed", "       detail"]
    assert all(_codes(r)[0] == "2;31" for r in raw)


@pytest.mark.parametrize("text", ["", "\n", "  \n \n"])
def test_a_blank_error_still_says_something(text: str) -> None:
    screen = _Screen()
    screen.printer.error(text)
    assert screen.rows() == ["     ✗ <no error message>"]


@pytest.mark.parametrize(("width", "rule"), [(30, 30), (10, 20)])
def test_a_halt_is_bold_red_between_rules(width: int, rule: int) -> None:
    screen = _Screen(width=width, color=True)
    screen.printer.halt("stopped")
    raw = screen.raw()
    assert [_visible(r) for r in raw if r][:1] == ["─" * min(rule, width)]
    assert "stopped" in [_visible(r) for r in raw]
    assert any(_codes(r) and _codes(r)[0] == "1;31" for r in raw)
    assert _codes(raw[0])[0] == "31"
    assert "".join(_visible(r) for r in raw).count("─") == 2 * rule


def _status(text: str) -> types.StatusChanged:
    return types.StatusChanged(text=text)


def test_the_title_is_an_osc_on_a_terminal_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tty = _Tty()
    monkeypatch.setattr(sys, "stderr", tty)
    title = Title()
    title(_status("  fix\n the   bug "))
    assert tty.getvalue() == "\x1b]0;fix the bug\x07"
    assert title.frame() == "fix the bug"


def test_the_title_is_written_only_when_it_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tty = _Tty()
    monkeypatch.setattr(sys, "stderr", tty)
    title = Title()
    title(_status("a"))
    title(_status(" a "))
    title(types.AgentIdle())
    title(_status("b"))
    assert tty.getvalue() == "\x1b]0;a\x07\x1b]0;b\x07"


def test_a_long_title_is_cut_to_eighty(monkeypatch: pytest.MonkeyPatch) -> None:
    tty = _Tty()
    monkeypatch.setattr(sys, "stderr", tty)
    title = Title()
    title.update(_status("x" * 80))
    title.update(_status("y" * 81))
    assert tty.getvalue() == f"\x1b]0;{'x' * 80}\x07\x1b]0;{'y' * 79}…\x07"


def test_no_title_is_set_off_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = io.StringIO()
    monkeypatch.setattr(sys, "stderr", pipe)
    title = Title()
    title.update(_status("t"))
    assert pipe.getvalue() == ""
    assert title.frame() == "t"


def test_a_failing_terminal_does_not_fail_the_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stderr", _BrokenTty())
    Title().update(_status("t"))


# What the model or a tool wrote reaches the terminal: an escape in it could clear
# the screen, retitle the window, or forge a prompt. Rich drops BEL but keeps ESC.
_HOSTILE = "a\x1b[2Jb\x1b]0;pwn\x07c\rd\x7fe"
_SHOWN = "a^[[2Jb^[]0;pwn^Gc^Md^?e"


def _print_hostile(printer: ConsolePrinter, kind: str) -> None:
    shown = ToolDisplay(output=DisplaySpec(lines=None), command_language="bash")
    match kind:
        case "agent":
            printer.agent(_HOSTILE, _HOSTILE, _HOSTILE)
        case "call":
            printer.call(f"{_HOSTILE}\n{_HOSTILE}", display=shown)
        case "output":
            printer.output(_HOSTILE, display=shown)
        case "thinking":
            printer.thinking(_HOSTILE, opens=True, closes=True)
        case "diff":
            printer.diff(
                f"@@ -1 +1 @@\n-{_HOSTILE}\n+{_HOSTILE}!",
                path=f"{_HOSTILE}.py",
            )
        case _:
            getattr(printer, kind)(_HOSTILE)


@pytest.mark.parametrize(
    "kind",
    [
        "line",
        "note",
        "user",
        "agent",
        "markdown",
        "thinking",
        "call",
        "output",
        "diff",
        "error",
        "halt",
    ],
)
@pytest.mark.parametrize("color", [False, True])
def test_control_characters_are_shown_never_sent(kind: str, color: bool) -> None:
    screen = _Screen(width=200, color=color)
    _print_hostile(screen.printer, kind)
    text = "".join(_visible(row) for row in screen.raw())
    assert "\x1b" not in text
    assert "\x07" not in screen.out.getvalue()
    assert "\r" not in screen.out.getvalue()
    if kind != "markdown":
        assert _SHOWN in text


def test_a_title_sends_one_escape_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    tty = _Tty()
    monkeypatch.setattr(sys, "stderr", tty)
    Title().update(_status("x\x07\x1b[2J\x1b]0;evil\x07y"))
    assert tty.getvalue() == "\x1b]0;x^G^[[2J^[]0;evil^Gy\x07"


def _runs(row: str) -> list[tuple[str, str]]:
    """Return a row's text runs, each with the SGR codes it was printed in."""
    return [
        (str(m.group(1)), str(m.group(2)))
        for m in re.finditer(r"\x1b\[([0-9;]*)m([^\x1b]*)", row)
        if m.group(2)
    ]


def test_an_input_row_and_its_mark_are_dim() -> None:
    screen = _Screen(color=True)
    screen.printer.call("H\nls -la", display=ToolDisplay())
    mark, body = _runs(screen.raw()[1])
    assert mark == ("2", "  ⎿  ")
    assert body == ("2", "ls -la")


def test_a_highlighted_input_row_is_dim_too() -> None:
    screen = _Screen(color=True)
    screen.printer.call("H\necho hi", display=ToolDisplay(command_language="bash"))
    runs = _runs(screen.raw()[1])[1:]
    assert len(runs) > 1
    assert all("2" in codes.split(";") for codes, _ in runs)


def test_a_header_is_capped_at_twelve_rows_by_default() -> None:
    screen = _Screen(width=30)
    screen.printer = ConsolePrinter(screen.printer.console)
    screen.printer.call("a" * 1000, display=ToolDisplay())
    rows = screen.rows()
    assert len(rows) == 12
    assert rows[-1] == "  ... (25 more lines)"


def test_a_header_on_a_tiny_console_still_wraps_by_one_cell() -> None:
    screen = _Screen(width=3)
    screen.printer.call("abc", display=ToolDisplay())
    assert [r.strip() for r in screen.rows()] == ["a", "b", "c"]


def test_a_user_bar_follows_a_reply_after_a_blank_line() -> None:
    screen = _Screen(width=12)
    screen.printer.markdown("m")
    screen.printer.user("u")
    assert screen.rows() == ["", "m", "", "> u"]


def test_closed_thinking_then_a_user_bar_or_reply_leaves_one_blank_line() -> None:
    for follow in ("user", "markdown"):
        screen = _Screen(width=12)
        screen.printer.thinking("t", opens=False, closes=True)
        getattr(screen.printer, follow)("u")
        assert screen.rows()[:3] == ["  t", "", "> u" if follow == "user" else "u"]


@pytest.mark.parametrize(
    "follow",
    ["call", "thinking", "note", "line", "agent", "error", "markdown", "user"],
)
def test_every_row_after_a_user_bar_follows_one_blank_line(follow: str) -> None:
    screen = _Screen(width=12)
    printer = screen.printer
    printer.user("u")
    match follow:
        case "call":
            printer.call("Bash ls", display=ToolDisplay())
        case "thinking":
            printer.thinking("t", opens=False, closes=False)
        case "agent":
            printer.agent("a", None, "w")
        case "user":
            printer.user("v")
        case _:
            getattr(printer, follow)("x")
    rows = screen.rows()
    assert rows[:3] == ["", "> u", ""]
    assert rows[3].strip()


def test_a_user_bar_on_a_three_cell_console_still_prompts_each_row() -> None:
    screen = _Screen(width=3)
    screen.printer.user("hi")
    assert screen.rows() == ["", "> h", "  i"]


def test_a_user_bar_keeps_a_trailing_x_and_drops_trailing_newlines() -> None:
    screen = _Screen(width=12)
    screen.printer.user("hello X\n\n")
    assert screen.rows() == ["", "> hello X"]


def test_an_error_keeps_its_text_but_the_final_newlines() -> None:
    screen = _Screen(color=True)
    screen.printer.error("\nfailed X  \n")
    assert [_visible(r) for r in screen.raw()] == ["     ✗ ", "       failed X  "]
    plain = _Screen()
    plain.printer.error("failed X\n")
    assert plain.rows() == ["     ✗ failed X"]


def test_a_diff_is_highlighted_by_its_path() -> None:
    python, text = _Screen(color=True), _Screen(color=True)
    diff = "@@ -1 +1 @@\n-def a(): pass\n+def b(): pass"
    python.printer.diff(diff, path="x.py")
    text.printer.diff(diff, path="x.txt")
    assert python.raw() != text.raw()


def test_the_style_picks_the_colors() -> None:
    out = io.StringIO()
    console = Console(file=out, width=20, color_system="truecolor", force_terminal=True)
    printer = ConsolePrinter(console, style=ConsoleStyle(quiet=Style(italic=True)))
    printer.note("n")
    assert _codes(out.getvalue())[0] == "3"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
