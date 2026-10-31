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
from sagent.tools.bash import Bash
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tui.console import ConsolePrinter, ConsoleStyle, Screen, Title
from sagent.tui.types import Header


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

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


def test_a_new_printer_owes_no_gap() -> None:
    assert ConsolePrinter(Console(file=io.StringIO())).gap_owed is False


def test_it_is_a_printer_that_hides_thinking_by_default() -> None:
    printer: Printer = ConsolePrinter(Console(file=io.StringIO()))
    assert printer.show_thinking is False
    assert (
        ConsolePrinter(Console(file=io.StringIO()), show_thinking=True).show_thinking
        is True
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
        "  web       ",
    ]
    assert all("48;2;55;55;55" in _codes(r)[0] for r in raw[1:])
    assert all(_codes(r)[0].startswith("37;") for r in raw[1:])


def test_a_user_bar_stops_at_80_in_a_wide_window() -> None:
    screen = _Screen(width=200, color=True)
    screen.printer.user("hi\n")
    assert [len(_visible(r)) for r in screen.raw()] == [0, 80]


def test_a_long_user_bar_still_stops_at_80() -> None:
    screen = _Screen(width=200, color=True)
    screen.printer.user("u" * 150)
    assert [len(_visible(r)) for r in screen.raw()] == [0, 80, 80]
    assert _visible(screen.raw()[1]) == "> " + "u" * 78


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


def test_a_bar_neither_sets_nor_follows_its_turn_s_width() -> None:
    screen = _Screen(width=200, color=True)
    screen.printer.user("u" * 95)
    screen.printer.markdown("```\nz = 2\n```")
    screen.printer.markdown(f"```\n{'w' * 95}\n```")
    screen.printer.agent("w", None, "done")
    assert _fills(screen) == [80, 97, 80]


def test_a_halt_s_rules_fill_like_every_other_edge() -> None:
    screen = _Screen(width=200)
    screen.printer.halt("stopped")
    screen.printer.markdown(f"```\n{'w' * 95}\n```")
    screen.printer.halt("h" * 150)
    rules = [len(r) for r in screen.rows() if r.startswith("─")]
    assert rules == [80, 80, 100, 100]
    assert "h" * 100 in screen.rows()


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
    screen.printer.call(
        Header(key="1", label="Bash list files\nls -la\ncd /", display=ToolDisplay()),
    )
    raw = screen.raw()
    assert [_visible(r) for r in raw] == [
        "  Bash list files",
        "  ⎿ ls -la",
        "    cd /",
    ]
    assert _codes(raw[0])[0] == "2"
    assert all("2" in _codes(r)[-2].split(";") for r in raw[1:])


def test_a_call_with_no_input_is_its_header_alone() -> None:
    screen = _Screen()
    screen.printer.call(Header(key="1", label="Read a.py", display=ToolDisplay()))
    screen.printer.call(Header(key="1", label="", display=ToolDisplay()))
    assert screen.rows() == ["  Read a.py", ""]


def test_a_long_header_wraps_at_the_indent() -> None:
    screen = _Screen(width=12)
    screen.printer.call(
        Header(key="1", label="abcdefghijklmnop", display=ToolDisplay()),
    )
    assert screen.rows() == ["  abcdefghij", "  klmnop"]


def test_a_wrapped_header_hangs_under_its_argument() -> None:
    screen = _Screen(width=20)
    screen.printer.call(
        Header(key="1", label="Grep " + "y" * 30, display=ToolDisplay()),
    )
    assert screen.rows() == [
        "  Grep " + "y" * 13,
        "       " + "y" * 13,
        "       " + "y" * 4,
    ]


def test_the_header_is_capped_and_says_how_many_rows_it_hid() -> None:
    screen = _Screen(width=24, max_label_lines=3)
    screen.printer.call(Header(key="1", label="a" * 110, display=ToolDisplay()))
    assert screen.rows() == ["  " + "a" * 22, "  " + "a" * 22, "  ... (3 more lines)"]


def test_a_header_at_the_cap_is_kept_whole() -> None:
    screen = _Screen(width=12, max_label_lines=2)
    screen.printer.call(Header(key="1", label="a" * 20, display=ToolDisplay()))
    assert screen.rows() == ["  aaaaaaaaaa", "  aaaaaaaaaa"]


def test_the_input_rows_follow_the_display_and_its_language() -> None:
    display = ToolDisplay(
        command=DisplaySpec(lines=(1, 1)),
        command_language="bash",
    )
    plain = _Screen()
    plain.printer.call(Header(key="1", label="Bash\na\nb\nc", display=display))
    assert plain.rows() == ["  Bash", "  ⎿ a", "    ⋯ 1 lines ⋯", "    c"]
    colored = _Screen(color=True)
    colored.printer.call(Header(key="1", label="Bash\necho hi", display=display))
    unhighlighted = _Screen(color=True)
    unhighlighted.printer.call(
        Header(key="1", label="Bash\necho hi", display=ToolDisplay()),
    )
    assert colored.raw() != unhighlighted.raw()
    assert [_visible(r) for r in colored.raw()] == ["  Bash", "  ⎿ echo hi"]


def test_input_rows_wrap_at_the_mark() -> None:
    screen = _Screen(width=10)
    screen.printer.call(Header(key="1", label="H\nabcdefghij", display=ToolDisplay()))
    assert screen.rows() == ["  H", "  ⎿ abcdef", "    ghij"]


def test_output_rows_are_dim_at_the_output_indent() -> None:
    screen = _Screen(width=10, color=True)
    shown = ToolDisplay(output=DisplaySpec())
    screen.printer.output("one\ntwo three\nX\n\n", display=shown)
    raw = screen.raw()
    assert [_visible(r) for r in raw] == ["    one", "    two th", "    ree", "    X"]
    assert all(r.startswith("\x1b[2m    ") for r in raw)
    assert all(codes == "2" for r in raw for codes, _ in _runs(r))


@pytest.mark.parametrize("width", [10, 80, 196])
def test_the_output_width_is_what_an_output_row_wraps_at(width: int) -> None:
    screen = _Screen(width=width)
    assert screen.printer.output_width == width - 4
    screen.printer.output("x" * 300, display=ToolDisplay(output=DisplaySpec()))
    assert {len(row) for row in screen.rows()[:-1]} == {width}


def test_the_output_width_follows_the_window() -> None:
    screen = _Screen(width=80)
    screen.printer.console.width = 60
    assert screen.printer.output_width == 56


def test_output_the_display_hides_prints_nothing() -> None:
    screen = _Screen()
    screen.printer.output("secret", display=ToolDisplay())
    assert screen.rows() == []


def test_a_diff_is_its_numbered_lines_alone() -> None:
    screen = _Screen()
    screen.printer.diff("@@ -1 +1 @@\n-a\n+b", path="x.py")
    assert screen.rows() == ["    1 a", "    1 b"]


def _under_a_header(kind: str) -> list[str]:
    """Return the rows a call's header, then one row of ``kind``, print under it."""
    screen = _Screen()
    printer = screen.printer
    shown = ToolDisplay(output=DisplaySpec())
    match kind:
        case "input":
            printer.call(Header(key="1", label="Bash ls\nls\npwd", display=shown))
        case "output":
            printer.call(Header(key="1", label="Bash ls", display=shown))
            printer.output("one\ntwo", display=shown)
        case "diff":
            printer.call(Header(key="1", label="Edit a.py", display=shown))
            printer.diff("@@ -1 +1 @@\n-a\n+b", path="a.py")
        case "failed":
            printer.call(Header(key="1", label="Read a.py", display=shown))
            printer.failed("gone\nwhy")
        case _:
            printer.call(Header(key="1", label="Read a.py", display=shown))
            printer.error("gone\nwhy")
    return screen.rows()[1:]


@pytest.mark.parametrize(
    ("kind", "rows"),
    [
        ("input", ["  ⎿ ls", "    pwd"]),
        ("output", ["    one", "    two"]),
        ("diff", ["    1 a", "    1 b"]),
        ("failed", ["    ✗ gone", "      why"]),
        ("error", ["  ✗ gone", "    why"]),
    ],
)
def test_only_input_takes_the_mark_and_results_sit_at_the_output_indent(
    kind: str,
    rows: list[str],
) -> None:
    assert _under_a_header(kind) == rows


def test_a_row_after_the_first_sits_unmarked_at_the_output_indent() -> None:
    screen = _Screen()
    shown = ToolDisplay(output=DisplaySpec())
    screen.printer.call(Header(key="1", label="Bash ls\nls", display=shown))
    screen.printer.output("[undetached; #9]", display=shown)
    screen.printer.diff("@@ -1 +1 @@\n-a\n+b", path="a.py")
    assert screen.rows()[1:] == [
        "  ⎿ ls",
        "    [undetached; #9]",
        "    1 a",
        "    1 b",
    ]


_OUTPUT = ToolDisplay(output=DisplaySpec())
_READ = Header(key="1", label="Read a.py", display=_OUTPUT)


def test_a_row_right_under_its_call_names_it_once() -> None:
    screen = _Screen()
    printer = screen.printer
    printer.call(_READ)
    printer.output("[undetached; #1]", display=_OUTPUT, header=_READ)
    printer.output("said", display=_OUTPUT, header=_READ)
    printer.diff("@@ -1 +1 @@\n-a\n+b", path="a.py", header=_READ)
    printer.failed("broke", header=_READ)
    assert screen.rows() == [
        "  Read a.py",
        "    [undetached; #1]",
        "    said",
        "    1 a",
        "    1 b",
        "    ✗ broke",
    ]


@pytest.mark.parametrize(
    "kind",
    [
        "line",
        "note",
        "user",
        "agent",
        "markdown",
        "thinking",
        "other-call",
        "other-output",
        "error",
        "halt",
        "screen",
    ],
)
@pytest.mark.parametrize("row", ["output", "diff", "failed"])
def test_a_row_below_anything_else_names_its_call_again(kind: str, row: str) -> None:
    screen = _Screen()
    printer = screen.printer
    printer.call(_READ)
    other = Header(key="2", label="Read b.py", display=_OUTPUT)
    match kind:
        case "agent":
            printer.agent("w", None, "x")
        case "thinking":
            printer.thinking("x", opens=False, closes=False)
        case "other-call":
            printer.call(other)
        case "other-output":
            printer.output("x", display=_OUTPUT, header=other)
        case "screen":
            printer.screen.write("a warning\n")
        case _:
            getattr(printer, kind)("x")
    match row:
        case "output":
            printer.output("said", display=_OUTPUT, header=_READ)
        case "diff":
            printer.diff("@@ -1 +1 @@\n-a\n+b", path="a.py", header=_READ)
        case _:
            printer.failed("broke", header=_READ)
    rows = screen.rows()
    assert rows.count("  Read a.py") == 2
    assert rows[rows.index("  Read a.py", 1) + 1].startswith("    ")


@pytest.mark.parametrize(
    "kind",
    ["hidden-output", "blank-thinking", "title", "partial-line", "nothing"],
)
def test_what_prints_no_row_leaves_its_call_last(kind: str) -> None:
    screen = _Screen()
    printer = screen.printer
    printer.call(_READ)
    match kind:
        case "hidden-output":
            printer.output("x", display=ToolDisplay(), header=_READ)
        case "blank-thinking":
            printer.thinking("", opens=False, closes=True)
        case "title":
            printer.screen.write("\x1b]0;working\x07")
        case "partial-line":
            printer.screen.write("")
        case _:
            pass
    printer.output("said", display=_OUTPUT, header=_READ)
    rows = [r for r in screen.rows() if r.strip()]
    assert (rows[0], rows[-1].endswith("    said"), len(rows)) == (
        "  Read a.py",
        True,
        2,
    )


def test_a_result_that_prints_no_row_names_no_call_again() -> None:
    screen = _Screen()
    printer = screen.printer
    printer.call(_READ)
    printer.note("x")
    printer.output("hidden", display=ToolDisplay(), header=_READ)
    printer.output("  \n", display=_OUTPUT, header=_READ)
    assert screen.rows() == ["  Read a.py", "x"]


def test_a_row_that_prints_only_after_cleaning_still_names_its_call() -> None:
    screen = _Screen()
    printer = screen.printer
    printer.call(_READ)
    printer.note("x")
    printer.output("\x0b", display=_OUTPUT, header=_READ)
    assert screen.rows()[-2:] == ["  Read a.py", "    ^K"]


def test_rows_written_through_the_given_screen_name_the_call_again() -> None:
    out = io.StringIO()
    screen = Screen(out)
    printer = ConsolePrinter(Console(file=out, width=40), screen=screen)
    printer.call(_READ)
    screen.write("a warning\n")
    printer.output("said", display=_OUTPUT, header=_READ)
    assert out.getvalue().count("Read a.py") == 2


def test_a_screen_counts_the_rows_it_passes_on(tmp_path: Path) -> None:
    with (tmp_path / "out").open("w") as file:
        screen = Screen(file)
        assert screen.rows == 0
        assert screen.write("a\x1b]0;t\x07") == 7
        screen.write("b\nc\n")
        screen.flush()
        assert (screen.rows, screen.isatty(), screen.fileno()) == (
            2,
            False,
            file.fileno(),
        )
    assert (tmp_path / "out").read_text() == "a\x1b]0;t\x07b\nc\n"


def test_the_mark_is_the_printer_s_to_choose() -> None:
    out = io.StringIO()
    printer = ConsolePrinter(Console(file=out, width=40, color_system=None), mark="└")
    printer.call(Header(key="1", label="Bash ls\nls", display=ToolDisplay()))
    assert out.getvalue().splitlines()[1] == "  └ ls"


_FAILURES = [("error", "  "), ("failed", "    ")]
"""Each way to write what failed, and the indent its ``✗`` follows."""


@pytest.mark.parametrize(("kind", "indent"), _FAILURES)
def test_a_failure_is_red_with_an_x_on_its_first_row(kind: str, indent: str) -> None:
    screen = _Screen(color=True)
    getattr(screen.printer, kind)("failed\ndetail\n")
    raw = screen.raw()
    assert [_visible(r) for r in raw] == [f"{indent}✗ failed", f"{indent}  detail"]
    assert all(_codes(r)[0] == "2;31" for r in raw)


@pytest.mark.parametrize(("kind", "indent"), _FAILURES)
@pytest.mark.parametrize("text", ["", "\n", "  \n \n"])
def test_a_blank_failure_still_says_something(
    kind: str,
    indent: str,
    text: str,
) -> None:
    screen = _Screen()
    getattr(screen.printer, kind)(text)
    assert screen.rows() == [f"{indent}✗ <no error message>"]


@pytest.mark.parametrize(
    ("kind", "rows"),
    [
        ("error", ["  ✗ " + "a" * 16, "    " + "a" * 14]),
        ("failed", ["    ✗ " + "a" * 14, "      " + "a" * 14, "      aa"]),
    ],
)
def test_a_long_failure_wraps_under_its_text(kind: str, rows: list[str]) -> None:
    screen = _Screen(width=20)
    getattr(screen.printer, kind)("a" * 30)
    assert screen.rows() == rows


@pytest.mark.parametrize(("width", "rule"), [(30, 30), (10, 10), (200, 80)])
def test_a_halt_is_bold_red_between_rules(width: int, rule: int) -> None:
    screen = _Screen(width=width, color=True)
    screen.printer.halt("stopped")
    raw = screen.raw()
    assert [_visible(r) for r in raw if r][:1] == ["─" * rule]
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
            printer.call(
                Header(key="1", label=f"{_HOSTILE}\n{_HOSTILE}", display=shown),
            )
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
    screen.printer.call(Header(key="1", label="H\nls -la", display=ToolDisplay()))
    mark, body = _runs(screen.raw()[1])
    assert mark == ("2", "  ⎿ ")
    assert body == ("2", "ls -la")


def test_a_highlighted_input_row_is_dim_too() -> None:
    screen = _Screen(color=True)
    screen.printer.call(
        Header(
            key="1",
            label="H\necho hi",
            display=ToolDisplay(command_language="bash"),
        ),
    )
    runs = _runs(screen.raw()[1])[1:]
    assert len(runs) > 1
    assert all("2" in codes.split(";") for codes, _ in runs)


def test_a_header_is_capped_at_twelve_rows_by_default() -> None:
    screen = _Screen(width=30)
    screen.printer = ConsolePrinter(screen.printer.console)
    screen.printer.call(Header(key="1", label="a" * 1000, display=ToolDisplay()))
    rows = screen.rows()
    assert len(rows) == 12
    assert rows[-1] == "  ... (25 more lines)"


def test_a_header_on_a_tiny_console_still_wraps_by_one_cell() -> None:
    screen = _Screen(width=3)
    screen.printer.call(Header(key="1", label="abc", display=ToolDisplay()))
    assert screen.rows() == ["  a", "  b", "  c"]


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
            printer.call(Header(key="1", label="Bash ls", display=ToolDisplay()))
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


@pytest.mark.parametrize(("kind", "indent"), _FAILURES)
def test_a_failure_keeps_its_text_but_the_final_newlines(
    kind: str,
    indent: str,
) -> None:
    screen = _Screen(color=True)
    getattr(screen.printer, kind)("\nfailed X  \n")
    assert [_visible(r) for r in screen.raw()] == [
        f"{indent}✗ ",
        f"{indent}  failed X  ",
    ]
    plain = _Screen()
    getattr(plain.printer, kind)("failed X\n")
    assert plain.rows() == [f"{indent}✗ failed X"]


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


_ALL = ToolDisplay(command=DisplaySpec(lines=None), output=DisplaySpec(lines=None))

_INPUTS = {
    "long": "w" * 90,
    "tabs": "a\tb\t" * 15,
    "cjk": "错误" * 30,
    "emoji": "🙂👍🏽" * 20,
    "spaced": "word " * 25,
    "multiline": "line one\n" + "z" * 70 + "\nline three",
    "trailing-newline": "only\n",
    "empty": "",
}


def _diff_of(text: str) -> str:
    return "@@ -1 +1 @@\n" + "\n".join(f"+{ln}" for ln in text.split("\n"))


# Each kind: how to print a text, then the patterns its first row and every later
# row start with. A row matching neither has fallen out of its column.
_KINDS: dict[str, tuple[Callable[[ConsolePrinter, str], None], str, str]] = {
    "note": (lambda p, s: p.note(s, lead="Agent_0: "), "Agent_0: ", " " * 9),
    "user": (lambda p, s: p.user(s), "> ", "  "),
    "agent": (lambda p, s: p.agent("w", "Agent", s), "w→Agent: ", "  "),
    "thinking": (lambda p, s: p.thinking(s, opens=False, closes=True), "  ", "  "),
    "header": (
        lambda p, s: p.call(Header(key="1", label="Bash " + s, display=_ALL)),
        "  ",
        "  ",
    ),
    "input": (
        lambda p, s: p.call(Header(key="1", label="Bash\n" + s, display=_ALL)),
        "  (Bash|⎿ )",
        "    ",
    ),
    "output": (lambda p, s: p.output(s, display=_ALL), "    ", "    "),
    "error": (lambda p, s: p.error(s), "  ✗ ", "    "),
    "failed": (lambda p, s: p.failed(s), "    ✗ ", "      "),
    "diff": (lambda p, s: p.diff(_diff_of(s), path="a.txt"), r"\s*\d+ ", "    "),
}


@pytest.mark.parametrize("given", list(_INPUTS))
@pytest.mark.parametrize("kind", list(_KINDS))
def test_every_row_stays_in_its_column_inside_the_pane(kind: str, given: str) -> None:
    """A wrapped row continues under its own prefix, never back at column 0.

    Each method wrapped its own rows: ``chop_cells`` counts a tab as no cells,
    so a tabbed row overran the pane and the console wrapped it to column 0;
    reasoning was not wrapped at all.
    """
    emit, first, rest = _KINDS[kind]
    screen = _Screen()
    emit(screen.printer, _INPUTS[given])
    rows = [r for r in screen.raw() if r.strip()]
    for row in rows:
        assert len(_visible(row)) <= 40, f"{row!r} overruns the pane"
        assert "\t" not in row, f"{row!r} sends a tab"
        assert re.match(first, row) or row.startswith(rest), f"{row!r} left its column"
    if _INPUTS[given].strip():
        assert _visible(screen.raw()[-1]).strip(), "a trailing blank row"


_SENTENCE = "alpha beta gamma delta epsilon zeta eta theta iota"

# What a person or a model wrote, and the lead its first row starts with.
_PROSE: dict[str, tuple[Callable[[ConsolePrinter, str], None], str]] = {
    "child-reply": (lambda p, s: p.note(s, lead="Agent_0: ", prose=True), "Agent_0: "),
    "user": (lambda p, s: p.user(s), "> "),
    "agent": (lambda p, s: p.agent("w", None, s), "w→me: "),
    "thinking": (lambda p, s: p.thinking(s, opens=False, closes=True), ""),
    "header": (
        lambda p, s: p.call(Header(key="1", label="Bash " + s, display=_ALL)),
        "Bash ",
    ),
}


@pytest.mark.parametrize("kind", list(_PROSE))
def test_prose_wraps_between_words(kind: str) -> None:
    """A sentence breaks at a space; cutting mid-word made prose hard to read."""
    emit, lead = _PROSE[kind]
    screen = _Screen(width=33)
    emit(screen.printer, _SENTENCE)
    rows = [r for r in screen.rows() if r.strip()]
    assert len(rows) > 1
    assert rows[0].lstrip().startswith(lead)
    words = " ".join(rows).replace(lead, "", 1).split()
    assert words == _SENTENCE.split(), rows


# What a tool or the harness wrote, the text's first and later leads: a space in it
# may be content, such as a column's alignment.
_VERBATIM: dict[str, tuple[Callable[[ConsolePrinter, str], None], int, int]] = {
    "failed": (lambda p, s: p.failed(s), 6, 6),
    "error": (lambda p, s: p.error(s), 4, 4),
    "note": (lambda p, s: p.note(s), 0, 0),
}


@pytest.mark.parametrize("kind", list(_VERBATIM))
def test_a_tool_s_or_the_harness_s_text_keeps_its_spacing(kind: str) -> None:
    """Breaking between words dropped the spaces at each break, losing columns."""
    emit, first, rest = _VERBATIM[kind]
    text = "col_a      col_b      col_c      col_d      col_e      col_f"
    screen = _Screen(width=40)
    emit(screen.printer, text)
    rows = screen.raw()
    assert rows[0][first:] == text[: 40 - first], rows
    assert rows[1][rest:].rstrip() == text[40 - first :][: 40 - rest].rstrip(), rows


def test_a_command_s_tab_is_as_wide_as_an_output_s() -> None:
    """``Bash.summary`` made a tab four spaces; the printer makes it a tab stop."""
    screen = _Screen()
    screen.printer.call(
        Header(key="1", label=Bash().summary({"command": "a\tb"}), display=_ALL),
    )
    screen.printer.output("a\tb", display=_ALL)
    command, output = screen.rows()[1:]
    assert command.removeprefix("  ⎿ ") == output.removeprefix("    ") == "a       b"


def test_a_row_wrapped_inside_a_tab_starts_at_the_next_text() -> None:
    """The rest of a tab's stop is no content to lead the next row with."""
    screen = _Screen(width=40)
    screen.printer.output("x" * 34 + "\tyz", display=_ALL)
    assert screen.rows() == ["    " + "x" * 34, "    yz"]


def test_a_capped_header_s_count_fits_a_narrow_pane() -> None:
    """The count row was printed whole, past the pane's edge."""
    screen = _Screen(width=12, max_label_lines=3)
    screen.printer.call(Header(key="1", label="a" * 200, display=ToolDisplay()))
    rows = screen.rows()
    assert len(rows) == 3
    assert all(len(r) <= 12 for r in rows), rows


def test_a_pane_narrower_than_the_indent_still_caps_the_header() -> None:
    """With no cell after the lead, the console's own wrap skipped the cap."""
    screen = _Screen(width=2, max_label_lines=3)
    screen.printer.call(Header(key="1", label="a" * 200, display=ToolDisplay()))
    rows = screen.rows()
    assert len(rows) == 3, rows
    assert all(len(r) <= 2 for r in rows), rows


@pytest.mark.parametrize("kind", ["output", "input"])
def test_code_still_wraps_at_the_edge(kind: str) -> None:
    """Output and commands are not prose: a space there is content, not a break."""
    screen = _Screen(width=20)
    text = "abc defghijklmnopqrstuvwxyz"
    if kind == "output":
        screen.printer.output(text, display=_ALL)
    else:
        screen.printer.call(Header(key="1", label="Bash\n" + text, display=_ALL))
    assert any(row.endswith(" abc defghijklmno") for row in screen.rows())


def test_ellipsis_cuts_a_wide_row_and_marks_the_cut_but_never_the_count() -> None:
    screen = _Screen(width=10)
    spec = DisplaySpec(lines=(1, 1), overflow="ellipsis")
    screen.printer.output("abcdefghij\n2\n3\n4", display=ToolDisplay(output=spec))
    assert screen.rows() == ["    abcde…", "    ⋯ 2 li", "    nes ⋯", "    4"]


def test_a_spec_width_narrows_the_row_and_tabs_take_their_cells() -> None:
    screen = _Screen(width=40)
    spec = DisplaySpec(lines=None, width=4)
    screen.printer.output("abcdef\na\tb\n\nc", display=ToolDisplay(output=spec))
    assert screen.rows() == ["    abcd", "    ef", "    a", "    b", "", "    c"]


def test_a_wrapped_command_keeps_its_strings_colored_past_the_cut() -> None:
    """Each row was highlighted alone, so a row inside a string lost its quote."""
    screen = _Screen(color=True)
    body = "\n".join(f"line{i}" for i in range(9)) + "\n" + "y" * 60 + '"'
    display = ToolDisplay(command=DisplaySpec(lines=(1, 1)), command_language="bash")
    screen.printer.call(Header(key="1", label=f'Bash\necho "{body}', display=display))
    tail = [r for r in screen.raw() if "y" in _visible(r)]
    assert len(tail) == 2
    head = [r for r in screen.raw() if "echo" in _visible(r)]
    (string,) = {codes for codes, text in _runs(head[0]) if "line0" in text}
    for row in tail:
        assert all(codes == string for codes, text in _runs(row) if "y" in text), row


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
