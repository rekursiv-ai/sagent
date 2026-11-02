"""Startup layout, literal metadata, and terminal fallbacks."""

from __future__ import annotations

from datetime import date
from itertools import groupby
from pathlib import Path
from typing import IO, TYPE_CHECKING, Literal, override

import io
import re
import time

from rich.cells import cell_len
from rich.console import Console
from rich.text import Text

import pytest

from sagent.tui import welcome
from sagent.tui.holiday import Icon
from sagent.tui.welcome import render_welcome


if TYPE_CHECKING:
    from random import Random


@pytest.fixture(autouse=True)
def _terminal_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")


def _render(
    *,
    width: int = 80,
    height: int = 24,
    terminal: bool = True,
    folder: Path = Path("/research"),
) -> str:
    stream = io.StringIO()
    console = Console(
        file=stream,
        force_terminal=terminal,
        no_color=True,
        width=width,
        height=height,
    )
    render_welcome(
        console,
        model="test-model",
        provider="TestProvider",
        folder=folder,
    )
    # Rich still emits bold/dim when no_color is set; export recorded text in
    # the dedicated NO_COLOR test below instead of assuming ANSI-free output.
    return stream.getvalue()


def test_fresh_terminal_has_banner_actual_metadata_and_invitation() -> None:
    out = _render()
    assert "███████" in out
    assert "test-model" in out
    assert "TestProvider" in out
    assert "/research" in out
    assert "Good news, everyone! The discovering begins." in out
    assert "/help" in out
    assert "/quit" in out
    assert "/tasks" in out


@pytest.mark.parametrize(("width", "height"), [(40, 24), (80, 18)])
def test_small_terminal_has_compact_heading(width: int, height: int) -> None:
    out = _render(width=width, height=height)
    assert "SAGENT" in out
    assert "█" not in out
    assert "test-model" in out


def test_welcome_without_configuration_has_identity_and_help() -> None:
    stream = io.StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        width=80,
        height=24,
        record=True,
    )
    render_welcome(console, folder=Path("/research"))
    text = console.export_text(styles=False)
    assert "███████" in text
    assert "sagent --help" not in text
    assert "/help" in text
    assert "/research" in text
    assert "model" not in text
    assert "provider" not in text
    assert text.count("╭") == 1


def test_redirected_unconfigured_welcome_is_silent() -> None:
    stream = io.StringIO()
    render_welcome(Console(file=stream, force_terminal=False), folder=Path("/research"))
    assert stream.getvalue() == ""


def test_print_welcome_writes_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    welcome.print_welcome(
        model="test-model",
        provider="TestProvider",
        folder=Path("/research"),
    )
    assert capsys.readouterr() == ("", "[TestProvider] test-model\n")


def test_print_welcome_passes_every_choice_on(monkeypatch: pytest.MonkeyPatch) -> None:
    console = Console(file=io.StringIO(), force_terminal=True, record=True, width=80)

    def to_stderr(*, stderr: bool) -> Console:
        assert stderr
        return console

    monkeypatch.setattr(welcome, "Console", to_stderr)
    welcome.print_welcome(folder=Path("/research"))
    bare = console.export_text(styles=False)
    welcome.print_welcome(model="solo", folder=Path("/research"))
    assert re.search(r"solo · +│", console.export_text(styles=False))
    welcome.print_welcome(
        model="test-model",
        provider="TestProvider",
        folder=Path("/elsewhere"),
    )
    told = console.export_text(styles=False)
    assert ("/research" in bare, "model" in bare) == (True, False)
    assert ("/elsewhere" in told, "test-model · TestProvider" in told) == (True, True)


def test_redirected_output_retains_only_plain_model_line() -> None:
    assert _render(terminal=False) == "[TestProvider] test-model\n"


def test_no_color_keeps_readable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, record=True)
    render_welcome(
        console,
        model="test-model",
        provider="TestProvider",
        folder=Path("/research"),
    )
    assert console.no_color
    assert "\x1b[38;" not in stream.getvalue()
    plain = console.export_text(styles=False)
    assert "test-model" in plain
    assert "\x1b" not in plain


def test_dumb_terminal_uses_text_heading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERM", "dumb")
    out = _render()
    assert "SAGENT" in out
    assert "█" not in out


def test_ascii_terminal_does_not_receive_unicode_blocks() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii")
    render_welcome(
        Console(file=stream, force_terminal=True, width=80),
        model="test-model",
        provider="TestProvider",
        folder=Path("/research"),
    )
    stream.flush()
    out = raw.getvalue().decode("ascii")
    assert "SAGENT" in out
    assert "█" not in out
    _ = out.encode("ascii")


def test_home_abbreviation_respects_path_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", "/people/alex")
    assert "~/research" in _render(folder=Path("/people/alex/research"))
    assert "/people/alexander/research" in _render(
        folder=Path("/people/alexander/research"),
    )


def test_literal_metadata_and_long_paths_fit_terminal() -> None:
    stream = io.StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        record=True,
        width=40,
    )
    folder = Path("/research/[bold]/a-very-long-project-directory/experiments")
    render_welcome(
        console,
        model="[red]literal-model[/red]",
        provider="TestProvider",
        folder=folder,
    )
    plain = console.export_text(styles=False)
    assert "[red]literal-model[/red]" in plain
    assert "…/" in plain
    assert "experiments" in plain
    assert all(len(line) <= 40 for line in plain.splitlines())


def test_long_directory_name_and_wide_characters_fit_metadata() -> None:
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, record=True, width=40)
    render_welcome(
        console,
        model="test-model",
        provider="TestProvider",
        folder=Path("/research/" + "研究" * 20 + "-project"),
    )
    plain = console.export_text(styles=False)
    assert "…" in plain
    assert "-project" in plain
    assert all(cell_len(line) <= 40 for line in plain.splitlines())


def test_literal_metadata_is_not_parsed_as_markup() -> None:
    out = _render(folder=Path("/research/[bold]"))
    assert "[bold]" in out


@pytest.mark.parametrize("width", [16, 40, 56, 80, 120])
def test_terminal_width_border_contains_identity_commands_with_greeting_outside(
    width: int,
) -> None:
    stream = io.StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        record=True,
        width=width,
        height=24,
    )
    render_welcome(
        console,
        model="test-model",
        provider="TestProvider",
        folder=Path("/research"),
    )
    plain = console.export_text(styles=False)
    lines = plain.splitlines()
    top = next(i for i, line in enumerate(lines) if line.startswith("╭"))
    bottom = next(i for i, line in enumerate(lines) if line.startswith("╰"))
    frame = lines[top : bottom + 1]
    assert all(cell_len(line) == min(width, 80) for line in frame)
    assert all(line.startswith("│ ") and line.endswith(" │") for line in frame[1:-1])
    assert "project" in "\n".join(frame)
    assert "/help" in "\n".join(frame)
    assert "everyone!" not in "\n".join(frame)
    assert "everyone!" in "\n".join(lines[bottom + 1 :])
    assert all(cell_len(line) <= width for line in lines)


def test_dumb_terminal_has_no_border(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERM", "dumb")
    assert "╭" not in _render()


def test_ascii_terminal_names_the_tree_key_in_words() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii")
    console = Console(file=stream, force_terminal=True, width=80, record=True)
    render_welcome(console, folder=Path("/research"))
    stream.flush()
    _ = raw.getvalue().decode("ascii")
    assert "Left agents" in console.export_text(styles=False)


def test_wide_terminal_keeps_the_banner_at_80_columns() -> None:
    console = Console(
        file=io.StringIO(),
        force_terminal=True,
        record=True,
        width=200,
        height=40,
    )
    render_welcome(console, folder=Path("/research"))
    lines = console.export_text(styles=False).splitlines()
    assert "███████╗" in "\n".join(lines)
    assert max(cell_len(line.rstrip()) for line in lines) == 80


def test_ascii_border_preserves_encoding() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii")
    render_welcome(
        Console(file=stream, force_terminal=True, width=80),
        model="test-model",
        provider="TestProvider",
        folder=Path("/research"),
    )
    stream.flush()
    plain = raw.getvalue().decode("ascii")
    assert "+---" in plain
    assert "project" in plain


def test_icon_is_left_of_wordmark_without_extra_brand_text() -> None:
    stream = io.StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        record=True,
        width=80,
        height=24,
    )
    render_welcome(
        console,
        model="test-model",
        provider="TestProvider",
        folder=Path("/research"),
    )
    plain = console.export_text(styles=False)
    assert "rekursiv.ai" not in plain
    first = next(line for line in plain.splitlines() if "███████╗" in line)
    assert first.index("╔") < first.index("███████╗")
    assert "▼" in plain
    assert " ██████   ║" in plain


@pytest.mark.parametrize("width", [56, 68])
def test_icon_and_wordmark_use_compact_layout_when_the_pair_would_wrap(
    width: int,
) -> None:
    out = _render(width=width)
    assert "SAGENT" in out
    assert "█" not in out
    assert "rekursiv.ai" not in out


def test_identity_keeps_a_rectangular_cell_grid() -> None:
    rows = welcome._banner_rows(
        icon=welcome._logo_rows(blue="blue", green="green", red="red"),
        blue="blue",
    )
    assert len(rows) == 6
    assert {row.cell_len for row in rows} == {66}
    # The icon's 11 columns, a 3-column gap, then the wordmark.
    assert {row.plain[11:14] for row in rows} == {"   "}
    assert rows[0].plain[14:].rstrip() == (
        "███████╗ █████╗  ██████╗ ███████╗███╗   ██╗████████╗"
    )


_PUMPKIN = Icon(
    name="test pumpkin",
    palette={"O": "#e8761c"},
    pixels=("OOOOOOOOOOO",) * 12,
)


def _holiday_console(*, color_system: Literal["truecolor", "standard"]) -> Console:
    return Console(
        file=io.StringIO(),
        force_terminal=True,
        color_system=color_system,
        record=True,
        width=80,
        height=24,
    )


def test_holiday_icon_replaces_the_logo_beside_the_wordmark() -> None:
    console = _holiday_console(color_system="truecolor")
    render_welcome(console, folder=Path("/research"), icon=_PUMPKIN)
    plain = console.export_text(styles=False, clear=False)
    first = next(line for line in plain.splitlines() if "███████╗" in line)
    assert first.startswith("│  " + "█" * 11 + "   ███████╗")
    assert "▼" not in plain
    assert "#e8761c" in console.export_html(inline_styles=True)


def test_few_color_terminal_keeps_the_logo_on_a_holiday() -> None:
    console = _holiday_console(color_system="standard")
    render_welcome(console, folder=Path("/research"), icon=_PUMPKIN)
    assert "▼" in console.export_text(styles=False)


def test_print_welcome_shows_todays_holiday(monkeypatch: pytest.MonkeyPatch) -> None:
    console = _holiday_console(color_system="truecolor")
    asked: list[date] = []

    def to_stderr(*, stderr: bool) -> Console:
        assert stderr
        return console

    def pumpkin(today: date, *, rng: Random) -> Icon:
        del rng
        asked.append(today)
        return _PUMPKIN

    def local(seconds: float | None = None) -> time.struct_time:
        del seconds
        # Late on Halloween here is already November in UTC.
        return time.struct_time((2026, 10, 31, 23, 0, 0, 5, 304, 0))

    monkeypatch.setattr(welcome, "Console", to_stderr)
    monkeypatch.setattr(welcome, "holiday_icon", pumpkin)
    monkeypatch.setattr(time, "localtime", local)
    welcome.print_welcome(folder=Path("/research"))
    assert "▼" not in console.export_text(styles=False)
    assert asked == [date(2026, 10, 31)]


def test_frame_has_equal_top_and_bottom_breathing_room() -> None:
    stream = io.StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        record=True,
        width=80,
        height=24,
    )
    render_welcome(
        console,
        model="test-model",
        provider="TestProvider",
        folder=Path("/research"),
    )
    lines = console.export_text(styles=False).splitlines()
    top = next(i for i, line in enumerate(lines) if line.startswith("╭"))
    bottom = next(i for i, line in enumerate(lines) if line.startswith("╰"))
    assert lines[top + 1] == lines[bottom - 1] == "│" + " " * 78 + "│"


@pytest.mark.parametrize("width", [16, 24, 40, 80])
def test_command_descriptions_stay_with_their_command(width: int) -> None:
    stream = io.StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        record=True,
        width=width,
        height=24,
    )
    render_welcome(
        console,
        model="test-model",
        provider="TestProvider",
        folder=Path("/research"),
    )
    plain = console.export_text(styles=False)
    lines = plain.splitlines()
    for command, description in [
        ("/help", "commands"),
        ("←", "agents"),
        ("/tasks", "tool calls"),
        ("/quit", "exit"),
    ]:
        matching = [line for line in lines if command in line]
        assert len(matching) == 1
        if width >= 24:
            assert command + " " + description in matching[0]
    assert all(cell_len(line) <= width for line in lines)


def test_wrapped_greeting_is_flush_with_the_frame_edge() -> None:
    stream = io.StringIO()
    console = Console(
        file=stream,
        force_terminal=True,
        record=True,
        width=40,
        height=24,
    )
    render_welcome(
        console,
        model="test-model",
        provider="TestProvider",
        folder=Path("/research"),
    )
    lines = console.export_text(styles=False).splitlines()
    greeting_start = next(
        i for i, line in enumerate(lines) if "Good news, everyone!" in line
    )
    assert lines[greeting_start].startswith("Good news")
    assert lines[greeting_start + 1] == "begins."


def _runs(text: Text) -> list[tuple[str, str]]:
    """Return ``text`` as runs of cells that share one style, its own first."""
    styles = [str(text.style)] * len(text.plain)
    for span in text.spans:
        styles[span.start : span.end] = [str(span.style)] * (span.end - span.start)
    return [
        ("".join(c for c, _ in run), style)
        for style, run in groupby(zip(text.plain, styles, strict=True), lambda p: p[1])
    ]


class _Printed(Console):
    """A terminal that keeps each ``Text`` printed instead of drawing it."""

    def __init__(
        self,
        *,
        width: int = 80,
        height: int = 24,
        color_system: Literal["256", "truecolor"] = "truecolor",
        file: IO[str] | None = None,
    ) -> None:
        super().__init__(
            file=file or io.StringIO(),
            force_terminal=True,
            width=width,
            height=height,
            color_system=color_system,
        )
        self.texts: list[Text] = []

    @override
    def print(self, *objects: object, **kwargs: object) -> None:
        del kwargs
        self.texts.extend(o for o in objects if isinstance(o, Text))


def _rendered(
    console: _Printed,
    *,
    model: str = "test-model",
    provider: str = "TestProvider",
    folder: Path = Path("/research"),
) -> list[str]:
    render_welcome(console, model=model, provider=provider, folder=folder)
    return [t.plain for t in console.texts]


def test_the_logo_colors_only_its_three_bars() -> None:
    rows = welcome._logo_rows(blue="b", green="g", red="r")
    assert [_runs(row) for row in rows] == [
        [("   ╔══════╗", "not bold")],
        [("   ▼      ║", "not bold")],
        [(" ", "not bold"), ("██████", "bold b"), ("   ║", "not bold")],
        [(" ", "not bold"), ("█████", "bold g"), ("    ║", "not bold")],
        [(" ", "not bold"), ("███████", "bold r"), ("  ║", "not bold")],
        [("   ╚══════╝", "not bold")],
    ]


def test_the_wordmark_dims_its_box_glyphs_beside_an_unstyled_gap() -> None:
    rows = welcome._banner_rows(
        icon=welcome._logo_rows(blue="b", green="g", red="r"),
        blue="b",
    )
    assert _runs(rows[0])[1:6] == [
        ("   ", ""),
        ("███████", "b"),
        ("╗", "dim b"),
        (" █████", "b"),
        ("╗", "dim b"),
    ]
    assert _runs(rows[2])[-1] == ("   ", "")


def test_an_icon_shorter_than_the_wordmark_is_refused() -> None:
    with pytest.raises(ValueError, match="zip"):
        welcome._banner_rows(icon=[Text("x")], blue="b")


@pytest.mark.parametrize(
    ("width", "rows"),
    [
        (80, ["/help commands   ← agents   /tasks tool calls   /quit exit"]),
        (25, ["/help commands   ← agents", "/tasks tool calls", "/quit exit"]),
        (17, ["/help commands", "← agents", "/tasks tool calls", "/quit exit"]),
        (16, ["/help   ←", "/tasks   /quit"]),
    ],
)
def test_commands_fill_each_row_and_drop_descriptions_that_cannot_fit(
    width: int,
    rows: list[str],
) -> None:
    assert [r.plain for r in welcome._command_rows(width, unicode_ok=True)] == rows


def test_a_command_is_bold_and_its_description_dim() -> None:
    (row,) = welcome._command_rows(80, unicode_ok=True)
    assert _runs(row)[:3] == [
        ("/help", "bold"),
        (" commands", "dim not bold"),
        ("   ", ""),
    ]


@pytest.mark.parametrize(
    ("unicode_ok", "framed", "lines"),
    [
        (True, True, [("│  ", "dim"), ("abcdef", ""), ("  │", "dim")]),
        (False, True, [("|  ", "dim"), ("abcdef", ""), ("  |", "dim")]),
    ],
)
def test_a_framed_row_folds_inside_dim_sides(
    unicode_ok: bool,
    framed: bool,
    lines: list[tuple[str, str]],
) -> None:
    console = _Printed()
    welcome._print_framed(
        console,
        [Text("abcdefghi")],
        width=12,
        framed=framed,
        unicode_ok=unicode_ok,
        padding=2,
    )
    side = lines[0][0][0]
    assert [_runs(t) for t in console.texts] == [
        lines,
        [(f"{side}  ", "dim"), ("ghi", ""), (f"     {side}", "dim")],
    ]


def test_an_unframed_row_is_only_indented() -> None:
    console = _Printed()
    welcome._print_framed(
        console,
        [Text("abc")],
        width=12,
        framed=False,
        unicode_ok=True,
        padding=2,
    )
    assert [t.plain for t in console.texts] == ["  abc"]


@pytest.mark.parametrize(("width", "height"), [(72, 22), (71, 22), (72, 21)])
def test_the_banner_shows_only_where_it_fits(width: int, height: int) -> None:
    plain = "\n".join(_rendered(_Printed(width=width, height=height)))
    assert ("███████╗" in plain) == (width >= 72 and height >= 22)


def test_the_frame_and_the_compact_heading_are_styled() -> None:
    console = _Printed(width=40)
    render_welcome(console, folder=Path("/research"), blue="b")
    top, *_, bottom = (t for t in console.texts if t.plain[:1] in "╭╰")
    assert _runs(top) == [("╭" + "─" * 38 + "╮", "dim")]
    assert _runs(bottom) == [("╰" + "─" * 38 + "╯", "dim")]
    (heading,) = (t for t in console.texts if "SAGENT" in t.plain)
    assert ("SAGENT", "bold b") in _runs(heading)


def test_an_ascii_frame_is_drawn_in_ascii() -> None:
    raw = io.BytesIO()
    console = _Printed(file=io.TextIOWrapper(raw, encoding="ascii"))
    lines = _rendered(console)
    assert lines[0] == "+" + "-" * 78 + "+"
    assert lines[-2] == "+" + "-" * 78 + "+"
    assert any("test-model / TestProvider" in line for line in lines)


def test_the_default_colors_paint_the_logos_bars() -> None:
    console = _Printed()
    render_welcome(console, folder=Path("/research"))
    styles = {s for t in console.texts for _, s in _runs(t)}
    assert {"bold #4a90c4", "bold #93c47d", "bold #e06666", "#4a90c4"} <= styles


@pytest.mark.parametrize("color_system", ["256", "truecolor"])
def test_a_many_color_terminal_shows_the_holiday_icon(
    color_system: Literal["256", "truecolor"],
) -> None:
    console = _Printed(color_system=color_system)
    assert "▼" not in "\n".join(
        _rendered_icon(console),
    )


def _rendered_icon(console: _Printed) -> list[str]:
    render_welcome(console, folder=Path("/research"), icon=_PUMPKIN)
    return [t.plain for t in console.texts]


@pytest.mark.parametrize(
    ("provider", "shown"),
    [
        ("OpenAISubscription", "OpenAI Subscription"),
        ("AnthropicCLI", "Anthropic CLI"),
        ("LlamaCpp", "Llama.cpp"),
        ("Other", "Other"),
    ],
)
def test_a_provider_is_shown_by_its_spoken_name(provider: str, shown: str) -> None:
    lines = _rendered(_Printed(), model="m", provider=provider)
    assert any(f"model     m · {shown} " in line for line in lines)


def test_a_metadata_row_dims_its_label_only() -> None:
    console = _Printed()
    render_welcome(console, model="m", provider="P", folder=Path("/research"))
    (row,) = (t for t in console.texts if "model" in t.plain)
    assert _runs(row)[:2] == [("│  model     ", "dim"), ("m · P", "not dim")]


@pytest.mark.parametrize(("length", "rows"), [(60, 1), (61, 2)])
def test_model_and_provider_share_a_row_while_they_fit(length: int, rows: int) -> None:
    lines = _rendered(_Printed(), model="m" * length, provider="P")
    labels = [line[3:13].strip() for line in lines]
    assert labels.count("model") + labels.count("provider") == rows


def test_a_model_with_no_provider_keeps_its_separator() -> None:
    console = _Printed()
    render_welcome(console, model="m", folder=Path("/research"))
    (row,) = (t for t in console.texts if "model" in t.plain)
    assert ("m · ", "not dim") in _runs(row)


def _screen_rows(width: int) -> list[str]:
    """Return the rows a ``width``-cell terminal shows of the welcome."""
    out = io.StringIO()
    console = Console(
        file=out,
        width=width,
        height=24,
        color_system=None,
        force_terminal=True,
    )
    render_welcome(console, model="m", provider="P", folder=Path("/ab"))
    return out.getvalue().splitlines()


def test_a_value_with_no_room_beside_its_label_keeps_one_cell() -> None:
    lines = _screen_rows(12)
    assert lines[lines.index("  project   ") + 1] == "…"


def test_commands_with_no_room_keep_one_cell_each() -> None:
    lines = _screen_rows(3)
    start = lines.index("  /")
    rows = lines[start : start + 10]
    # Each command alone, folded at the 3-cell edge under its indent.
    assert "".join(rows) == "  /help  ←  /tasks  /quit"
    assert [len(r) for r in rows] == [3, 3, 1, 3, 3, 3, 2, 3, 3, 1]


def test_an_unframed_terminal_gives_values_its_width_less_the_label() -> None:
    lines = _rendered(_Printed(width=15), folder=Path("/ab"))
    assert "  project   /ab" in lines


@pytest.mark.parametrize(
    ("folder", "home", "width", "unicode_ok", "label"),
    [
        ("/abc", "/h", 4, True, "/abc"),
        ("/a/b/c", "/h", 5, True, "…/b/c"),
        ("/a/b/c", "/h", 5, False, ".../c"),
        ("/h/aa/bb/cc", "/h", 9, True, "~/…/bb/cc"),
        ("/abcdef", "/h", 1, True, "…"),
        ("/abcdefg", "/h", 3, True, "…fg"),
    ],
)
def test_a_long_folder_keeps_its_end(
    folder: str,
    home: str,
    width: int,
    unicode_ok: bool,
    label: str,
) -> None:
    assert (
        welcome._folder_label(
            Path(folder),
            home=Path(home),
            width=width,
            unicode_ok=unicode_ok,
        )
        == label
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
