"""Startup layout, literal metadata, and terminal fallbacks."""

from __future__ import annotations

from pathlib import Path

import io
import re

from rich.cells import cell_len
from rich.console import Console

import pytest

from sagent.tui import welcome
from sagent.tui.welcome import render_welcome


@pytest.fixture(autouse=True)
def _terminal_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")


def _render(
    *,
    width: int = 80,
    height: int = 24,
    terminal: bool = True,
    resumed: bool = False,
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
        resumed=resumed,
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


def test_resume_keeps_transcript_prominent() -> None:
    out = _render(resumed=True)
    assert "SAGENT" in out
    assert "Resuming your session." in out
    assert "█" not in out
    assert "Good news, everyone! The discovering begins." in out


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
        resumed=True,
    )
    told = console.export_text(styles=False)
    assert ("/research" in bare, "model" in bare, "Resuming" in bare) == (
        True,
        False,
        False,
    )
    assert ("/elsewhere" in told, "test-model · TestProvider" in told) == (True, True)
    assert "Resuming your session." in told


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
    rows = welcome._banner_rows(blue="blue", green="green", red="red")
    assert len(rows) == 6
    assert {row.cell_len for row in rows} == {66}
    # The icon's 11 columns, a 3-column gap, then the wordmark.
    assert {row.plain[11:14] for row in rows} == {"   "}
    assert rows[0].plain[14:].rstrip() == (
        "███████╗ █████╗  ██████╗ ███████╗███╗   ██╗████████╗"
    )


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


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
