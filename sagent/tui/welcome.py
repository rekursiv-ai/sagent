"""Local startup identity and orientation for the interactive CLI."""

from __future__ import annotations

from datetime import date
from functools import partial
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Final

import sys
import time

from sagent.tui.console import terminal_console
from sagent.tui.holiday import icon_rows, todays_holiday
from sagent.tui.theme import load_theme
from sagent.tui.types import fill


_BRAND_BLUE: Final = "#4a90c4"

if TYPE_CHECKING:
    from rich.console import Console
    from rich.text import Text

    from sagent.tui.holiday import Icon
    from sagent.tui.theme import Theme
else:
    from wrapt import lazy_import

    Text = lazy_import("rich.text", "Text")


def print_welcome(
    *,
    model: str = "",
    provider: str = "",
    folder: Path,
    theme: Theme | None = None,
) -> None:
    """Print identity and help before credentials or prompt-toolkit are needed.

    Args:
      model: The model line; empty before one is configured.
      provider: The provider serving it.
      folder: The project directory.
      theme: Frame, metadata, and help styles; artwork keeps its own palette.

    """
    render_welcome(
        terminal_console(sys.stderr),
        model=model,
        provider=provider,
        folder=folder,
        theme=theme,
        holiday=todays_holiday(
            date(*time.localtime()[:3]),
            rng=Random(),  # noqa: S311 -- Picks a picture, not a secret.
        ),
    )


# Text cells only: no image protocol or extra font required. Holiday
# backgrounds pack two artwork pixels into one cell.
def render_welcome(
    console: Console,
    *,
    model: str = "",
    provider: str = "",
    folder: Path,
    holiday: tuple[str, Icon] | None = None,
    theme: Theme | None = None,
    padding: int = 2,
) -> None:
    """Render a width-aware welcome without performing any model request.

    Redirected stderr retains the old plain model line. Small, dumb, and
    non-Unicode terminals use a compact text heading. Resumes keep their
    transcript prominent instead of repeating the large fresh-session banner.
    Rich honors NO_COLOR; body text inherits the terminal foreground.

    Args:
      console: Where the welcome is printed.
      model: The model line; empty before one is configured.
      provider: The provider serving it.
      folder: The project directory.
      holiday: Today's holiday, which the greeting names, and its icon to show
        in place of the logo. Terminals without 256 colors, or with NO_COLOR
        set, keep the logo: the icon's shape is drawn in its colors.
      theme: Frame, metadata, and help styles; artwork keeps its own palette.
      padding: Columns between the frame's sides and its content.

    """
    if not console.is_terminal:
        if model:
            console.print(Text(f"[{provider}] {model}"))
        return
    theme = theme or load_theme("terminal")
    unicode_ok = _supports_blocks(console.encoding)
    # pragma: no mutate start -- an empty block needs the 80-cell floor; a cell more does not move it.
    frame_width = fill(0, total_padding=0, width=console.width)
    # pragma: no mutate end
    many_colors = console.color_system in {"256", "truecolor"} and not console.no_color
    banner = _banner_rows(
        icon=icon_rows(holiday[1]) if holiday and many_colors else _logo_rows(),
    )
    large = (
        not console.is_dumb_terminal
        and unicode_ok
        and frame_width >= max(row.cell_len for row in banner) + 2 * (padding + 1)
        and console.height >= 22
    )
    framed = not console.is_dumb_terminal and frame_width >= 16
    print_rows = partial(
        _print_framed,
        console,
        width=frame_width,
        framed=framed,
        unicode_ok=unicode_ok,
        padding=padding,
        theme=theme,
    )
    console.print()
    if framed:
        left, horizontal, right = ("╭", "─", "╮") if unicode_ok else ("+", "-", "+")
        console.print(
            Text(
                left + horizontal * (frame_width - 2) + right,
                style=theme.style("muted"),
            ),
        )
        print_rows([Text()])
    if large:
        print_rows(banner)
    else:
        print_rows([Text("SAGENT", style=f"bold {_BRAND_BLUE}")])
    print_rows([Text()])
    # Keep metadata inside the terminal frame.
    # Never crop a model ID; long IDs/providers fold normally. Folder paths
    # explicitly mark omitted ancestors, preserving the project suffix.
    content_width = frame_width - (2 * (padding + 1) if framed else 2)
    value_width = max(1, content_width - 10)
    display_folder = _folder_label(
        folder,
        home=Path("~").expanduser(),
        width=value_width,
        unicode_ok=unicode_ok,
    )
    display_provider = {
        "OpenAISubscription": "OpenAI Subscription",
        "AnthropicCLI": "Anthropic CLI",
        "LlamaCpp": "Llama.cpp",
    }.get(provider, provider)
    metadata = [("model", model), ("provider", display_provider)] if model else []
    separator = " · " if unicode_ok else " / "
    combined = model + separator + display_provider
    if model and Text(combined).cell_len <= value_width:
        metadata = [("model", combined)]
    metadata.append(("project", display_folder))
    for label, value in metadata:
        row = Text()
        row.append(f"{label:<10}", style=theme.style("muted"))
        row.append(value, style=theme.style("text"))
        print_rows([row])
    print_rows([Text()])
    print_rows(_command_rows(content_width, unicode_ok=unicode_ok, theme=theme))
    print_rows([Text()])
    news = f"Behold, {holiday[0]}!" if holiday else "The discovering begins."
    print_rows([Text(f"Good news, everyone! {news}", style=theme.style("text"))])
    if framed:
        print_rows([Text()])
        left, horizontal, right = ("╰", "─", "╯") if unicode_ok else ("+", "-", "+")
        console.print(
            Text(
                left + horizontal * (frame_width - 2) + right,
                style=theme.style("muted"),
            ),
        )
    console.print()


def _banner_rows(*, icon: list[Text]) -> list[Text]:
    """Return the icon beside the wordmark, one row per banner line, all one width."""
    wordmark = (
        "███████╗ █████╗  ██████╗ ███████╗███╗   ██╗████████╗",
        "██╔════╝██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝",
        "███████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║",
        "╚════██║██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║",
        "███████║██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║",
        "╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝",
    )
    icon_width = max(row.cell_len for row in icon)
    width = icon_width + 3 + max(len(row) for row in wordmark)
    rows: list[Text] = []
    for icon_row, letters in zip(icon, wordmark, strict=True):
        lettering = icon_row.copy()
        lettering.append(" " * (icon_width - icon_row.cell_len + 3))
        for char in letters:
            style = f"dim {_BRAND_BLUE}" if char in "╔╗╚╝═║" else _BRAND_BLUE
            lettering.append(char, style=style)
        # Keep the lockup rectangular, including its final alignment row.
        lettering.append(" " * (width - lettering.cell_len))
        rows.append(lettering)
    return rows


def _logo_rows() -> list[Text]:
    """Return the document-loop logo, with its three colored bars."""
    # Terminal adaptation of the supplied document-loop icon, with its three bars.
    logo = (
        "   ╔══════╗",
        "   ▼      ║",
        " ██████   ║",
        " █████    ║",
        " ███████  ║",
        "   ╚══════╝",
    )
    rows: list[Text] = []
    for row_index, logo_row in enumerate(logo):
        row = Text()
        for char in logo_row:
            # Some terminal bold faces lack box glyphs.
            style = "not bold #808080"
            if char == "█" and 2 <= row_index <= 4:
                color = (_BRAND_BLUE, "#93c47d", "#e06666")[row_index - 2]
                style = f"bold {color}"
            row.append(char, style=style)
        rows.append(row)
    return rows


def _print_framed(
    console: Console,
    rows: list[Text],
    *,
    width: int,
    framed: bool,
    unicode_ok: bool,
    padding: int,
    theme: Theme,
) -> None:
    """Print ``rows`` inside a quiet border, wrapped to terminal cell widths."""
    if not framed:
        for row in rows:
            console.print(Text(" " * padding) + row)
        return
    side = "│" if unicode_ok else "|"
    content_width = width - 2 * (padding + 1)
    for row in rows:
        for line in row.wrap(console, content_width, overflow="fold") or [Text()]:
            framed_line = Text()
            framed_line.append(side + " " * padding, style=theme.style("muted"))
            framed_line.append_text(line)
            framed_line.append(
                " " * (content_width - line.cell_len + padding) + side,
                style=theme.style("muted"),
            )
            console.print(framed_line)


def _command_rows(width: int, *, unicode_ok: bool, theme: Theme) -> list[Text]:
    """Keep each command beside its description when the terminal is narrow."""
    commands = (
        ("/help", "commands"),
        # Left on an empty prompt line opens the team's tree.
        ("←" if unicode_ok else "Left", "agents"),
        ("/tasks", "tool calls"),
        ("/quit", "exit"),
    )
    show_descriptions = width >= max(
        len(command) + len(label) + 1 for command, label in commands
    )
    rows: list[Text] = []
    row = Text()
    for command, description in commands:
        item = Text()
        item.append(command, style=theme.style("heading"))
        if show_descriptions:
            item.append(" " + description, style=theme.style("muted"))
        if row and row.cell_len + 3 + item.cell_len > width:
            rows.append(row)
            row = Text()
        if row:
            row.append("   ")
        row.append_text(item)
    rows.append(row)
    return rows


def _supports_blocks(encoding: str) -> bool:
    try:
        _ = "█╔╗╚╝═║━…╭╮╰╯─│▼←".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _folder_label(folder: Path, *, home: Path, width: int, unicode_ok: bool) -> str:
    """Abbreviate long ancestor paths while retaining the visible project name."""
    display = (
        str(Path("~") / folder.relative_to(home))
        if folder.is_relative_to(home)
        else str(folder)
    )
    if Text(display).cell_len <= width:
        return display
    marker = "…" if unicode_ok else "..."
    prefix = "~/" if folder.is_relative_to(home) else ""
    parts = folder.parts
    for start in range(1, len(parts)):
        candidate = prefix + marker + "/" + "/".join(parts[start:])
        if Text(candidate).cell_len <= width:
            return candidate
    # A single directory name can itself be wider than the terminal. Mark
    # truncation on its left rather than overflowing the metadata column.
    suffix = Text(folder.name)
    while suffix.cell_len > max(0, width - len(marker)):
        suffix = suffix[1:]
    return marker[:width] + suffix.plain
