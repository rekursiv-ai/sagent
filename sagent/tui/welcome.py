"""Local startup identity and orientation for the interactive CLI."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from sagent.tui.types import fill


if TYPE_CHECKING:
    from rich.console import Console
    from rich.text import Text
else:
    from wrapt import lazy_import

    Console = lazy_import("rich.console", "Console")
    Text = lazy_import("rich.text", "Text")


def print_welcome(
    *,
    model: str = "",
    provider: str = "",
    folder: Path,
) -> None:
    """Print identity and help before credentials or prompt-toolkit are needed.

    Args:
      model: The model line; empty before one is configured.
      provider: The provider serving it.
      folder: The project directory.

    """
    render_welcome(
        Console(stderr=True),
        model=model,
        provider=provider,
        folder=folder,
    )


# Text cells only: no image protocol, extra font, or background color required.
# The blue is slightly darker than the logo reference to work on light terminals.
def render_welcome(
    console: Console,
    *,
    model: str = "",
    provider: str = "",
    folder: Path,
    blue: str = "#4a90c4",
    green: str = "#93c47d",
    red: str = "#e06666",
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
      blue: The wordmark's color, and the first bar's.
      green: The second bar's color.
      red: The third bar's color.
      padding: Columns between the frame's sides and its content.

    """
    if not console.is_terminal:
        if model:
            console.print(Text(f"[{provider}] {model}"))
        return
    unicode_ok = _supports_blocks(console.encoding)
    frame_width = fill(0, total_padding=0, width=console.width)
    banner = _banner_rows(blue=blue, green=green, red=red)
    large = (
        not console.is_dumb_terminal
        and unicode_ok
        and frame_width >= banner[0].cell_len + 2 * (padding + 1)
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
    )
    console.print()
    if framed:
        left, horizontal, right = ("╭", "─", "╮") if unicode_ok else ("+", "-", "+")
        console.print(Text(left + horizontal * (frame_width - 2) + right, style="dim"))
        print_rows([Text()])
    if large:
        print_rows(banner)
    else:
        print_rows([Text("SAGENT", style=f"bold {blue}")])
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
        row = Text(f"{label:<10}", style="dim")
        row.append(value, style="not dim")
        print_rows([row])
    print_rows([Text()])
    print_rows(_command_rows(max(1, content_width), unicode_ok=unicode_ok))
    if framed:
        print_rows([Text()])
        left, horizontal, right = ("╰", "─", "╯") if unicode_ok else ("+", "-", "+")
        console.print(Text(left + horizontal * (frame_width - 2) + right, style="dim"))
    console.print()
    greeting = Text("Good news, everyone! The discovering begins.")
    for line in greeting.wrap(console, frame_width):
        console.print(line)
    console.print()


def _banner_rows(*, blue: str, green: str, red: str) -> list[Text]:
    """Return the icon beside the wordmark, one row per banner line, all one width."""
    # Terminal adaptation of the supplied document-loop icon, with its three bars.
    icon = (
        "   ╔══════╗",
        "   ▼      ║",
        " ██████   ║",
        " █████    ║",
        " ███████  ║",
        "   ╚══════╝",
    )
    wordmark = (
        "███████╗ █████╗  ██████╗ ███████╗███╗   ██╗████████╗",
        "██╔════╝██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝",
        "███████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║",
        "╚════██║██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║",
        "███████║██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║",
        "╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝",
    )
    icon_width = max(len(row) for row in icon)
    width = icon_width + 3 + max(len(row) for row in wordmark)
    rows: list[Text] = []
    for row_index, (icon_row, letters) in enumerate(zip(icon, wordmark, strict=True)):
        lettering = Text()
        for char in icon_row.ljust(icon_width):
            # Box glyphs share the wordmark's regular face. Some terminal bold
            # faces lack them; the double-line shape supplies their weight.
            style = "not bold"
            if char == "█" and 2 <= row_index <= 4:
                style = f"bold {(blue, green, red)[row_index - 2]}"
            lettering.append(char, style=style)
        lettering.append("   ")
        for char in letters:
            style = f"dim {blue}" if char in "╔╗╚╝═║" else blue
            lettering.append(char, style=style)
        # Keep the lockup rectangular, including its final alignment row.
        lettering.append(" " * (width - lettering.cell_len))
        rows.append(lettering)
    return rows


def _print_framed(
    console: Console,
    rows: list[Text],
    *,
    width: int,
    framed: bool,
    unicode_ok: bool,
    padding: int,
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
            framed_line.append(side + " " * padding, style="dim")
            framed_line.append_text(line)
            framed_line.append(
                " " * (content_width - line.cell_len + padding) + side,
                style="dim",
            )
            console.print(framed_line)


def _command_rows(width: int, *, unicode_ok: bool) -> list[Text]:
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
        item = Text(command, style="bold")
        if show_descriptions:
            item.append(" " + description, style="dim not bold")
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
