"""Local startup identity and orientation for the interactive CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import os
import time


if TYPE_CHECKING:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.text import Text
else:
    from wrapt import lazy_import

    Console = lazy_import("rich.console", "Console")
    Group = lazy_import("rich.console", "Group")
    Live = lazy_import("rich.live", "Live")
    Text = lazy_import("rich.text", "Text")


# Text cells only: no image protocol, extra font, or background color required.
# The blue is slightly darker than the logo reference to work on light terminals.
_BLUE = "#4a90c4"
_GREEN = "#93c47d"
_RED = "#e06666"
_BANNER = (
    "███████╗ █████╗  ██████╗ ███████╗███╗   ██╗████████╗",
    "██╔════╝██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝",
    "███████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║",
    "╚════██║██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║",
    "███████║██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║",
    "╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝",
)
_BANNER_WIDTH = max(len(row) for row in _BANNER)
_OUTLINE = frozenset("╔╗╚╝═║")
# Terminal adaptation of the supplied document-loop icon, with its three bars.
_ICON = (
    "  ╔═══════╗",
    "  ▼       ║",
    "██████    ║",
    "█████     ║",
    "███████   ║",
    "  ╚═══════╝",
)
_ICON_WIDTH = max(len(row) for row in _ICON)
_ICON_GAP = 3
_FRAME_PADDING = 2
_IDENTITY_WIDTH = _ICON_WIDTH + _ICON_GAP + _BANNER_WIDTH
# Follow the open stroke from bottom left, around the page, to the arrow.
_LOOP = (
    *((5, column) for column in range(2, 11)),
    *((row, 10) for row in range(4, -1, -1)),
    *((0, column) for column in range(9, 1, -1)),
    (1, 2),
)


def print_welcome(
    *,
    model: str = "",
    provider: str = "",
    folder: Path,
    resumed: bool = False,
) -> None:
    """Print identity and help before credentials or prompt-toolkit are needed."""
    render_welcome(
        Console(stderr=True),
        model=model,
        provider=provider,
        folder=folder,
        resumed=resumed,
        animate=not bool(os.environ.get("SAGENT_NO_ANIMATION")),
    )


def render_welcome(
    console: Console,
    *,
    model: str = "",
    provider: str = "",
    folder: Path,
    resumed: bool = False,
    animate: bool = False,
) -> None:
    """Render a width-aware welcome without performing any model request.

    Redirected stderr retains the old plain model line. Small, dumb, and
    non-Unicode terminals use a compact text heading. Resumes keep their
    transcript prominent instead of repeating the large fresh-session banner.
    Rich honors NO_COLOR; body text inherits the terminal foreground.
    """
    if not console.is_terminal:
        if model:
            console.print(Text(f"[{provider}] {model}"))
        return
    unicode_ok = _supports_blocks(console.encoding)
    large = (
        not resumed
        and not console.is_dumb_terminal
        and unicode_ok
        and console.width >= _IDENTITY_WIDTH + 2 * (_FRAME_PADDING + 1)
        and console.height >= 22
    )
    framed = not console.is_dumb_terminal and console.width >= 16
    frame_width = console.width

    def framed_rows(rows: list[Text]) -> list[Text]:
        return _frame_rows(
            console, rows, width=frame_width, framed=framed, unicode_ok=unicode_ok
        )

    def print_rows(rows: list[Text]) -> None:
        for row in framed_rows(rows):
            console.print(row)

    console.print()
    if framed:
        left, horizontal, right = ("╭", "─", "╮") if unicode_ok else ("+", "-", "+")
        console.print(Text(left + horizontal * (frame_width - 2) + right, style="dim"))
        print_rows([Text()])
    if large:
        if animate and not console.no_color and console.color_system is not None:
            # Finish before prompt-toolkit takes ownership. Manual refresh keeps
            # these two loops bounded and leaves no background refresh thread.
            with Live(
                Group(*framed_rows(_banner_rows())),
                console=console,
                auto_refresh=False,
                redirect_stdout=False,
                redirect_stderr=False,
            ) as live:
                for _lap in range(2):
                    for phase in range(len(_LOOP)):
                        live.update(
                            Group(*framed_rows(_banner_rows(phase))), refresh=True
                        )
                        time.sleep(0.014)
                live.update(Group(*framed_rows(_banner_rows())), refresh=True)
        else:
            print_rows(_banner_rows())
    else:
        heading = Text("SAGENT", style=f"bold {_BLUE}")
        print_rows([heading])
    if resumed:
        print_rows([Text("Resuming your session.")])
    print_rows([Text()])
    home = Path.home()  # noqa: TID251 -- Display abbreviation only, not a per-user storage location.
    # Keep metadata inside the terminal frame.
    # Never crop a model ID; long IDs/providers fold normally. Folder paths
    # explicitly mark omitted ancestors, preserving the project suffix.
    content_width = frame_width - (2 * (_FRAME_PADDING + 1) if framed else 2)
    value_width = max(1, content_width - 10)
    display_folder = _folder_label(
        folder, home=home, width=value_width, unicode_ok=unicode_ok
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
    shell_help = Text("sagent --help", style="bold")
    shell_help.append(" startup options", style="dim not bold")
    print_rows([shell_help])
    print_rows(_command_rows(max(1, content_width)))
    if framed:
        print_rows([Text()])
        left, horizontal, right = ("╰", "─", "╯") if unicode_ok else ("+", "-", "+")
        console.print(Text(left + horizontal * (frame_width - 2) + right, style="dim"))
    console.print()
    indent = " " * (_FRAME_PADDING + 1 if framed else 2)
    greeting = Text("Hello, scientist! What are we doing today?")
    for line in greeting.wrap(console, max(1, console.width - len(indent))):
        console.print(Text(indent) + line)
    console.print()


def _banner_rows(phase: int | None = None) -> list[Text]:
    """Keep the wordmark steady while a highlight follows the bold icon loop."""
    active: set[tuple[int, int]] = (
        set() if phase is None else set(_LOOP[max(0, phase - 3) : phase + 1])
    )
    rows: list[Text] = []
    for row_index, icon in enumerate(_ICON):
        lettering = Text()
        for column, char in enumerate(icon.ljust(_ICON_WIDTH)):
            # Box glyphs share the wordmark's regular face. Some terminal bold
            # faces lack them; the double-line shape supplies their weight.
            style = "not bold"
            if char == "█" and 2 <= row_index <= 4:
                style = f"bold {(_BLUE, _GREEN, _RED)[row_index - 2]}"
            elif (row_index, column) in active:
                style = f"not bold {_BLUE}"
            lettering.append(char, style=style)
        lettering.append(" " * _ICON_GAP)
        if row_index < len(_BANNER):
            for char in _BANNER[row_index]:
                style = f"dim {_BLUE}" if char in _OUTLINE else _BLUE
                lettering.append(char, style=style)
        # Keep the lockup rectangular, including its final alignment row.
        lettering.append(" " * (_IDENTITY_WIDTH - lettering.cell_len))
        rows.append(lettering)
    return rows


def _frame_rows(
    console: Console,
    rows: list[Text],
    *,
    width: int,
    framed: bool,
    unicode_ok: bool,
) -> list[Text]:
    """Wrap content inside a quiet border using terminal cell widths."""
    if not framed:
        return [Text("  ") + row for row in rows]
    side = "│" if unicode_ok else "|"
    content_width = width - 2 * (_FRAME_PADDING + 1)
    result: list[Text] = []
    for row in rows:
        for line in row.wrap(console, content_width, overflow="fold") or [Text()]:
            framed_line = Text()
            framed_line.append(side + " " * _FRAME_PADDING, style="dim")
            framed_line.append_text(line)
            framed_line.append(
                " " * (content_width - line.cell_len + _FRAME_PADDING) + side,
                style="dim",
            )
            result.append(framed_line)
    return result


def _command_rows(width: int) -> list[Text]:
    """Keep each command beside its description when the terminal is narrow."""
    commands = (("/help", "commands"), ("/tasks", "agents"), ("/quit", "exit"))
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
        _ = "█╔╗╚╝═║━…╭╮╰╯─│▼".encode(encoding)
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
