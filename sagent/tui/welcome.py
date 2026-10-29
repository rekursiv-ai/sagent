"""Local startup identity and orientation for the interactive CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from rich.console import Console
    from rich.text import Text
else:
    from wrapt import lazy_import

    Console = lazy_import("rich.console", "Console")
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
    "   ╔══════╗",
    "   ▼      ║",
    " ██████   ║",
    " █████    ║",
    " ███████  ║",
    "   ╚══════╝",
)
_ICON_WIDTH = max(len(row) for row in _ICON)
_ICON_GAP = 3
_FRAME_PADDING = 2
_MAX_WIDTH = 80
_IDENTITY_WIDTH = _ICON_WIDTH + _ICON_GAP + _BANNER_WIDTH


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
    )


def render_welcome(
    console: Console,
    *,
    model: str = "",
    provider: str = "",
    folder: Path,
    resumed: bool = False,
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
    frame_width = min(console.width, _MAX_WIDTH)
    large = (
        not resumed
        and not console.is_dumb_terminal
        and unicode_ok
        and frame_width >= _IDENTITY_WIDTH + 2 * (_FRAME_PADDING + 1)
        and console.height >= 22
    )
    framed = not console.is_dumb_terminal and frame_width >= 16

    def framed_rows(rows: list[Text]) -> list[Text]:
        return _frame_rows(
            console,
            rows,
            width=frame_width,
            framed=framed,
            unicode_ok=unicode_ok,
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
        folder,
        home=home,
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
    greeting = Text("Good news, everyone! The discovering begins!")
    for line in greeting.wrap(console, frame_width):
        console.print(line)
    console.print()


def _banner_rows() -> list[Text]:
    """Return the icon beside the wordmark, one row per banner line."""
    rows: list[Text] = []
    for row_index, icon in enumerate(_ICON):
        lettering = Text()
        for char in icon.ljust(_ICON_WIDTH):
            # Box glyphs share the wordmark's regular face. Some terminal bold
            # faces lack them; the double-line shape supplies their weight.
            style = "not bold"
            if char == "█" and 2 <= row_index <= 4:
                style = f"bold {(_BLUE, _GREEN, _RED)[row_index - 2]}"
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
