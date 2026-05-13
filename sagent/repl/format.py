"""Leaf terminal-formatting utilities used by the REPL.

Pure helpers shared between :mod:`repl.console_pane`, :mod:`repl.replay`,
:mod:`repl.status_pane`, and any external surface (orchestrator stdout,
slack adapter, headless output) that needs the same formatting:

- :func:`print_user_bar` -- full-width dark-gray user-message bar.
- :func:`set_terminal_title` -- OSC 0 title escape, no-op off TTY.
- :func:`format_elapsed` -- compact duration with seconds always shown,
  growing into ``m`` / ``h`` / ``d`` as it spills over.
- :func:`format_count` -- abbreviate token counts (``12K``, ``1.8M``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import contextlib
import sys

from rich.cells import chop_cells
from rich.text import Text


if TYPE_CHECKING:
    from rich.console import Console


def print_user_bar(
    out: Console,
    text: str,
    user_msg_style: str = "on rgb(55,55,55)",
) -> None:
    """Render a user message as full-width dark-gray bar(s).

    Args:
      out: Rich console to print to.
      text: User message text.
      user_msg_style: Rich style string for the bar background.

    """
    try:
        width = int(out.width)
    except (TypeError, ValueError):
        width = 0
    if width <= 2:
        out.print(Text(f"> {text}", style="white"), style=user_msg_style)
        return
    for idx, raw in enumerate(text.rstrip("\n").split("\n")):
        prefix = "> " if idx == 0 else "  "
        content_width = width - len(prefix)
        chunks = chop_cells(raw, content_width) or [""]
        for j, chunk in enumerate(chunks):
            pfx = prefix if j == 0 else "  "
            line = Text()
            line.append(pfx, style="white")
            line.append(chunk, style="white")
            pad = max(0, width - line.cell_len)
            if pad > 0:
                line.append(" " * pad)
            out.print(line, style=user_msg_style)


def set_terminal_title(text: str, max_len: int = 80) -> None:
    r"""Write an OSC 0 (icon+window title) escape to stdout.

    Safe under ``patch_stdout`` -- the proxy routes writes above the
    prompt without tearing it down. No-op when not a tty.

    Args:
      text: Title text to set.
      max_len: Maximum character length before truncation.

    """
    if not sys.stderr.isatty():
        return
    one_line = text.replace("\n", " ").strip()
    if len(one_line) > max_len:
        one_line = one_line[: max_len - 1] + "…"
    with contextlib.suppress(OSError):
        sys.stderr.write(f"\x1b]0;{one_line}\x07")
        sys.stderr.flush()


def format_elapsed(seconds: float) -> str:
    """Format a duration with seconds always shown; spill into m / h / d.

    Args:
      seconds: Elapsed time in seconds. Sub-second values floor to ``"0s"``.

    Returns:
      formatted: E.g. ``"12s"``, ``"1m 23s"``, ``"2h 17m 4s"``,
          ``"3d 1h 2m 5s"``.

    """
    s = int(seconds)
    days, rem = divmod(s, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if days or hours:
        parts.append(f"{hours}h")
    if days or hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def format_count(n: int) -> str:
    """Abbreviate large token counts; sub-10K values render verbatim.

    Args:
      n: Token count.

    Returns:
      formatted: E.g. ``"412"``, ``"12K"``, ``"1.8M"``.

    """
    if n < 10_000:
        return str(n)
    # Threshold lifted to 999_500 so banker's-rounded ``f"{n/1000:.0f}K"``
    # never produces ``"1000K"`` -- step straight to the M scale instead.
    if n < 999_500:
        return f"{n / 1000:.0f}K"
    return f"{n / 1_000_000:.1f}M"
