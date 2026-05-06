"""Leaf terminal render functions: user bar, terminal title, toolbar."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import contextlib
import sys

from rich.cells import chop_cells
from rich.text import Text


if TYPE_CHECKING:
    from rich.console import Console

    from sagent.agent import Agent


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


def render_toolbar(
    agent: Agent,
    spinner: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
) -> str:
    """Build the bottom toolbar string showing session-cumulative stats.

    Single bracketed block: ``[elapsed input↑ output↓ cw↟ cr↡ $cost]``.
    Active runs prefix a braille spinner; the elapsed value ticks live
    while token / cost values snap in at each model-call boundary.

    Args:
      agent: Agent whose session totals drive the toolbar content.
      spinner: Braille spinner character sequence.

    Returns:
      toolbar: Formatted toolbar text, or empty string before the first run.

    """
    elapsed = agent.total_active_elapsed_seconds
    if elapsed <= 0 and not agent.active:
        return ""
    tokens = agent.total_tokens
    cost = float(agent.total_cost_usd)
    # Add the in-flight chars/4 estimate to output count so the bracket
    # ticks during long single generations rather than freezing until
    # the next ``cost_tracker.record()`` lands.
    output_tokens = tokens.output_tokens + (
        agent.live_model_response_tokens if agent.active else 0
    )
    bracket = (
        f"[{format_elapsed(elapsed)}"
        f" {format_count(tokens.input_tokens)}↑"
        f" {format_count(output_tokens)}↓"
        f" {format_count(tokens.cache_creation_tokens)}↟"
        f" {format_count(tokens.cache_read_tokens)}↡"
        f" ${cost:.2f}]"
    )
    if agent.active:
        live_delta = asyncio.get_running_loop().time() - agent.request_start_time
        frame = spinner[int(live_delta * 5) % len(spinner)]
        return f"{frame} {bracket}"
    return bracket


def set_terminal_title(text: str, max_len: int = 80) -> None:
    r"""Write an OSC 0 (icon+window title) escape to stdout.

    Safe under ``patch_stdout`` -- the proxy routes writes above the
    prompt without tearing it down.  No-op when not a tty.

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
    """Format a duration as a compact string with integer-second granularity.

    Args:
      seconds: Elapsed time in seconds. Sub-second values floor to ``"0s"``.

    Returns:
      formatted: E.g. ``"12s"``, ``"1m 23s"``, ``"2h 17m"``.

    """
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s"
    h, rem = divmod(s, 3600)
    return f"{h}h {rem // 60}m"


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
