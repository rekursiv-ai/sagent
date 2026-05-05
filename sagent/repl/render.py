"""Leaf terminal render functions: user bar, terminal title, toolbar."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import asyncio
import contextlib
import sys
import time

from rich.cells import chop_cells
from rich.text import Text


if TYPE_CHECKING:
    from rich.console import Console

    from sagent.agent import Agent
    from sagent.tools.agent_spawn import ChildStats


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
    """Build the bottom toolbar string showing elapsed time and cost.

    Args:
      agent: Agent whose status drives the toolbar content.
      spinner: Braille spinner character sequence.

    Returns:
      toolbar: Formatted toolbar text, or empty string when idle.

    """
    if agent.active:
        elapsed = asyncio.get_running_loop().time() - agent.request_start_time
        frame = spinner[int(elapsed * 5) % len(spinner)]
        parts = [f"{frame} {format_elapsed(elapsed)}"]
        if agent.live_model_response_tokens:
            parts.append(f"{agent.live_model_response_tokens}↓")
        # While active, this is the live subtree ledger, not just the parent call.
        cost = float(agent.last_run_cost_usd)
        if cost > 0:
            parts.append(f"${cost:.2f}")
        now = time.monotonic()
        active_children = cast("dict[str, ChildStats]", agent.active_children)
        for st in active_children.values():
            if st.done:
                continue
            child_parts: list[str] = [format_elapsed(now - st.start)]
            if st.model_response_tokens:
                child_parts.append(f"{st.model_response_tokens}↓")
            if st.cost_usd > 0:
                child_parts.append(f"${st.cost_usd:.2f}")
            parts.append(f"{st.label}[{' · '.join(child_parts)}]")
        return " · ".join(parts)
    if agent.last_elapsed:
        cost = float(agent.last_run_cost_usd)
        cost_str = f" ${cost:.2f}" if cost > 0 else ""
        tokens = agent.last_run_tokens
        return (
            f"[{format_elapsed(agent.last_elapsed)}"
            f" {tokens.input_tokens}↑"
            f" {tokens.output_tokens}↓"
            f"{cost_str}]"
        )
    return ""


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
    """Format a duration as a human-readable string.

    Args:
      seconds: Elapsed time in seconds.

    Returns:
      formatted: E.g. ``"12s"``, ``"2 min 5 sec"``, ``"1 hr 3 min"``.

    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m} min {s} sec"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h} hr {m} min"
