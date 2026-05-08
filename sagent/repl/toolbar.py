"""Bottom-toolbar renderer for the v2 REPL.

Mirrors v1's ``render_toolbar``: single bracketed block of session
totals plus a braille spinner while a model call is in flight.

Reads the agent's ``activity`` (an :class:`ActivityTracker` registered
on the agent) for elapsed time and live-call state; reads
``cost_tracker`` for token / cost counters. The activity tracker is
maintained by :class:`ActivityHandler` (see ``handlers/activity.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio

from sagent.repl.format import format_count, format_elapsed


if TYPE_CHECKING:
    from sagent.agent.agent import Agent


_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def render_toolbar(agent: Agent) -> str:
    """Build the bottom-toolbar string for ``agent``.

    Format: ``[elapsed input↑ output↓ cw↟ cr↡ $cost]``. Active runs
    prefix a braille spinner that ticks live; the bracket values snap
    in at each model-call boundary.

    Args:
      agent: Agent whose totals drive the toolbar.

    Returns:
      toolbar: Formatted toolbar string, or ``""`` before the first run.

    """
    activity = agent.activity
    if activity.elapsed_seconds <= 0 and not activity.active:
        return ""
    tokens = agent.cost_tracker.total
    cost = float(agent.cost_tracker.total_cost_usd)
    output_tokens = tokens.output_tokens + (
        agent.live_model_response_tokens if activity.active else 0
    )
    bracket = (
        f"[{format_elapsed(activity.elapsed_seconds)}"
        f" {format_count(tokens.input_tokens)}↑"
        f" {format_count(output_tokens)}↓"
        f" {format_count(tokens.cache_creation_tokens)}↟"
        f" {format_count(tokens.cache_read_tokens)}↡"
        f" ${cost:.2f}]"
    )
    if activity.active:
        live_delta = asyncio.get_running_loop().time() - activity.current_call_start
        frame = _SPINNER[int(live_delta * 5) % len(_SPINNER)]
        return f"{frame} {bracket}"
    return bracket
