"""Status renderer for the REPL (the line below the input bar).

Single bracketed block of session totals plus a braille spinner while
a model call is in flight. Reads ``agent.activity`` (an
:class:`ActivityTracker` on the Agent) for elapsed time and
live-call state; reads ``agent.cost_tracker`` for token / cost
counters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio

from sagent.repl.format import format_count, format_elapsed


if TYPE_CHECKING:
    from sagent.agent.agent import Agent

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def render_status_pane(agent: Agent) -> str:
    """Build the status-pane string for ``agent``.

    Format: ``[elapsed input↑ output↓ cw↟ cr↡ $cost]``. Active runs
    prefix a braille spinner that ticks live; the bracket values snap
    in at each model-call boundary.

    Args:
      agent: Agent whose totals drive the status line.

    Returns:
      status: Formatted status string, or ``""`` before the first run.

    """
    activity = agent.activity
    if (
        activity.elapsed_seconds <= 0
        and not activity.active
        and activity.current_compact_start <= 0.0
    ):
        return ""
    tokens = agent.cost_tracker.total
    cost = float(agent.cost_tracker.total_cost_usd)
    live_response_tokens = (
        activity.live_response_chars // agent.budget.chars_per_token
        if activity.active
        else 0
    )
    output_tokens = tokens.output_tokens + live_response_tokens
    elapsed = activity.elapsed_seconds
    if activity.active:
        elapsed += asyncio.get_running_loop().time() - activity.current_call_start
    bracket = (
        f"[{format_elapsed(elapsed)}"
        f" {format_count(tokens.input_tokens)}↑"
        f" {format_count(output_tokens)}↓"
        f" {format_count(tokens.cache_creation_tokens)}↟"
        f" {format_count(tokens.cache_read_tokens)}↡"
        f" ${cost:.2f}]"
    )
    if activity.current_compact_start > 0.0:
        live_delta = asyncio.get_running_loop().time() - activity.current_compact_start
        frame = _SPINNER[int(live_delta * 5) % len(_SPINNER)]
        return f"{frame} [compacting] {bracket}"
    if activity.active:
        live_delta = asyncio.get_running_loop().time() - activity.current_call_start
        frame = _SPINNER[int(live_delta * 5) % len(_SPINNER)]
        return f"{frame} {bracket}"
    return bracket
