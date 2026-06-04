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
import time

from sagent.repl.format import format_count, format_elapsed


if TYPE_CHECKING:
    from sagent.agent.agent import Agent

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_WAIT_REASON_THRESHOLD_SEC = 15.0


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
    # ``output_tokens`` includes a live char-count estimate so the user
    # sees a moving counter while the model streams; ``cost`` is
    # intentionally settled-only -- pricing requires the provider-
    # reported usage block, which only lands on response completion.
    # The ``$cost`` field thus lags the ``output↓`` counter by one
    # model-call boundary; the discrepancy snaps back in at each
    # ``ModelResponseComplete``.
    live_response_tokens = (
        activity.live_response_chars // agent.budget.chars_per_token
        if activity.active
        else 0
    )
    output_tokens = tokens.output_tokens + live_response_tokens
    elapsed = activity.elapsed_seconds
    if activity.active:
        elapsed += asyncio.get_running_loop().time() - activity.current_call_start
    metrics = (
        f"{format_elapsed(elapsed)}"
        f" {format_count(tokens.input_tokens)}↑"
        f" {format_count(output_tokens)}↓"
        f" {format_count(tokens.cache_creation_tokens)}↟"
        f" {format_count(tokens.cache_read_tokens)}↡"
        f" ${cost:.2f}"
    )
    # The wait-reason threshold is per-turn, not session-cumulative:
    # users want "waiting on model." to appear when the CURRENT call has
    # been in flight too long, not because total session time crossed
    # a threshold that resets at session boundaries only.
    current_turn_elapsed = (
        asyncio.get_running_loop().time() - activity.current_call_start
        if activity.active
        else 0.0
    )
    reason = (
        _wait_reason(agent, current_turn_elapsed) if _has_live_activity(agent) else ""
    )
    bracket = f"[{metrics}{f'; {reason}' if reason else ''}]"
    if activity.current_compact_start > 0.0:
        live_delta = asyncio.get_running_loop().time() - activity.current_compact_start
        frame = _SPINNER[int(live_delta * 5) % len(_SPINNER)]
        return f"{frame} {bracket}"
    if activity.active:
        live_delta = asyncio.get_running_loop().time() - activity.current_call_start
        frame = _SPINNER[int(live_delta * 5) % len(_SPINNER)]
        return f"{frame} {bracket}"
    return bracket


def _has_live_activity(agent: Agent) -> bool:
    return agent.activity.active or agent.activity.current_compact_start > 0.0


def _wait_reason(agent: Agent, elapsed: float) -> str:
    runtime = agent.runtime
    # ``service_suspended_until`` is a Unix wall-clock epoch
    # (``ModelServiceSuspended.retry_at``), so the countdown compares against
    # ``time.time()`` -- NOT the monotonic loop clock, which would render a
    # multi-decade wait (Issue#316 RUNTIME-003).
    suspended_until = runtime.service_suspended_until
    if suspended_until is not None and suspended_until > time.time():
        return f"retrying in {format_elapsed(suspended_until - time.time())}."
    # Compaction outranks the gate-armed check: during compaction the
    # gate can be armed (waiting for the compactor's own model response)
    # but the user-visible state is "compacting", not "waiting for
    # input". Reporting "waiting for input" while compacting would
    # invite the user to type when no input is actually accepted.
    if agent.activity.current_compact_start > 0.0:
        return "compacting."
    if runtime.inbox.gate_armed:
        return "waiting for input."
    if not agent.activity.active:
        return ""
    if elapsed < _WAIT_REASON_THRESHOLD_SEC:
        return ""
    if runtime.running_tools or runtime.cohort:
        return "waiting on tools."
    return "waiting on model."
