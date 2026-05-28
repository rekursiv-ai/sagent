"""Compaction orchestration helpers -- extracted from Agent."""

from __future__ import annotations

from collections.abc import Mapping

import dataclasses
import logging
import time

from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.state import ToolState
from sagent.lib.compaction import reattach_files
from sagent.types.compactor import CompactRestorable
from sagent.types.model import ContextBudget
from sagent.types.runtime import (
    ModelContextEvent,
    UserMessage,
)
from sagent.types.tools import Tool


logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_COMPACT_FAILURES = 3
"""Auto-compaction circuit breaker.

After this many consecutive auto-compact failures, ``compact_if_needed``
short-circuits (returns ``False``) without invoking the compactor. The
caller surfaces the underlying error rather than retrying a broken
compactor indefinitely. Reset on any successful compaction.
"""


@dataclasses.dataclass(kw_only=True, slots=True)
class CompactionState:
    """Mutable compaction bookkeeping owned by the Agent."""

    compact_count: int = 0
    """Number of successful compactions applied."""

    compact_failures: int = 0
    """Count of consecutive compactor errors. Reset to 0 on success."""

    compacting: bool = False
    """True while a compaction is in flight; gates re-entry."""


def append_to_first_user(history: list[ModelContextEvent], text: str) -> None:
    """Append ``text`` to the first ``UserMessage`` in ``history``, or insert one.

    The compactor and its post-enrich steps inject context (reattached
    files, background-task status, skill bodies) ahead of the prompt.
    The simplest place is the first user message; if none exists yet
    (e.g. compactor returned an assistant-led summary), insert a fresh
    one at position 0.

    Args:
      history: History to mutate in place.
      text: Content to append (or seed a new ``UserMessage`` with).

    """
    for j, entry in enumerate(history):
        if isinstance(entry, UserMessage):
            joined = f"{entry.text}\n\n{text}" if entry.text else text
            history[j] = dataclasses.replace(entry, text=joined)
            return
    history.insert(0, UserMessage(text=text))


def inject_background_status(
    history: list[ModelContextEvent],
    background_tasks: Mapping[str, BackgroundTaskEntry],
) -> None:
    """Re-surface running background jobs after compaction.

    Args:
      history: History to mutate in place via ``append_to_first_user``.
      background_tasks: Live background-task registry; status lines are
          built from each entry's ``tool_name`` / ``started`` / ``done``.

    """
    lines: list[str] = []
    now = time.time()
    for qid, job in background_tasks.items():
        done = job.task.done()
        status = "completed" if done else "running"
        elapsed = int(now - job.started)
        lines.append(f"- [{qid}] {job.tool_name}: {status} ({elapsed}s ago)")
    if lines:
        text = "Active background tasks (re-surfaced post-compaction):\n" + "\n".join(
            lines
        )
        append_to_first_user(history, text)


async def post_compact_enrich(
    *,
    history: list[ModelContextEvent],
    tool_state: ToolState,
    budget: ContextBudget,
    tools: Mapping[str, Tool],
    background_tasks: Mapping[str, BackgroundTaskEntry],
    estimate_tokens: int,
    headroom: int,
) -> None:
    """Run best-effort post-compaction enrichment pipeline.

    Runs 3 steps; each is individually isolated so a failure in one
    doesn't block the others: re-attach recent files, invoke tool
    ``post_compact_restore`` hooks, re-surface background-job status.

    Args:
      history: Payload-under-construction; mutated in place.
      tool_state: Active tool state; ``recent_files`` drives re-attach.
      budget: Context budget for re-attach sizing and hook budgets.
      tools: Tool registry; entries implementing ``CompactRestorable``
          are notified.
      background_tasks: Live background-task registry.
      estimate_tokens: Tokens still available for hook content.
      headroom: Reserved tokens (response + buffer) deducted from the
          hook budget.

    """
    # 1. Re-attach recently-read files.
    try:
        await reattach_files(
            history,
            tool_state.recent_files,
            count=budget.reattach_count,
            max_chars=budget.reattach_max_chars,
            budget=budget.reattach_budget,
        )
    except Exception:  # noqa: BLE001 -- provider errors are heterogeneous
        logger.warning("reattach_files failed", exc_info=True)

    # 2. Run post-compact hooks on tools.
    available = max(0, estimate_tokens - headroom)
    hook_budget = available * budget.chars_per_token
    for tool in tools.values():
        if isinstance(tool, CompactRestorable):
            try:
                await tool.post_compact_restore(
                    history,
                    tool_state,
                    budget_chars=hook_budget,
                )
            except Exception:  # noqa: BLE001 -- provider errors are heterogeneous
                logger.warning(
                    "post_compact_restore failed for %s",
                    tool.__class__.__name__,
                    exc_info=True,
                )

    # 3. Re-surface background job status.
    try:
        inject_background_status(history, background_tasks)
    except Exception:  # noqa: BLE001 -- provider errors are heterogeneous
        logger.warning("inject_background_status failed", exc_info=True)
