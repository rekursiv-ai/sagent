"""Compaction orchestration helpers -- extracted from Agent."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import dataclasses
import logging
import time

from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.state import ToolState
from sagent.lib.compaction import reattach_files
from sagent.types.compactor import CompactRestorable
from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    UserMessage,
)
from sagent.types.model import ContextBudget, Model
from sagent.types.tools import Tool


logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True)
class CompactionState:
    """Mutable compaction bookkeeping owned by the Agent."""

    compact_count: int = 0
    """Number of successful compactions applied."""

    compact_failures: int = 0
    """Count of compactor errors since last success."""

    summary_pointers: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    """``(path, topic)`` pairs to ``summary_<N>.md`` files."""

    compacting: bool = False
    """True while a compaction is in flight; gates re-entry."""


_CONTINUATION_MARKER = "continued from a previous"


def is_summary(continuation: str) -> bool:
    """Decide whether a continuation string is a real compactor summary.

    Args:
      continuation: Candidate continuation text from the compactor.

    Returns:
      is_real: True when the marker phrase is present (not an error msg).

    """
    return bool(continuation) and _CONTINUATION_MARKER in continuation


def extract_topic(continuation: str) -> str:
    """Pick the first substantive line of a continuation as a summary topic.

    Args:
      continuation: Compactor-produced summary text.

    Returns:
      topic: First non-bullet, non-heading line (≤120 chars), or a
          fallback when nothing qualifies.

    """
    for line in continuation.splitlines():
        s = line.strip()
        if s and not s.startswith("- ") and not s.endswith(":"):
            return s[:120]
    return "(compacted context)"


def append_to_first_user(history: list[HistoryEntry], text: str) -> None:
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
    history: list[HistoryEntry],
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
    result: list[HistoryEntry],
    history: list[HistoryEntry],
    state: CompactionState,
    session_dir: Path | None,
    tool_state: ToolState,
    budget: ContextBudget,
    tools: Mapping[str, Tool],
    background_tasks: Mapping[str, BackgroundTaskEntry],
    estimate_tokens: int,
    headroom: int,
) -> None:
    """Run best-effort post-compaction enrichment pipeline.

    Runs 4 steps; each is individually isolated so a failure in one
    doesn't block the others: save summary to disk, re-attach recent
    files, invoke tool ``post_compact_restore`` hooks, re-surface
    background-job status.

    Args:
      result: The compactor's output history (first user message holds
          the continuation summary).
      history: History to enrich in place; usually the same list as
          ``result`` post-splice.
      state: Compaction bookkeeping; ``summary_pointers`` is appended to.
      session_dir: Where ``summary_<N>.md`` is written; ``None`` skips
          the disk save step.
      tool_state: Active tool state; ``recent_files`` drives re-attach.
      budget: Context budget for re-attach sizing and hook budgets.
      tools: Tool registry; entries implementing ``CompactRestorable``
          are notified.
      background_tasks: Live background-task registry.
      estimate_tokens: Tokens still available for hook content.
      headroom: Reserved tokens (response + buffer) deducted from the
          hook budget.

    """
    # 1. Save summary to file and accumulate pointer.
    try:
        continuation = (
            result[0].text if result and isinstance(result[0], UserMessage) else ""
        )
        if session_dir is not None and is_summary(continuation):
            sp = session_dir / f"summary_{state.compact_count}.md"
            sp.write_text(continuation, encoding="utf-8")
            topic = extract_topic(continuation)
            state.summary_pointers.append((str(sp), topic))
    except Exception:  # noqa: BLE001 -- provider errors are heterogeneous
        logger.warning("Summary save failed", exc_info=True)

    # 2. Re-attach recently-read files.
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

    # 3. Run post-compact hooks on tools.
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

    # 4. Re-surface background job status.
    try:
        inject_background_status(history, background_tasks)
    except Exception:  # noqa: BLE001 -- provider errors are heterogeneous
        logger.warning("inject_background_status failed", exc_info=True)


def estimate_total_tokens(
    system: str,
    history: list[HistoryEntry],
    model: Model,
) -> int:
    """Estimate total input tokens for a system prompt plus history.

    Args:
      system: System prompt text.
      history: Conversation history.
      model: Model whose tokenizer estimates are used.

    Returns:
      tokens: Estimated total input token count.

    """
    total = model.estimate_text_token_count(system)
    for entry in history:
        total += _estimate_entry_tokens(entry, model)
    return total


def _estimate_entry_tokens(entry: HistoryEntry, model: Model) -> int:
    """Estimate tokens for one history entry (text plus image attachments)."""
    if isinstance(entry, UserMessage):
        total = model.estimate_text_token_count(entry.text)
        for att in entry.attachments:
            if att.descriptor.startswith("image/"):
                total += model.estimate_image_token_count(att.data)
        return total
    if isinstance(entry, AssistantMessage):
        return model.estimate_text_token_count(entry.text)
    # ToolResult
    total = model.estimate_text_token_count(entry.content)
    for att in entry.attachments:
        if att.descriptor.startswith("image/"):
            total += model.estimate_image_token_count(att.data)
    return total
