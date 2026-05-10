"""Compaction orchestration helpers -- extracted from Agent."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import dataclasses
import logging
import time

from sagent.custom_types import (
    CompactRestorable,
    ContextBudget,
    Message,
    Model,
    Tool,
)
from sagent.lib.compaction import reattach_files
from sagent.lib.descriptors import is_binary, is_multipart
from sagent.lib.message import append_to_first_user_message
from sagent.tools.background_task import BackgroundTaskEntry
from sagent.tools.core import ToolState


logger = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True)
class CompactionState:
    """Mutable compaction bookkeeping owned by the Agent."""

    compact_count: int = 0
    compact_failures: int = 0
    summary_pointers: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    compacting: bool = False


_CONTINUATION_MARKER = "continued from a previous"


def is_summary(continuation: str) -> bool:
    """Return True if the continuation is a real summary, not a failure message.

    Args:
      continuation: First message content after compaction.

    Returns:
      valid: Whether the text contains the continuation marker.

    """
    return bool(continuation) and _CONTINUATION_MARKER in continuation


def extract_topic(continuation: str) -> str:
    """Extract the first substantive line of a continuation for summary pointers.

    Args:
      continuation: Compaction summary text.

    Returns:
      topic: Truncated first content line, or a fallback placeholder.

    """
    for line in continuation.splitlines():
        s = line.strip()
        if s and not s.startswith("- ") and not s.endswith(":"):
            return s[:120]
    return "(compacted context)"


def inject_background_status(
    messages: list[Message],
    background_tasks: dict[str, BackgroundTaskEntry],
) -> None:
    """Re-surface running background jobs after compaction.

    Args:
      messages: Post-compaction conversation (mutated in-place).
      background_tasks: Active background tasks keyed by queue id.

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
        append_to_first_user_message(messages, text)


async def post_compact_enrich(
    *,
    result: list[Message],
    messages: list[Message],
    state: CompactionState,
    session_dir: Path | None,
    tool_state: ToolState,
    budget: ContextBudget,
    tools: dict[str, Tool],
    background_tasks: dict[str, BackgroundTaskEntry],
    estimate_tokens: int,
    headroom: int,
) -> None:
    """Run best-effort post-compaction enrichment pipeline.

    Runs 4 steps; each is individually isolated so a failure in one
    doesn't block the others.

    Args:
      result: Compactor output messages.
      messages: Live conversation list (mutated in-place).
      state: Mutable compaction bookkeeping.
      session_dir: Session directory for summary persistence.
      tool_state: Agent's mutable tool state.
      budget: Context budget controlling reattach limits.
      tools: Available tools for post-compact hooks.
      background_tasks: Active background tasks keyed by queue id.
      estimate_tokens: Estimated remaining token budget.
      headroom: Reserved tokens for response + buffer.

    """
    # 1. Save summary to file and accumulate pointer.
    try:
        continuation = str(result[0].content) if result else ""
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
            messages,
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
                    messages,
                    tool_state,
                    budget_chars=hook_budget,
                )
            except Exception:  # noqa: BLE001 -- provider errors are heterogeneous
                logger.warning(
                    "post_compact_restore failed for %s",
                    tool.tool_id,
                    exc_info=True,
                )

    # 4. Re-surface background job status.
    try:
        inject_background_status(messages, background_tasks)
    except Exception:  # noqa: BLE001 -- provider errors are heterogeneous
        logger.warning("inject_background_status failed", exc_info=True)


def estimate_total_tokens(
    system: str,
    messages: list[Message],
    model: Model,
) -> int:
    """Estimate total tokens for a (system, messages) pair.

    Args:
      system: System prompt text.
      messages: Conversation history.
      model: Model used for token estimation.

    Returns:
      total: Estimated input token count.

    """
    total = model.estimate_text_token_count(system)
    for msg in messages:
        total += _estimate_message_tokens(msg, model)
    return total


def _estimate_message_tokens(msg: Message, model: Model) -> int:
    """Recursively estimate tokens for one message; handles multipart + binary."""
    if is_multipart(msg.descriptor):
        return sum(
            _estimate_message_tokens(p, model)
            for p in cast("tuple[Message, ...]", msg.content)
        )
    if is_binary(msg.descriptor):
        return model.estimate_image_token_count(cast("bytes", msg.content))
    return model.estimate_text_token_count(str(msg.content))
