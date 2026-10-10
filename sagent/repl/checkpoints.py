"""Persistent presentation of explicit reports, without inferring job lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import re

from rich.text import Text

from sagent.types.runtime import Checkpoint


if TYPE_CHECKING:
    from sagent.agent.agent import Agent


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckpointEntry:
    """A report with canonical sender and freshness information."""

    label: str
    checkpoint: Checkpoint
    observed_at: float
    restored: bool


def checkpoint_entries(agent: Agent) -> list[CheckpointEntry]:
    """Read this agent and descendants only; independent senders never coalesce."""
    label = agent.name or "Agent"
    entries: list[CheckpointEntry] = []
    checkpoint = getattr(agent, "checkpoint", None)
    if isinstance(checkpoint, Checkpoint):
        entries.append(
            CheckpointEntry(
                label=label,
                checkpoint=checkpoint,
                observed_at=agent.checkpoint_updated_at,
                restored=agent.checkpoint_restored,
            ),
        )
    for source, change in getattr(agent, "child_checkpoints", {}).items():
        if change.checkpoint is not None:
            entries.append(
                CheckpointEntry(
                    label=f"{label}/{source}",
                    checkpoint=change.checkpoint,
                    observed_at=change.observed_at,
                    restored=source in agent.restored_child_checkpoints,
                ),
            )
    return entries


def checkpoint_text(checkpoint: Checkpoint | None) -> str:
    """Format a complete report; empty fields explicitly mean none reported."""
    if checkpoint is None:
        return "Checkpoint cleared (agent report)."
    return "\n".join(
        (
            "Checkpoint (agent report)",
            f"Progress: {checkpoint.progress}",
            f"Pending: {checkpoint.pending or 'None reported.'}",
            f"User action: {checkpoint.user_action or 'None reported.'}",
        ),
    )


def format_pending(agent: Agent) -> str:
    """Full latest reports for local inspection, including saved-state caveats."""
    entries = checkpoint_entries(agent)
    if not entries:
        return "No checkpoint reported. Agent idleness does not establish assignment completion."
    blocks: list[str] = []
    for entry in entries:
        when = (
            datetime.fromtimestamp(entry.observed_at, UTC).isoformat()
            if entry.observed_at > 0
            else "unknown time"
        )
        heading = f"{entry.label} | reported {when}"
        if entry.restored:
            heading += (
                "\nSaved checkpoint; current pending work has not been rechecked."
            )
        blocks.append(heading + "\n" + checkpoint_text(entry.checkpoint))
    return "\n\n".join(blocks)


def render_pending_pane(agent: Agent, *, width: int) -> str:
    """Bound the persistent report to four physical rows; /pending shows full text."""
    entries = checkpoint_entries(agent)
    if not entries:
        return ""
    actionable = [entry for entry in entries if entry.checkpoint.user_action]
    latest = max(actionable or entries, key=lambda entry: entry.observed_at)
    heading = "Saved, not rechecked" if latest.restored else "Agent report"
    lines = (
        f"/pending ({len(entries)}) | {heading}: {latest.label}",
        f"Progress: {latest.checkpoint.progress}",
        f"Pending: {latest.checkpoint.pending or 'None reported.'}",
        f"User action: {latest.checkpoint.user_action or 'None reported.'}",
    )
    return "\n".join(_clip(line, width) for line in lines)


def _clip(line: str, width: int) -> str:
    """Clip terminal cells, keeping control bytes and multiline text out of a row."""
    if width <= 0:
        return ""
    text = Text(re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", line))
    text.truncate(max(0, width), overflow="ellipsis")
    return text.plain
