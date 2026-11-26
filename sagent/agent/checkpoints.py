"""Pending-work reports held in the root session's standing records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from sagent.journal.types import UncategorizedRecord


if TYPE_CHECKING:
    from sagent.agent.types import Session


@dataclass(frozen=True, slots=True, kw_only=True)
class Checkpoint:
    """A worker's handoff. Empty pending and user_action fields mean none."""

    progress: Annotated[str, "Work completed so far."]
    pending: Annotated[str, "What is still waiting, or empty."]
    user_action: Annotated[str, "Action needed from the user, or empty."]


def reports(session: Session) -> dict[str, Checkpoint]:
    """Read current reports, including ones restored from an earlier session.

    Args:
      session: Root session holding the reports.

    Returns:
      reports: Latest report by worker label.

    """
    found: dict[str, Checkpoint] = {}
    for record in session.standing():
        if record.kind != "work_checkpoint":
            continue
        match record.payload:
            case {
                "agent": str(label),
                "progress": str(progress),
                "pending": str(pending),
                "user_action": str(action),
            }:
                found[label] = Checkpoint(
                    progress=progress,
                    pending=pending,
                    user_action=action,
                )
            case _:
                continue
    return found


def report(session: Session, label: str, checkpoint: Checkpoint | None) -> None:
    """Replace or withdraw one worker's report; journal only changes.

    Args:
      session: Root session holding the reports.
      label: Agent whose report changed.
      checkpoint: New report, or None to clear it.

    """
    if reports(session).get(label) == checkpoint:
        return
    session.stand(
        f"work_checkpoint:{label}",
        None
        if checkpoint is None
        else UncategorizedRecord(
            kind="work_checkpoint",
            payload={
                "agent": label,
                "progress": checkpoint.progress,
                "pending": checkpoint.pending,
                "user_action": checkpoint.user_action,
            },
        ),
    )


def pending_text(session: Session, *, brief: bool = False) -> str:
    """Show action requests first; the toolbar uses just the highest-priority item.

    Args:
      session: Root session holding the reports.
      brief: Show one short line for the toolbar.

    Returns:
      text: Current reports in display order.

    """
    lines: list[str] = []
    for label, item in sorted(
        reports(session).items(),
        key=lambda pair: (not pair[1].user_action, pair[0]),
    ):
        parts = [f"Action needed: {item.user_action}"] if item.user_action else []
        if item.pending:
            parts.append(f"Waiting: {item.pending}")
        parts.append(item.progress)
        if brief:
            return f"{label}: {' '.join(parts[0].split())[:100]}"
        lines.append(f"{label}: " + "\n  ".join(parts))
    return "\n".join(lines)
