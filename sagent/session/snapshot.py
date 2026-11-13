"""UI-neutral snapshots of native Sagent session records.

The projector reads the native ``Agent.session`` window and returns a stable
data shape for transports and presenters.  It does not own the journal, alter
records, or decide how a frontend renders them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.runtime import types
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    FileEditResult,
    FileWriteResult,
    Thinking,
    ToolCall,
    ToolResult,
    TurnContext,
    UserMessage,
)


if TYPE_CHECKING:
    from sagent.agent.agent import Agent


__all__ = ["SessionSnapshotProjector"]


class SessionSnapshotProjector:
    """Project one native agent session without taking ownership of it."""

    def project(self, agent: Agent) -> dict[str, object]:
        """Return the persisted conversation window for ``agent``."""
        items: list[dict[str, object]] = []
        calls: dict[str, dict[str, object]] = {}
        effective_model: str | None = None
        turn = 0

        for index, record in enumerate(agent.session):
            if isinstance(record, TurnContext):
                if record.model is not None:
                    effective_model = record.model
                continue

            if isinstance(record, (UserMessage, AgentToAgentMessage)):
                turn += 1
                item: dict[str, object] = {
                    "id": f"record-{index}",
                    "type": "agent"
                    if isinstance(record, AgentToAgentMessage)
                    else "user",
                    "turnId": f"turn-{turn}",
                    "text": record.content or "",
                    **_record_metadata(record),
                }
                if effective_model is not None:
                    item["model"] = effective_model
                if isinstance(record, AgentToAgentMessage):
                    item.update(
                        {"sender": record.sender, "recipient": record.recipient},
                    )
                items.append(item)
                continue

            if isinstance(record, AssistantMessage):
                if (text := record.content or "") and text.strip():
                    items.append(
                        {
                            "id": f"record-{index}",
                            "type": "assistant",
                            "turnId": f"turn-{turn}",
                            "text": text,
                            **_record_metadata(record),
                        },
                    )
                continue

            if isinstance(record, Thinking):
                if (text := record.content or "") and text.strip():
                    items.append(
                        {
                            "id": f"record-{index}",
                            "type": "thinking",
                            "turnId": f"turn-{turn}",
                            "text": text,
                            **_record_metadata(record),
                        },
                    )
                continue

            if isinstance(record, ToolCall):
                if record.call_id.endswith(types.Markers().arrival):
                    continue
                tool = agent.tools.get(record.name)
                summary = (
                    str(record.extra.get("title") or "")
                    or (tool.summary(record.arguments) if tool else None)
                    or str(record.arguments.get("toolSummary") or "")
                    or str(record.arguments.get("CommandLine") or "")
                    or record.name
                )
                item = {
                    "id": f"tool-{record.call_id}",
                    "type": "tool",
                    "turnId": f"turn-{turn}",
                    "callId": record.call_id,
                    "name": record.name,
                    "summary": summary,
                    "arguments": dict(record.arguments),
                    "status": "running",
                    **_record_metadata(record),
                }
                items.append(item)
                calls[record.call_id] = item
                continue

            if isinstance(record, ToolResult):
                call_id = record.call_id.removesuffix(types.Markers().arrival)
                item = calls.get(call_id)
                if item is None:
                    # Match the native TUI replay rule: successful orphaned
                    # results are hidden, while failures remain visible.
                    if record.is_error is not True:
                        continue
                    item = {
                        "id": f"tool-{call_id}",
                        "type": "tool",
                        "turnId": f"turn-{turn}",
                        "callId": call_id,
                        "name": "unknown",
                        "summary": "Tool result",
                        "status": "failed",
                    }
                    items.append(item)
                    calls[call_id] = item
                item.update(tool_result_payload(record))

        return {"version": 1, "items": items}


def _record_metadata(record: object) -> dict[str, object]:
    """Return persisted metadata without imposing frontend formatting."""
    metadata: dict[str, object] = {}
    timestamp = getattr(record, "timestamp", None)
    if timestamp is not None:
        metadata["timestamp"] = timestamp
    context_id = getattr(record, "context_id", None)
    if context_id is not None:
        metadata["contextId"] = context_id
    return metadata


def tool_result_payload(result: ToolResult) -> dict[str, object]:
    """Return the snapshot fields for one native tool result."""
    call_id = result.call_id.removesuffix(types.Markers().arrival)
    path = result.path if isinstance(result, (FileEditResult, FileWriteResult)) else None
    status = (
        "detached" if result.stub == "pending" else
        "cancelled" if result.stub == "cancelled" else
        "failed" if result.is_error is True else "completed"
    )
    return {
        "callId": call_id,
        "resultCallId": result.call_id,
        "status": status,
        "text": result.text(),
        "isError": result.is_error,
        **({} if result.stub is None else {"stub": result.stub}),
        **({} if path is None else {"path": path}),
        **_file_change_payload(result),
    }


def _file_change_payload(result: ToolResult) -> dict[str, object]:
    if isinstance(result, FileEditResult):
        return {
            "fileChange": {
                "kind": "edit",
                "edits": [
                    {
                        "before": splice.before,
                        "after": splice.after,
                        "start": splice.start,
                        "count": splice.count,
                    }
                    for splice in result.edits
                ],
            },
        }
    if isinstance(result, FileWriteResult):
        return {"fileChange": {"kind": "write", "content": result.content}}
    return {}
