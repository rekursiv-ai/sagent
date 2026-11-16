"""UI-neutral envelopes for the generic Sagent daemon protocol.

The values inside these envelopes remain native Sagent objects until the
daemon writer passes the complete message to :func:`treekle.json_dumps`.
This module only adds protocol metadata; it does not duplicate treekle's
serialization rules.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from sagent.runtime import types
from sagent.server.protocol import NATIVE_PROTOCOL
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    Thinking,
    ToolCall,
    ToolResult,
    UserMessage,
    record_classes,
)


if TYPE_CHECKING:
    from sagent.runtime.types import RuntimeEvent
    from sagent.session.types import SessionRecord


type ToolSummary = Callable[[ToolCall], str | None]
_SESSION_RECORD_TYPES = frozenset(record_classes().values())


def runtime_event_message(
    event: RuntimeEvent,
    *,
    agent_label: str,
    session_id: str | None = None,
    sequence: int | None = None,
    tool_summary: ToolSummary | None = None,
) -> dict[str, object]:
    """Build a lossless envelope for one native runtime event."""
    message = native_event_message(
        type(event).__name__,
        event,
        agent_label=agent_label,
        session_id=session_id,
        sequence=sequence,
    )
    message["category"] = native_category(event)
    if metadata := native_metadata(event, tool_summary=tool_summary):
        message["native"] = metadata
    if isinstance(event, types.ModelResponseComplete):
        message["records"] = [
            session_record_message(record, tool_summary=tool_summary)
            for record in event.response.records
        ]
    return message


def session_notification_message(
    event_type: str,
    value: object,
    *,
    agent_label: str | None = None,
    session_id: str | None = None,
    sequence: int | None = None,
) -> dict[str, object]:
    """Build an envelope for a native daemon notification."""
    return native_event_message(
        event_type,
        value,
        agent_label=agent_label,
        session_id=session_id,
        sequence=sequence,
    )


def native_event_message(
    event_type: str,
    value: object,
    *,
    agent_label: str | None = None,
    session_id: str | None = None,
    sequence: int | None = None,
) -> dict[str, object]:
    """Build the shared envelope used by runtime and daemon notifications."""
    message: dict[str, object] = {
        "protocol": NATIVE_PROTOCOL,
        "eventType": event_type,
        "value": value,
    }
    if agent_label is not None:
        message["agentLabel"] = agent_label
    if session_id is not None:
        message["sessionId"] = session_id
    if sequence is not None:
        message["sequence"] = sequence
    return message


def session_record_message(
    record: SessionRecord,
    *,
    tool_summary: ToolSummary | None = None,
) -> dict[str, object]:
    """Build a lossless envelope for one native journal record."""
    message: dict[str, object] = {
        "recordType": type(record).__name__,
        "category": native_category(record),
        "value": record,
    }
    if metadata := native_metadata(record, tool_summary=tool_summary):
        message["native"] = metadata
    return message


def native_category(value: object) -> str:
    """Classify a native value without assigning frontend presentation."""
    if isinstance(value, UserMessage):
        return "user-message"
    if isinstance(value, AgentToAgentMessage):
        return "agent-message"
    if isinstance(value, AssistantMessage):
        return "assistant-message"
    if isinstance(value, Thinking):
        return "thinking"
    if isinstance(value, ToolCall | types.ToolCallStarted):
        return "tool-call"
    if isinstance(value, ToolResult | types.ToolResultStreamed):
        return "tool-result"
    if isinstance(value, types.ModelResponsePartial):
        return "assistant-stream"
    if isinstance(value, types.ModelResponseThinking):
        return "thinking-stream"
    if isinstance(value, types.ModelResponseComplete):
        return "model-response"
    if isinstance(
        value,
        (
            types.ModelCallStarted,
            types.ModelResponseCancelled,
            types.ModelResponseError,
            types.ModelCallSuspended,
            types.ModelIdle,
            types.AgentIdle,
        ),
    ):
        return "lifecycle"
    return "metadata" if _is_session_record(value) else "runtime"


def native_metadata(
    value: object,
    *,
    tool_summary: ToolSummary | None = None,
) -> dict[str, object]:
    """Return behavior computed by native Sagent types and tools.

    These fields keep clients from reproducing Python subclass behavior while
    leaving visual grouping and status decisions to the client.
    """
    if isinstance(value, types.ToolCallStarted):
        return _call_metadata(
            value.call,
            summary=value.title or _summary(value.call, tool_summary),
        )
    if isinstance(value, ToolCall):
        return _call_metadata(value, summary=_summary(value, tool_summary))
    if isinstance(value, types.ToolResultStreamed):
        return _result_metadata(value.result)
    if isinstance(value, ToolResult):
        return _result_metadata(value)
    return {}


def _summary(call: ToolCall, resolve: ToolSummary | None) -> str | None:
    explicit = call.extra.get("title")
    if isinstance(explicit, str) and explicit:
        return explicit
    if resolve is None:
        return None
    try:
        return resolve(call)
    except Exception:  # noqa: BLE001 -- metadata must not suppress the raw value.
        return None


def _call_metadata(
    call: ToolCall,
    *,
    summary: str | None,
) -> dict[str, object]:
    """Return native call identity without making clients parse runtime markers."""
    markers = types.Markers()
    arrival = call.call_id.endswith(markers.arrival)
    return {
        "callId": call.call_id.removesuffix(markers.arrival),
        **({"arrival": True} if arrival else {}),
        **({"summary": summary} if summary is not None else {}),
    }


def _result_metadata(result: ToolResult) -> dict[str, object]:
    markers = types.Markers()
    metadata: dict[str, object] = {
        "callId": result.call_id.removesuffix(markers.arrival),
        **({"arrival": True} if result.call_id.endswith(markers.arrival) else {}),
    }
    try:
        metadata["text"] = result.text()
    except Exception:  # noqa: BLE001 -- metadata must not suppress the raw value.
        pass
    return metadata


def _is_session_record(value: object) -> bool:
    """Return whether ``value`` is one of the native persisted record types."""
    # SessionRecord is a runtime union, so use the registry instead of trying
    # to use it in isinstance(). Attachments are registry members but are never
    # top-level records.
    return type(value) in _SESSION_RECORD_TYPES
