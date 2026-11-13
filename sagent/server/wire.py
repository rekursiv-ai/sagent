"""UI-neutral envelopes for the generic Sagent daemon protocol.

The values inside these envelopes remain native Sagent objects until the
daemon writer passes the complete message to :func:`treekle.json_dumps`.
This module only adds protocol metadata; it does not duplicate treekle's
serialization rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.server.protocol import NATIVE_PROTOCOL


if TYPE_CHECKING:
    from sagent.runtime.types import RuntimeEvent
    from sagent.session.types import SessionRecord


def runtime_event_message(
    event: RuntimeEvent,
    *,
    agent_label: str,
    session_id: str | None = None,
    sequence: int | None = None,
) -> dict[str, object]:
    """Build an envelope for one native runtime event."""
    return native_event_message(
        type(event).__name__,
        event,
        agent_label=agent_label,
        session_id=session_id,
        sequence=sequence,
    )


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


def session_record_message(record: SessionRecord) -> dict[str, object]:
    """Build an envelope for one native journal record."""
    return {"recordType": type(record).__name__, "value": record}
