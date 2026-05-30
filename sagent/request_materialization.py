"""Provider-request materialization shared by model-call producers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import dataclasses

from sagent.types.model import ModelRequest
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ModelContextEvent,
    ToolResult,
    UserMessage,
)


ELIDED_TOOL_RESULT_TAG = "<elided>"


def materialize_request(
    request: ModelRequest,
    *,
    tool_result_budget_chars: int = 0,
) -> ModelRequest:
    """Return ``request`` with provider-visible messages budgeted.

    Args:
      request: Fully-built provider request.
      tool_result_budget_chars: Aggregate character budget for full
          ``ToolResult.content`` values. ``0`` disables tool-result elision.

    Returns:
      materialized: Request with the same knobs and budgeted messages.

    """
    return dataclasses.replace(
        request,
        messages=materialize_messages(
            request.messages,
            tool_result_budget_chars=tool_result_budget_chars,
        ),
    )


def materialize_messages(
    messages: Sequence[ModelContextEvent],
    *,
    tool_result_budget_chars: int = 0,
) -> list[ModelContextEvent]:
    """Return provider-visible messages with full results under budget.

    Args:
      messages: Resolved provider-facing history.
      tool_result_budget_chars: Aggregate character budget for full
          ``ToolResult.content`` values. ``0`` disables tool-result elision.

    Returns:
      materialized: Message list with excess tool results replaced by
          non-empty placeholders while preserving message ordering and ids.

    """
    if tool_result_budget_chars <= 0:
        return _coalesce_adjacent_users(_label_agent_sends(messages))
    labelled = list(_label_agent_sends(messages))
    used = 0
    keep_calls: set[str] = set()
    out_reversed: list[ModelContextEvent] = []
    for idx in range(len(labelled) - 1, -1, -1):
        entry = labelled[idx]
        if isinstance(entry, ToolResult):
            content = entry.content
            if used + len(content) <= tool_result_budget_chars:
                used += len(content)
                keep_calls.add(entry.call_id)
                out_reversed.append(entry)
            elif used + len(ELIDED_TOOL_RESULT_TAG) <= tool_result_budget_chars:
                used += len(ELIDED_TOOL_RESULT_TAG)
                keep_calls.add(entry.call_id)
                out_reversed.append(
                    dataclasses.replace(entry, content=ELIDED_TOOL_RESULT_TAG)
                )
        elif isinstance(entry, AssistantMessage) and entry.tool_calls:
            call_ids = {tc.id for tc in entry.tool_calls}
            if call_ids <= keep_calls:
                out_reversed.append(entry)
            else:
                out_reversed = [
                    newer
                    for newer in out_reversed
                    if not (isinstance(newer, ToolResult) and newer.call_id in call_ids)
                ]
                kept_call_ids = call_ids & keep_calls
                keep_calls.difference_update(call_ids)
                materialized = dataclasses.replace(entry, tool_calls=())
                next_newer = out_reversed[-1] if out_reversed else None
                if (
                    not kept_call_ids
                    and idx > 0
                    and _assistant_has_payload(materialized)
                    and next_newer is None
                ):
                    out_reversed.append(materialized)
        else:
            out_reversed.append(entry)
    return _coalesce_adjacent_users(reversed(out_reversed))


def _label_agent_sends(
    messages: Iterable[ModelContextEvent],
) -> Iterable[ModelContextEvent]:
    """Prepend ``[from <source>]: `` to each ``AgentSendMessage`` text.

    Applies the sender label before coalescing so that adjacent messages
    from different agents retain per-sender attribution after merging.
    Idempotent: a text already starting with this entry's own
    ``[from <source>]: `` prefix is left unchanged. Without this guard,
    re-materializing previously-materialized history (e.g. after a
    compactor rewrite) would compound prefixes into
    ``[from X]: [from X]: ...``.
    """
    for entry in messages:
        if isinstance(entry, AgentSendMessage):
            prefix = f"[from {entry.source}]: "
            if entry.text.startswith(prefix):
                yield entry
            else:
                yield dataclasses.replace(entry, text=f"{prefix}{entry.text}")
        else:
            yield entry


def _coalesce_adjacent_users(
    messages: Iterable[ModelContextEvent],
) -> list[ModelContextEvent]:
    out: list[ModelContextEvent] = []
    for entry in messages:
        if (
            isinstance(entry, (AgentSendMessage, UserMessage))
            and out
            and type(out[-1]) is type(entry)
            and _same_source(out[-1], entry)
        ):
            prev = out[-1]
            if isinstance(entry, AgentSendMessage):
                assert isinstance(prev, AgentSendMessage)
            else:
                assert isinstance(prev, UserMessage)
            out[-1] = dataclasses.replace(
                prev,
                text=f"{prev.text}\n\n{entry.text}",
                attachments=(*prev.attachments, *entry.attachments),
            )
        else:
            out.append(entry)
    return out


def _same_source(left: ModelContextEvent, right: ModelContextEvent) -> bool:
    """Return True iff two user-side entries share their sender identity.

    ``UserMessage`` carries no sender field (all are the human), so any
    two are same-source. ``AgentSendMessage`` carries ``source``; merging
    across different sources would silently re-attribute one agent's
    content to another.
    """
    if isinstance(left, AgentSendMessage) and isinstance(right, AgentSendMessage):
        return left.source == right.source
    return True


def _assistant_has_payload(entry: AssistantMessage) -> bool:
    """Return whether an assistant turn has provider-visible payload."""
    return bool(entry.text or entry.thinking_blocks or entry.tool_calls)
