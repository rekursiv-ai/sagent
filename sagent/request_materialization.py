"""Provider-request materialization shared by model-call producers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import dataclasses

from sagent.agent.context import wire_role
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
                # Emit the text-only assistant whenever it carries
                # provider-visible payload and would not collide with a
                # newer assistant turn (role alternation requires
                # assistant/user/assistant/...). The previous gate that
                # required ``next_newer is None`` silently dropped AM text
                # whenever any newer entry followed; loosen it to "newer
                # role is not assistant".
                if (
                    not kept_call_ids
                    and _assistant_has_payload(materialized)
                    and (next_newer is None or wire_role(next_newer) != "assistant")
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
    ``[from <source>]: `` marker is left unchanged. ``startswith`` (not
    substring ``in``) avoids the trap where a body that legitimately
    quotes the marker -- e.g. ``"please write [from bob]: literally"``
    -- gets silently passed through unlabelled. Cross-type coalescing
    is handled by ``_coalesce_adjacent_users`` after labeling: any
    merged user-side entry whose text quotes a marker has already been
    labelled here at the producer, so re-labeling never happens.
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
    """Merge adjacent user-side entries so the wire stays alternation-valid.

    Same-source merges preserve the source type and attribution. Cross-
    source merges -- ``UserMessage`` with ``AgentSendMessage`` or two
    ``AgentSendMessage`` carrying different ``source`` values -- demote
    to ``UserMessage``: the structured ``source`` cannot honestly
    represent two different senders, and the textual ``[from X]: ``
    labels added by :func:`_label_agent_sends` upstream already carry
    per-sender attribution in the merged text.
    """
    deferred = _defer_user_between_tool_pair(messages)
    out: list[ModelContextEvent] = []
    for entry in deferred:
        if (
            isinstance(entry, (AgentSendMessage, UserMessage))
            and out
            and wire_role(out[-1]) == "user"
        ):
            prev = out[-1]
            assert isinstance(prev, (UserMessage, AgentSendMessage))
            text = f"{prev.text}\n\n{entry.text}"
            attachments = (*prev.attachments, *entry.attachments)
            if _same_source(prev, entry):
                out[-1] = dataclasses.replace(prev, text=text, attachments=attachments)
            else:
                # Sources differ: the structured ``source`` field would
                # silently claim one sender owns the other's content.
                # Demote to ``UserMessage``; the in-text ``[from X]: ``
                # labels (already applied upstream) preserve attribution.
                out[-1] = UserMessage(text=text, attachments=attachments)
        else:
            out.append(entry)
    return out


def _defer_user_between_tool_pair(
    messages: Iterable[ModelContextEvent],
) -> list[ModelContextEvent]:
    """Move user-side entries appearing between AM(tool_calls) and its TR.

    Provider APIs reject any user-role turn between an
    ``AssistantMessage`` carrying ``tool_calls`` and the matching
    ``ToolResult`` for those calls. Such interleavings can be produced
    by overrides whose payloads splice cross-source agent traffic
    (``AgentSendMessage``) into a position that breaks the tool pair.
    Defer any such entries to immediately after the matching tool
    results close so the wire ordering stays valid while the user
    content is preserved.
    """
    deferred_buffer: list[UserMessage | AgentSendMessage] = []
    pending: set[str] = set()
    out: list[ModelContextEvent] = []
    for entry in messages:
        if isinstance(entry, AssistantMessage):
            pending = {tc.id for tc in entry.tool_calls}
            out.append(entry)
        elif isinstance(entry, ToolResult):
            out.append(entry)
            pending.discard(entry.call_id)
            if not pending and deferred_buffer:
                out.extend(deferred_buffer)
                deferred_buffer = []
        elif pending and wire_role(entry) == "user":
            assert isinstance(entry, (UserMessage, AgentSendMessage))
            deferred_buffer.append(entry)
        else:
            out.append(entry)
    out.extend(deferred_buffer)
    return out


def _same_source(left: ModelContextEvent, right: ModelContextEvent) -> bool:
    """Return True iff two user-side entries share sender identity.

    ``UserMessage`` represents the human; two are same-source. Two
    ``AgentSendMessage`` are same-source iff their ``source`` values
    match. A ``UserMessage`` paired with an ``AgentSendMessage`` is
    *not* same-source: the human did not author the agent's content
    and the agent did not author the human's. Cross-type or
    cross-source pairs still need merging for wire alternation, but
    the merge path (see :func:`_coalesce_adjacent_users`) demotes them
    to ``UserMessage`` so structured attribution is not falsified.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, AgentSendMessage) and isinstance(right, AgentSendMessage):
        return left.source == right.source
    return True


def _assistant_has_payload(entry: AssistantMessage) -> bool:
    """Return whether an assistant turn has provider-visible payload."""
    return bool(entry.text or entry.thinking_blocks or entry.tool_calls)
