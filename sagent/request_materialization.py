"""Provider-request materialization shared by model-call producers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Final

import dataclasses

from sagent.agent.state import approx_tokens
from sagent.types.model import ModelRequest
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ModelContextEvent,
    ToolResult,
    UserMessage,
    labeled_agent_send_text,
    wire_role,
)


ELIDED_TOOL_RESULT_TAG: Final = "<elided>"

_ELISION_NOTICE: Final = (
    "<elided: {chars:,} chars dropped to fit the request budget."
    " Re-run the tool with a narrower window to see this content.>"
)

_TRUNCATION_NOTICE: Final = (
    "\n<truncated: kept the first {kept:,} of {total:,} chars to fit the"
    " request budget. Re-run the tool from character {kept:,} for the rest.>"
)


def elided_placeholder(original_chars: int) -> str:
    """Render the wire placeholder for an over-budget tool result.

    Says what was dropped and how to get it back. A bare tag left the
    model unable to distinguish "the tool returned nothing" from "the
    result was too large to ship", and the second is recoverable.

    Args:
      original_chars: Length of the content being replaced.

    Returns:
      placeholder: Self-describing replacement text.

    """
    return _ELISION_NOTICE.format(chars=original_chars)


def truncated_result(content: str, budget_tokens: int) -> str:
    """Return ``content``'s head plus a marker naming where to resume.

    The read tool truncates and names an ``offset=``; the persist path writes
    the whole text to disk and prints the path. This layer runs after both and
    used to replace the result outright, discarding content they had already
    bounded -- and its notice told the reader to retry with a smaller window,
    which an agent obeyed by paging one file in 35-line reads.

    Args:
      content: Full result text.
      budget_tokens: Tokens this result may occupy, marker included.

    Returns:
      truncated: Head of ``content`` plus a resume marker, or ``""`` when the
          budget cannot even hold the marker.

    """
    if approx_tokens(_render_truncated(content, 0)) > budget_tokens:
        return ""
    # Bisect on the estimator rather than dividing by a chars-per-token ratio:
    # ``approx_tokens`` is the ACTIVE model's tokenizer when one is in context,
    # so no fixed ratio is correct, and the marker itself grows with the kept
    # count. Monotone in ``kept``, so bisection lands on the largest fit.
    low, high = 0, len(content)
    while low < high:
        mid = (low + high + 1) // 2
        if approx_tokens(_render_truncated(content, mid)) <= budget_tokens:
            low = mid
        else:
            high = mid - 1
    return _render_truncated(content, low) if low else ""


def _render_truncated(content: str, kept: int) -> str:
    """Return ``content``'s first ``kept`` chars plus the resume marker."""
    return content[:kept] + _TRUNCATION_NOTICE.format(kept=kept, total=len(content))


def materialize_request(
    request: ModelRequest,
    *,
    tool_result_budget_tokens: int = 0,
) -> ModelRequest:
    """Return ``request`` with provider-visible messages budgeted.

    Args:
      request: Fully-built provider request.
      tool_result_budget_tokens: Aggregate TOKEN budget for full
          ``ToolResult.content`` values. ``0`` disables tool-result elision.

    Returns:
      materialized: Request with the same knobs and budgeted messages.

    """
    return dataclasses.replace(
        request,
        messages=materialize_messages(
            request.messages,
            tool_result_budget_tokens=tool_result_budget_tokens,
        ),
    )


def materialize_messages(
    messages: Sequence[ModelContextEvent],
    *,
    tool_result_budget_tokens: int = 0,
) -> list[ModelContextEvent]:
    """Return provider-visible messages with full results under budget.

    Args:
      messages: Resolved provider-facing history.
      tool_result_budget_tokens: Aggregate TOKEN budget for full
          ``ToolResult.content`` values. ``0`` disables tool-result elision.

    Returns:
      materialized: Message list with excess tool results replaced by
          non-empty placeholders while preserving message ordering and ids.

    """
    if tool_result_budget_tokens <= 0:
        return _coalesce_adjacent_users(_label_agent_sends(messages))
    labelled = list(_label_agent_sends(messages))
    used = 0
    keep_calls: set[str] = set()
    out_reversed: list[ModelContextEvent] = []
    older = _older_tool_result_counts(labelled)
    tag_cost = approx_tokens(ELIDED_TOOL_RESULT_TAG)
    for idx in range(len(labelled) - 1, -1, -1):
        entry = labelled[idx]
        if isinstance(entry, ToolResult):
            content = entry.content
            cost = approx_tokens(content)
            if used + cost <= tool_result_budget_tokens:
                used += cost
                keep_calls.add(entry.call_id)
                out_reversed.append(entry)
            elif used + tag_cost <= tool_result_budget_tokens:
                # Prefer the HEAD of the content plus a resume marker; fall
                # back to a placeholder only when the remaining budget cannot
                # hold even that. Dropping outright discarded text the read
                # tool had already truncated for exactly this budget, and the
                # placeholder's "retry with a narrower window" then drove the
                # reader into paging the same file dozens of times.
                #
                # Reserve one NOTICE per older result before truncating --
                # never a bare tag. Elision cost ~20 tokens and was
                # self-limiting, so every older turn survived as a placeholder;
                # truncation is not, and giving the newest result the whole
                # remaining budget starved the turns behind it (6 turns
                # collapsed to 1). But reserving the 8-byte TAG while falling
                # back to the longer NOTICE under-buys: the reservation cannot
                # pay for what it promises, so at batch scale the oldest
                # entries -- carrying the largest reservation -- collapsed to
                # the tag anyway. On a real 21-file batch that was 11 results
                # reaching the wire as 8 bytes, a 10,576-char file among them
                # while a 49,565-char file passed whole.
                #
                # The floor is the self-describing notice: a bare tag names
                # neither the size dropped nor a way back, and reads to the
                # model like a tool that returned nothing. The tag survives
                # only as a genuine last resort -- a budget too small to hold
                # one notice, where the alternative is dropping the result and
                # unpairing its ``tool_use``.
                notice = elided_placeholder(len(content))
                notice_cost = approx_tokens(notice)
                reserved = min(
                    older[idx] * notice_cost,
                    max(tool_result_budget_tokens - used - notice_cost, 0),
                )
                share = tool_result_budget_tokens - used - reserved
                placeholder = truncated_result(content, share) if share > 0 else ""
                if not placeholder or approx_tokens(placeholder) > share:
                    placeholder = notice
                if used + approx_tokens(placeholder) > tool_result_budget_tokens:
                    placeholder = ELIDED_TOOL_RESULT_TAG
                used += approx_tokens(placeholder)
                keep_calls.add(entry.call_id)
                out_reversed.append(dataclasses.replace(entry, content=placeholder))
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


def _older_tool_result_counts(messages: Sequence[ModelContextEvent]) -> list[int]:
    """Return, per index, how many ``ToolResult`` entries precede it.

    The budget walk runs newest-first, so what a result may spend is bounded by
    what the results BEHIND it still need. This is that count, precomputed once
    rather than rescanned per entry (the walk is already O(n)).
    """
    counts = [0] * len(messages)
    seen = 0
    for idx, entry in enumerate(messages):
        counts[idx] = seen
        if isinstance(entry, ToolResult):
            seen += 1
    return counts


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
    -- gets silently passed through unlabelled. Coalescing runs after
    labeling: a cross-source merge demotes to a ``UserMessage`` (skipped
    here) carrying per-segment labels, so it is never re-labeled; a
    same-source ``AgentSendMessage`` merge keeps its type and un-labeled
    text, and this prepends one outer label to the whole -- correct, since
    one source authored every segment.
    """
    for entry in messages:
        if isinstance(entry, AgentSendMessage):
            labeled = labeled_agent_send_text(entry)
            yield (
                entry
                if labeled == entry.text
                else dataclasses.replace(entry, text=labeled)
            )
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
            # ``pending`` is "all still-unanswered tool calls", NOT "this AM's
            # calls". Accumulate -- a later text-only AM (or one opening fresh
            # calls) must not clear an earlier call that has no ToolResult yet,
            # or an interleaved user turn would ship inside the still-open pair.
            pending |= {tc.id for tc in entry.tool_calls}
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
    and the agent did not author the human's. :func:`_coalesce_adjacent_users`
    still merges cross-type / cross-source pairs (both are wire-``user`` and
    cannot reach the provider as adjacent turns) but *demotes* them to a
    plain ``UserMessage`` so the structured ``source`` never falsely claims
    one sender authored the other's text; same-source pairs preserve type.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, AgentSendMessage) and isinstance(right, AgentSendMessage):
        return left.source == right.source
    return True


def _assistant_has_payload(entry: AssistantMessage) -> bool:
    """Return whether an assistant turn has provider-visible payload."""
    return bool(entry.text or entry.thinking_blocks or entry.tool_calls)
