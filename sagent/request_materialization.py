"""Provider-request materialization shared by model-call producers."""

from __future__ import annotations

from collections.abc import Sequence

import dataclasses

from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolResult,
)
from sagent.types.model import ModelRequest


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
    messages: Sequence[HistoryEntry],
    *,
    tool_result_budget_chars: int = 0,
) -> list[HistoryEntry]:
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
        return list(messages)
    used = 0
    keep_calls: set[str] = set()
    out_reversed: list[HistoryEntry] = []
    for entry in reversed(messages):
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
            calls = tuple(tc for tc in entry.tool_calls if tc.id in keep_calls)
            if calls or entry.text or entry.thinking_blocks:
                out_reversed.append(dataclasses.replace(entry, tool_calls=calls))
        else:
            out_reversed.append(entry)
    return list(reversed(out_reversed))
