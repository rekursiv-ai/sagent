"""Pure history-mutation helpers used by compaction strategies.

These helpers operate on raw ``list[ModelContextEvent]`` and have no
coupling to ``Agent``, ``ToolState``, or any runtime state. The
Agent-coupled enrichment pipeline lives in
:mod:`sagent.agent.compaction`; it imports from here.
"""

from __future__ import annotations

import dataclasses

from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ModelContextEvent,
    UserMessage,
)


__all__ = [
    "MAX_CONSECUTIVE_COMPACT_FAILURES",
    "append_to_first_user",
    "entry_chars",
]


MAX_CONSECUTIVE_COMPACT_FAILURES = 3
"""Auto-compaction circuit breaker.

After this many consecutive auto-compact failures, ``compact_if_needed``
short-circuits (returns ``False``) without invoking the compactor. The
caller surfaces the underlying error rather than retrying a broken
compactor indefinitely. Reset on any successful compaction.
"""


def append_to_first_user(history: list[ModelContextEvent], text: str) -> None:
    """Append ``text`` to the first ``UserMessage`` in ``history``, or insert one.

    The compactor and its post-enrich steps inject context (reattached
    files, background-task status, skill bodies) ahead of the prompt.
    The simplest place is the first user message; if none exists yet
    (e.g. compactor returned an assistant-led summary), insert a fresh
    one at position 0.

    Args:
      history: History to mutate in place.
      text: Content to append (or seed a new ``UserMessage`` with).

    """
    for j, entry in enumerate(history):
        if isinstance(entry, UserMessage):
            joined = f"{entry.text}\n\n{text}" if entry.text else text
            history[j] = dataclasses.replace(entry, text=joined)
            return
    history.insert(0, UserMessage(text=text))


def entry_chars(entry: ModelContextEvent) -> int:
    """Approximate character count of an entry's payload.

    The shared size estimator for compaction planners and producers, so a
    partition's pre/post estimates stay aligned across modules.
    """
    if isinstance(entry, (AgentSendMessage, UserMessage)):
        return len(entry.text)
    if isinstance(entry, AssistantMessage):
        n = len(entry.text)
        for tc in entry.tool_calls:
            n += len(tc.name) + sum(len(str(v)) for v in tc.args.values())
        return n
    return len(entry.content)
