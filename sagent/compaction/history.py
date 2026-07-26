"""Pure history-mutation helpers used by compaction strategies.

These helpers operate on raw ``list[ModelContextEvent]`` and have no
coupling to ``Agent``, ``ToolState``, or any runtime state. The
Agent-coupled enrichment pipeline lives in
:mod:`sagent.agent.compaction`; it imports from here.
"""

from __future__ import annotations

from collections.abc import Sequence

import dataclasses

from sagent.lib.token_count import entry_tokens
from sagent.types.model import Model
from sagent.types.runtime import (
    AgentSendMessage,
    ModelContextEvent,
    UserMessage,
)


__all__ = [
    "MAX_CONSECUTIVE_COMPACT_FAILURES",
    "append_to_first_user",
    "estimate_entry_tokens",
]


MAX_CONSECUTIVE_COMPACT_FAILURES = (
    3  # config-globals: ignore -- circuit-breaker retry count (runtime knob)
)
"""Auto-compaction circuit breaker.

After this many consecutive auto-compact failures, ``compact_if_needed``
short-circuits (returns ``False``) without invoking the compactor. The
caller surfaces the underlying error rather than retrying a broken
compactor indefinitely. Reset on any successful compaction.
"""


def append_to_first_user(history: list[ModelContextEvent], text: str) -> None:
    """Append ``text`` to the first user-role message, or insert one.

    The compactor and its post-enrich steps inject context (reattached
    files, background-task status, skill bodies) ahead of the prompt.
    The simplest place is the first user-role message -- ``UserMessage``
    or ``AgentSendMessage`` (both are user-role on the wire). Appending
    to whichever comes first keeps the injected context after that
    message rather than prepended before it, which would reorder the
    conversation. If no user-role message exists yet (e.g. the compactor
    returned an assistant-led summary), insert a fresh one at position 0.

    Args:
      history: History to mutate in place.
      text: Content to append (or seed a new ``UserMessage`` with).

    """
    for j, entry in enumerate(history):
        if isinstance(entry, (UserMessage, AgentSendMessage)):
            joined = f"{entry.text}\n\n{text}" if entry.text else text
            history[j] = dataclasses.replace(entry, text=joined)
            return
    history.insert(0, UserMessage(text=text))


def estimate_entry_tokens(model: Model, entries: Sequence[ModelContextEvent]) -> int:
    """Model-derived token estimate of ``entries`` across every wire surface.

    Delegates to :func:`token_count.entry_tokens` -- the single per-entry
    estimator the request builder uses -- so compaction sizing
    (``token_before``/``token_after``, scrunch partitions, group drops)
    can never drift from the request the provider actually receives.
    Counts tool-call id/name/args JSON, thinking blocks, and attachments,
    not just bare text.
    """
    return sum(entry_tokens(e, model) for e in entries)
