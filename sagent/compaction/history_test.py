"""Tests for ``compaction.history``: shared history mutators and estimators."""

from __future__ import annotations

from sagent.compaction import scrunch, summary
from sagent.compaction.history import (
    append_to_first_user,
    entry_chars,
)
from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
    UserMessage,
)


def test_entry_chars_shared_across_compaction_modules() -> None:
    # ``entry_chars`` is the canonical size estimator; ``summary`` and
    # ``scrunch`` must reuse it, not keep private copies that silently drift
    # from one another (one rule everywhere).
    assert summary.entry_chars is entry_chars
    assert scrunch.entry_chars is entry_chars


def test_entry_chars_counts_text_and_tool_calls() -> None:
    assert entry_chars(UserMessage(text="hello")) == 5
    assert entry_chars(AssistantMessage(text="hi")) == 2


def test_append_to_first_user_inserts_when_absent() -> None:
    events: list[ModelContextEvent] = []
    append_to_first_user(events, "seed")
    first = events[0]
    assert isinstance(first, UserMessage)
    assert first.text == "seed"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
