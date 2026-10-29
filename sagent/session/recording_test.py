"""Tests for the recording journal: the window since the last clear, and no file."""

from __future__ import annotations

from sagent.session.recording import RecordingJournal
from sagent.session.types import AssistantMessage, ContextClear, UserMessage


def test_it_has_no_file() -> None:
    assert RecordingJournal().path is None


def test_it_yields_what_was_appended_in_order() -> None:
    journal = RecordingJournal()
    journal.append(UserMessage(content="a"), AssistantMessage(content="b"))
    journal.append(UserMessage(content="c"))
    assert list(journal) == [
        UserMessage(content="a"),
        AssistantMessage(content="b"),
        UserMessage(content="c"),
    ]


def test_a_clear_starts_the_window_it_yields() -> None:
    journal = RecordingJournal()
    clear = ContextClear(summary="s")
    journal.append(UserMessage(content="old"), clear, UserMessage(content="new"))
    assert list(journal) == [clear, UserMessage(content="new")]


def test_its_history_is_every_window_from_the_first() -> None:
    journal = RecordingJournal()
    clear = ContextClear(summary="s")
    journal.append(UserMessage(content="old"), clear, UserMessage(content="new"))
    seen = journal.history()
    journal.append(UserMessage(content="later"))
    assert list(seen) == [UserMessage(content="old"), clear, UserMessage(content="new")]


def test_iteration_is_a_snapshot() -> None:
    journal = RecordingJournal()
    journal.append(UserMessage(content="a"))
    seen = iter(journal)
    journal.append(UserMessage(content="b"))
    assert list(seen) == [UserMessage(content="a")]


def test_flush_keeps_everything() -> None:
    journal = RecordingJournal()
    journal.append(UserMessage(content="a"))
    journal.flush()
    assert list(journal) == [UserMessage(content="a")]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
