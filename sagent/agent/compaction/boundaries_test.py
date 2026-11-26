"""Tests for boundaries: where a history may be cut."""

from __future__ import annotations

from sagent.agent.compaction.boundaries import safe_boundaries
from sagent.journal.types import ToolMessage, UserMessage
from sagent.model.scripted import call, reply


def test_the_boundary_before_the_first_entry_is_safe() -> None:
    assert safe_boundaries([]) == [True]


def test_no_boundary_falls_between_a_call_and_its_result() -> None:
    history = [
        UserMessage(content="go"),
        reply(calls=[call("Read", "a"), call("Read", "b")]),
        ToolMessage(call_id="a", state="done"),
        ToolMessage(call_id="b", state="done"),
        reply("done"),
    ]

    assert safe_boundaries(history) == [True, True, False, False, True, True]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
