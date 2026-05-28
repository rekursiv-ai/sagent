"""Tests for ``types.runtime`` event documentation."""

from __future__ import annotations

import time

from sagent.types.runtime import Recompact, UserMessage


def test_recompact_event_docstring_describes_compact_alias() -> None:
    assert Recompact.__doc__ is not None
    assert "alias" in Recompact.__doc__.lower()
    assert "/compact" in Recompact.__doc__
    assert "reload" not in Recompact.__doc__.lower()


def test_session_message_timestamp_is_wall_clock() -> None:
    before = time.time()
    entry = UserMessage(text="hi")
    after = time.time()
    assert before <= entry.timestamp <= after


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
