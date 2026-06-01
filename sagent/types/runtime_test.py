"""Tests for ``types.runtime`` event documentation."""

from __future__ import annotations

import threading
import time

import pytest

from sagent.types import runtime as types_runtime
from sagent.types.runtime import (
    AssistantMessage,
    Recompact,
    ToolCall,
    UserMessage,
    reset_id_counter,
)


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


def test_assistant_message_rejects_duplicate_tool_call_ids() -> None:
    """A46: AssistantMessage must reject duplicate ToolCall ids at construction.

    Duplicate ids collide in ``running_tools[id]`` and collapse the
    cohort set, leaking tasks. Fail loudly at the type boundary.
    """
    with pytest.raises(ValueError, match="duplicate tool_call id"):
        AssistantMessage(
            tool_calls=(
                ToolCall(id="x", name="a", args={}),
                ToolCall(id="x", name="b", args={}),
            )
        )


def test_assistant_message_accepts_unique_tool_call_ids() -> None:
    """Sanity: distinct ids construct without error."""
    msg = AssistantMessage(
        tool_calls=(
            ToolCall(id="a", name="t1", args={}),
            ToolCall(id="b", name="t2", args={}),
        )
    )
    assert tuple(tc.id for tc in msg.tool_calls) == ("a", "b")


def test_reset_id_counter_concurrent_threads_stay_monotonic() -> None:
    """A20: ``reset_id_counter`` is thread-safe under concurrent calls.

    Without the internal lock, two threads can interleave the
    ``next(_id_counter)`` peek with the chain/count replacement and
    produce non-monotonic ids. The lock serialises peek-and-replace so
    the post-reset id is strictly greater than every pre-reset id.
    """
    reset_id_counter(10_000)
    pre = UserMessage(text="pre").id
    start = threading.Event()
    barrier = threading.Barrier(8)

    def _resetter(target: int) -> None:
        start.wait()
        barrier.wait()
        reset_id_counter(target)

    threads = [
        threading.Thread(target=_resetter, args=(pre + 1 + i,)) for i in range(8)
    ]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    post = UserMessage(text="post").id
    assert post > pre, f"reset_id_counter raced; pre={pre} post={post}"


def test_id_counter_lock_module_attribute_exists() -> None:
    """Locks the no-regress contract: the threading lock must exist."""
    assert isinstance(types_runtime._id_counter_lock, type(threading.Lock()))


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
