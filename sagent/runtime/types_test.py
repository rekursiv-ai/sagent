"""Tests for the runtime event vocabulary."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import math
import re

import pytest

from sagent.model.cost import TokenCost, TokenCount
from sagent.model.exceptions import RateLimitedError, RetryableError
from sagent.model.types import Response
from sagent.runtime.tape import ContextSplice, TapeRef
from sagent.runtime.types import (
    CompactComplete,
    DetachedResult,
    DetachSchedule,
    Markers,
    ModelCallSuspended,
    ModelContextEvent,
    WakeSchedule,
    labeled_agent_send_text,
    wire_role,
)
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    SystemMessage,
    ToolResult,
    TurnContext,
    UncategorizedToolResult,
    UserMessage,
)


_CWD: Final[Path] = Path(__file__).resolve().parent


def test_whatever_watches_a_call_fail_decides_about_a_suspended_one_too() -> None:
    """A suspension ends a call as a failure does, but not the turn.

    An observer that hears only the failure leaves a waiting agent looking
    busy, or its parent untold. ``Agent`` waits a suspension out, as its
    ``persistent_retry`` promises.
    """
    package = _CWD.parent
    unheard = {
        str(path.relative_to(package))
        for path in package.rglob("*.py")
        if not path.name.endswith("_test.py")
        and "ModelResponseError" in (text := path.read_text())
        and "ModelCallSuspended" not in text
    }
    assert unheard == {"agent/agent.py"}


def test_agent_text_is_labeled_with_its_sender() -> None:
    entry = AgentToAgentMessage(sender="reviewer", content="finding")
    assert labeled_agent_send_text(entry) == "[from reviewer]: finding"


def test_agent_text_is_labeled_once() -> None:
    entry = AgentToAgentMessage(sender="reviewer", content="[from reviewer]: x")
    assert labeled_agent_send_text(entry) == "[from reviewer]: x"


def test_another_senders_label_is_not_mistaken_for_this_one() -> None:
    entry = AgentToAgentMessage(sender="b", content="[from a]: x")
    assert labeled_agent_send_text(entry) == "[from b]: [from a]: x"


def test_agent_message_without_content_is_just_the_label() -> None:
    assert labeled_agent_send_text(AgentToAgentMessage(sender="a")) == "[from a]: "


def test_compact_complete_takes_every_field_from_the_override() -> None:
    override = ContextSplice(
        ref=TapeRef(session_id="s", ordinal=4),
        mask=(),
        insert_after=None,
        payload=(UserMessage(content="summary"), _turn("ack")),
        strategy="summary",
        token_before=900,
        token_after=40,
        fallback_reason="summary failed",
        preserved_tail_count=3,
    )
    event = CompactComplete.from_override(override)
    assert event == CompactComplete(
        token_before=900,
        token_after=40,
        payload_entries=2,
        fallback_reason="summary failed",
        preserved_tail_count=3,
    )


def test_detached_result_is_keyed_by_its_results_call() -> None:
    result = ToolResult(
        call_id="c7",
        state="done",
        stdout=("ok",),
        result=UncategorizedToolResult(),
    )
    assert DetachedResult(result=result).call_id == "c7"


@pytest.mark.parametrize(
    ("entry", "role"),
    [
        (UserMessage(content="u"), "user"),
        (AgentToAgentMessage(sender="a", content="a"), "user"),
        (
            Response(
                records=(),
                tokens=TokenCount(),
                spend=TokenCost(),
                stop_reason="",
            ),
            "assistant",
        ),
        (ToolResult(call_id="c", state="done", result=UncategorizedToolResult()), None),
        (SystemMessage(content="n"), None),
        (TurnContext(cwd="/x"), None),
    ],
)
def test_wire_role(entry: ModelContextEvent, role: str | None) -> None:
    assert wire_role(entry) == role


def test_the_system_note_names_the_tool_deliveries_come_under() -> None:
    assert f"`{Markers().delivery}` foreground" in Markers().note
    assert Markers().arrived not in Markers().note


def _turn(text: str) -> Response:
    return Response(
        records=(AssistantMessage(content=text),),
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="end_turn",
    )


@pytest.mark.parametrize(
    ("exception", "attempt", "opens"),
    [
        (RateLimitedError("spent"), 1, True),
        (RateLimitedError("spent"), 2, False),
        (RetryableError("down"), 1, False),
    ],
)
def test_only_a_first_rate_limit_opens_a_quota_wait(
    exception: RetryableError,
    attempt: int,
    opens: bool,
) -> None:
    suspended = ModelCallSuspended(exception=exception, until=0.0, attempt=attempt)
    assert suspended.opens_a_quota_wait is opens


@pytest.mark.parametrize(
    ("elapsed", "cache_left", "gap", "forced"),
    [
        (0.0, math.inf, 15.0, False),
        (15.0, math.inf, 15.0, False),
        (30.0, math.inf, 30.0, False),
        (240.0, math.inf, 240.0, False),
        (299.0, math.inf, 299.0, False),
        (300.0, math.inf, 300.0, True),
        (480.0, math.inf, 300.0, True),
        (240.0, 30.0, 30.0, True),
        (10.0, 20.0, 15.0, False),
        (10.0, 15.0, 15.0, True),
    ],
)
def test_a_wake_comes_early_then_less_often_but_within_the_cap_and_the_cache(
    elapsed: float,
    cache_left: float,
    gap: float,
    forced: bool,
) -> None:
    """Paced by growth it fires on new output only; capped, it fires regardless."""
    assert WakeSchedule().gap(elapsed, cache_left_sec=cache_left) == (gap, forced)


def test_a_wake_with_no_growth_comes_at_a_fixed_gap() -> None:
    schedule = WakeSchedule(first_sec=10.0, growth=0.0, max_gap_sec=60.0)
    assert [schedule.gap(t, cache_left_sec=math.inf)[0] for t in (0, 50, 500)] == [
        10.0,
        10.0,
        10.0,
    ]


@pytest.mark.parametrize(
    ("knobs", "message"),
    [
        ({"first_sec": 0.0}, "first_sec must be a number of seconds above 0, got 0.0"),
        ({"first_sec": math.nan}, "first_sec must be a number of seconds above 0"),
        ({"growth": -1.0}, "growth must be a number at least 0, got -1.0"),
        ({"growth": math.inf}, "growth must be a number at least 0, got inf"),
        ({"max_gap_sec": 0.0}, "max_gap_sec must be a number of seconds above 0"),
        ({"max_gap_sec": math.inf}, "max_gap_sec must be a number of seconds above 0"),
    ],
)
def test_a_wake_schedule_refuses_knobs_that_never_wake(
    knobs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(message)}"):
        WakeSchedule(**knobs)


def test_a_wake_schedule_says_its_knobs() -> None:
    assert WakeSchedule().text() == "15s 1 300s"
    assert WakeSchedule(first_sec=0.5, growth=2, max_gap_sec=60).text() == "0.5s 2 60s"


def test_a_tool_without_an_override_wakes_as_the_default() -> None:
    fast = WakeSchedule(first_sec=5.0)
    schedule = DetachSchedule(
        default=fast,
        tools={"Bash": None, "Read": WakeSchedule()},
    )
    assert schedule.of("Grep") is fast
    assert schedule.of("Bash") is None
    assert schedule.of("Read") == WakeSchedule()
    assert DetachSchedule().of("Grep") == WakeSchedule()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
