"""Tests for the runtime event vocabulary."""

from __future__ import annotations

from datetime import UTC, datetime
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
    ModelContextEvent,
    ModelResponseSuspended,
    NextWake,
    SessionCompacted,
    WakeSchedule,
    wire_role,
)
from sagent.session.types import (
    AgentMessage,
    AssistantMessage,
    SessionContext,
    SystemMessage,
    ToolMessage,
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
        and "ModelResponseCompleted" in (text := path.read_text())
        and "Failed" in text
        and "ModelResponseSuspended" not in text
    }
    # The pruner hears a call only when it replied; its ``Failed`` is a prune's.
    assert unheard == {"agent/agent.py", "agent/pruner.py"}


def test_what_a_compaction_did_takes_every_field_from_the_override() -> None:
    override = ContextSplice(
        ref=TapeRef(session_id="s", ordinal=4),
        mask=(),
        insert_after=None,
        replacement=(UserMessage(content="summary"), _turn("ack")),
        strategy="summary",
        masked_tokens=900,
        replacement_tokens=40,
        summary_fallback_reason="summary failed",
        entries_kept_verbatim=3,
    )
    assert SessionCompacted.from_splice(override) == SessionCompacted(
        masked_tokens=900,
        replacement_tokens=40,
        replacement_entries=2,
        summary_fallback_reason="summary failed",
        entries_kept_verbatim=3,
    )


@pytest.mark.parametrize(
    ("entry", "role"),
    [
        (UserMessage(content="u"), "user"),
        (AgentMessage(sender="a", content="a"), "user"),
        (
            Response(
                records=(),
                tokens=TokenCount(),
                spend=TokenCost(),
                stop_reason="",
            ),
            "assistant",
        ),
        (
            ToolMessage(call_id="c", state="done", result=UncategorizedToolResult()),
            None,
        ),
        (SystemMessage(content="n"), None),
        (SessionContext(cwd=Path("/x")), None),
    ],
)
def test_wire_role(entry: ModelContextEvent, role: str | None) -> None:
    assert wire_role(entry) == role


def _turn(text: str) -> Response:
    return Response(
        records=(AssistantMessage(content=text),),
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="end_turn",
    )


@pytest.mark.parametrize(
    ("exception", "failures", "opens"),
    [
        (RateLimitedError("spent"), 1, True),
        (RateLimitedError("spent"), 2, False),
        (RetryableError("down"), 1, False),
    ],
)
def test_only_a_first_rate_limit_opens_a_quota_wait(
    exception: RetryableError,
    failures: int,
    opens: bool,
) -> None:
    suspended = ModelResponseSuspended(
        exception=exception,
        retry_at=datetime(2026, 1, 1, tzinfo=UTC),
        failures_in_a_row=failures,
    )
    assert suspended.opens_a_quota_wait is opens


@pytest.mark.parametrize(
    ("elapsed", "cache_left", "delay", "forced"),
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
    delay: float,
    forced: bool,
) -> None:
    """Paced by gap_growth it fires on new output only; capped, it fires regardless."""
    assert WakeSchedule().delay_sec_until_next_wake(
        elapsed,
        cache_left_sec=cache_left,
    ) == NextWake(delay_sec=delay, fires_even_without_new_output=forced)


def test_a_wake_with_no_growth_comes_at_a_fixed_gap() -> None:
    schedule = WakeSchedule(first_gap_sec=10.0, gap_growth=0.0, max_gap_sec=60.0)
    assert [
        schedule.delay_sec_until_next_wake(t, cache_left_sec=math.inf).delay_sec
        for t in (0, 50, 500)
    ] == [
        10.0,
        10.0,
        10.0,
    ]


@pytest.mark.parametrize(
    ("knobs", "message"),
    [
        (
            {"first_gap_sec": 0.0},
            "first_gap_sec must be a number of seconds above 0, got 0.0",
        ),
        (
            {"first_gap_sec": math.nan},
            "first_gap_sec must be a number of seconds above 0",
        ),
        ({"gap_growth": -1.0}, "gap_growth must be a number at least 0, got -1.0"),
        ({"gap_growth": math.inf}, "gap_growth must be a number at least 0, got inf"),
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
    assert str(WakeSchedule()) == "15s 1 300s"
    assert (
        str(WakeSchedule(first_gap_sec=0.5, gap_growth=2, max_gap_sec=60))
        == "0.5s 2 60s"
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
