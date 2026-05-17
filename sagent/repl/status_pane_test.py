"""Tests for ``repl.status_pane``: status-pane string assembly."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast
from unittest.mock import patch

import pytest

from sagent.agent.agent import ActivityTracker, Agent
from sagent.agent.cost_tracker import CostTracker
from sagent.repl.status_pane import render_status_pane
from sagent.types.model import ContextBudget, TokenCount


@dataclass(slots=True, kw_only=True)
class _FakeCostTracker:
    total: TokenCount = field(default_factory=TokenCount)
    total_cost_usd: float = 0.0


@dataclass(slots=True, kw_only=True)
class _FakeAgent:
    """Minimal stand-in for ``Agent`` matching only the surface the status pane reads."""

    activity: ActivityTracker = field(default_factory=ActivityTracker)
    cost_tracker: _FakeCostTracker = field(default_factory=_FakeCostTracker)
    budget: ContextBudget = field(
        default_factory=lambda: ContextBudget(
            max_request_tokens=200_000,
            max_response_tokens=8_192,
            keep_recent_on_compact=8,
            buffer_tokens=4_096,
            chars_per_token=4,
        ),
    )


def _agent(**overrides: object) -> _FakeAgent:
    a = _FakeAgent()
    for k, v in overrides.items():
        if k in {
            "input_tokens",
            "output_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
        }:
            a.cost_tracker.total = TokenCount(
                input_tokens=cast(
                    int,
                    overrides.get("input_tokens", a.cost_tracker.total.input_tokens),
                ),
                output_tokens=cast(
                    int,
                    overrides.get("output_tokens", a.cost_tracker.total.output_tokens),
                ),
                cache_creation_tokens=cast(
                    int,
                    overrides.get(
                        "cache_creation_tokens",
                        a.cost_tracker.total.cache_creation_tokens,
                    ),
                ),
                cache_read_tokens=cast(
                    int,
                    overrides.get(
                        "cache_read_tokens", a.cost_tracker.total.cache_read_tokens
                    ),
                ),
            )
        elif k == "total_cost_usd":
            a.cost_tracker.total_cost_usd = cast(float, v)
        elif k == "elapsed_seconds":
            a.activity.elapsed_seconds = cast(float, v)
        elif k == "active":
            a.activity.active = cast(bool, v)
        elif k == "current_call_start":
            a.activity.current_call_start = cast(float, v)
        elif k == "live_response_chars":
            a.activity.live_response_chars = cast(int, v)
    return a


def _as_agent(a: _FakeAgent) -> Agent:
    return cast(Agent, a)


@pytest.fixture
def patched_loop_time() -> Iterator[None]:
    """Patch ``asyncio.get_running_loop().time()`` to a deterministic value."""
    with patch("asyncio.get_running_loop") as mock_loop:
        mock_loop.return_value.time.return_value = 10.0
        yield


def test_empty_when_no_activity_no_tokens() -> None:
    assert render_status_pane(_as_agent(_agent())) == ""


def test_idle_renders_bracket() -> None:
    a = _agent(
        elapsed_seconds=50.0,
        input_tokens=18,
        output_tokens=3114,
        cache_creation_tokens=140_000,
        cache_read_tokens=1_800_000,
        total_cost_usd=0.98,
    )
    assert render_status_pane(_as_agent(a)) == "[50s 18↑ 3114↓ 140K↟ 1.8M↡ $0.98]"


def test_idle_minutes_format() -> None:
    a = _agent(
        elapsed_seconds=83.0,
        input_tokens=67,
        output_tokens=8902,
        total_cost_usd=14.71,
    )
    assert render_status_pane(_as_agent(a)) == "[1m 23s 67↑ 8902↓ 0↟ 0↡ $14.71]"


def test_idle_hours_format() -> None:
    a = _agent(
        elapsed_seconds=8220.0,
        input_tokens=3200,
        output_tokens=412_000,
        cache_creation_tokens=18_000_000,
        cache_read_tokens=241_000_000,
        total_cost_usd=487.12,
    )
    assert (
        render_status_pane(_as_agent(a))
        == "[2h 17m 0s 3200↑ 412K↓ 18.0M↟ 241.0M↡ $487.12]"
    )


def test_idle_zero_cost_still_renders() -> None:
    a = _agent(
        elapsed_seconds=1.0,
        input_tokens=1,
        output_tokens=2,
        total_cost_usd=0.0,
    )
    assert render_status_pane(_as_agent(a)) == "[1s 1↑ 2↓ 0↟ 0↡ $0.00]"


@pytest.mark.usefixtures("patched_loop_time")
def test_active_prefixes_spinner() -> None:
    a = _agent(
        active=True,
        current_call_start=5.0,
        elapsed_seconds=0.0,
        input_tokens=10,
        output_tokens=20,
        total_cost_usd=0.05,
    )
    s = render_status_pane(_as_agent(a))
    assert s.endswith("[5s 10↑ 20↓ 0↟ 0↡ $0.05]")
    assert s[0] in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    assert s[1] == " "


@pytest.mark.usefixtures("patched_loop_time")
def test_active_includes_live_output_estimate() -> None:
    a = _agent(
        active=True,
        current_call_start=0.0,
        elapsed_seconds=0.0,
        input_tokens=100,
        output_tokens=500,
        total_cost_usd=0.10,
        live_response_chars=1000,  # / 4 chars_per_token = 250 tokens added
    )
    s = render_status_pane(_as_agent(a))
    assert "750↓" in s


def test_idle_ignores_live_output_estimate() -> None:
    a = _agent(
        active=False,
        elapsed_seconds=10.0,
        input_tokens=100,
        output_tokens=500,
        total_cost_usd=0.10,
        live_response_chars=1000,
    )
    s = render_status_pane(_as_agent(a))
    assert "500↓" in s
    assert "750" not in s


@pytest.mark.usefixtures("patched_loop_time")
def test_active_zero_elapsed_renders() -> None:
    a = _agent(active=True, current_call_start=8.0, elapsed_seconds=0.0)
    s = render_status_pane(_as_agent(a))
    assert "[2s 0↑ 0↓ 0↟ 0↡ $0.00]" in s


def test_real_agent_cost_tracker_is_compatible() -> None:
    """``render_status_pane`` accepts a real ``CostTracker``/``ActivityTracker``."""

    class _Dummy:
        activity = ActivityTracker(elapsed_seconds=2.0)
        cost_tracker = CostTracker()
        budget = ContextBudget(
            max_request_tokens=1000,
            max_response_tokens=100,
            keep_recent_on_compact=4,
            buffer_tokens=10,
            chars_per_token=4,
        )

    s = render_status_pane(cast(Agent, _Dummy()))
    assert s.startswith("[2s ")


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
