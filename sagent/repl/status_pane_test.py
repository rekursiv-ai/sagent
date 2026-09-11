"""Tests for ``repl.status_pane``: status-pane string assembly."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast
from unittest.mock import patch

import time

import pytest

from sagent.agent.agent import ActivityTracker, Agent
from sagent.agent.cost_tracker import CostTracker
from sagent.lib.custom_json import FloatCodec
from sagent.repl.status_pane import render_status_pane
from sagent.types.cost import TokenCost, TokenCount
from sagent.types.model import AgentSettings


@dataclass(slots=True, kw_only=True)
class _FakeCostTracker:
    total: TokenCount = field(default_factory=TokenCount)
    spend: TokenCost = field(default_factory=TokenCost)


@dataclass(slots=True, kw_only=True)
class _FakeInbox:
    gate_armed: bool = False


@dataclass(slots=True, kw_only=True)
class _FakeRuntime:
    model_call: object = None
    compact_task: object = None
    running_tools: dict[str, object] = field(default_factory=dict)
    cohort: set[str] = field(default_factory=set)
    inbox: _FakeInbox = field(default_factory=_FakeInbox)
    service_suspended_until: float | None = None


@dataclass(slots=True, kw_only=True)
class _FakeModel:
    """Model stub: ``approx_text_tokens`` at 4 chars/token (truncating)."""

    def approx_text_tokens(self, text: str) -> int:
        return len(text) // 4


@dataclass(slots=True, kw_only=True)
class _FakeAgent:
    """Minimal stand-in for ``Agent`` matching only the surface the status pane reads."""

    activity: ActivityTracker = field(default_factory=ActivityTracker)
    cost_tracker: _FakeCostTracker = field(default_factory=_FakeCostTracker)
    runtime: _FakeRuntime = field(default_factory=_FakeRuntime)
    model: _FakeModel = field(default_factory=_FakeModel)
    budget: AgentSettings = field(
        default_factory=lambda: AgentSettings(
            max_request_tokens=200_000,
            max_response_tokens=8_192,
            buffer_tokens=4_096,
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
                request=cast(
                    int,
                    overrides.get("input_tokens", a.cost_tracker.total.request),
                ),
                response=cast(
                    int,
                    overrides.get("output_tokens", a.cost_tracker.total.response),
                ),
                cache_write=cast(
                    int,
                    overrides.get(
                        "cache_creation_tokens",
                        a.cost_tracker.total.cache_write,
                    ),
                ),
                cache_read=cast(
                    int,
                    overrides.get("cache_read_tokens", a.cost_tracker.total.cache_read),
                ),
            )
        elif k == "total_cost_usd":
            a.cost_tracker.spend = TokenCost(request=FloatCodec.coerce(v))
        elif k == "elapsed_seconds":
            a.activity.elapsed_seconds = FloatCodec.coerce(v)
        elif k == "active":
            a.activity.active = bool(v)
        elif k == "current_call_start":
            a.activity.current_call_start = FloatCodec.coerce(v)
        elif k == "current_compact_start":
            a.activity.current_compact_start = FloatCodec.coerce(v)
        elif k == "live_response_text":
            a.activity.live_response_text = str(v)
        elif k == "compact_task":
            a.runtime.compact_task = v
        elif k == "service_suspended_until":
            a.runtime.service_suspended_until = FloatCodec.coerce(v)
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
    assert s[1:].startswith(" [")


@pytest.mark.usefixtures("patched_loop_time")
def test_active_includes_live_output_estimate() -> None:
    # The status pane tokenizes the streamed text as a whole and adds it
    # to the settled output total while active (1000 chars // 4 = 250).
    a = _agent(
        active=True,
        current_call_start=0.0,
        elapsed_seconds=0.0,
        input_tokens=100,
        output_tokens=500,
        total_cost_usd=0.10,
        live_response_text="x" * 1000,
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
        live_response_text="x" * 1000,
    )
    s = render_status_pane(_as_agent(a))
    assert "500↓" in s
    assert "750" not in s


@pytest.mark.usefixtures("patched_loop_time")
def test_active_tool_phase_keeps_bracket_prefix_stable() -> None:
    a = _agent(
        active=True,
        current_call_start=5.0,
        elapsed_seconds=0.0,
    )
    a.runtime.running_tools["c1"] = object()
    a.runtime.cohort.add("c1")
    s = render_status_pane(_as_agent(a))
    assert s[0] in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    assert s[1:] == " [5s 0↑ 0↓ 0↟ 0↡ $0.00]"


@pytest.mark.usefixtures("patched_loop_time")
def test_active_zero_elapsed_renders() -> None:
    a = _agent(active=True, current_call_start=8.0, elapsed_seconds=0.0)
    s = render_status_pane(_as_agent(a))
    assert "[2s 0↑ 0↓ 0↟ 0↡ $0.00]" in s


@pytest.mark.usefixtures("patched_loop_time")
def test_compacting_branch_renders_suffix_reason() -> None:
    a = _agent(
        current_compact_start=5.0,
        elapsed_seconds=0.0,
        input_tokens=10,
        output_tokens=20,
        total_cost_usd=0.05,
    )
    s = render_status_pane(_as_agent(a))
    assert s[0] in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    assert s[1:] == " [0s 10↑ 20↓ 0↟ 0↡ $0.05; compacting.]"


@pytest.mark.usefixtures("patched_loop_time")
def test_active_wait_reason_appears_after_threshold() -> None:
    a = _agent(active=True, current_call_start=-6.0)
    assert (
        render_status_pane(_as_agent(a))[1:]
        == " [16s 0↑ 0↓ 0↟ 0↡ $0.00; waiting on model.]"
    )


@pytest.mark.usefixtures("patched_loop_time")
def test_active_wait_reason_hidden_before_threshold() -> None:
    a = _agent(active=True, current_call_start=0.0)
    assert render_status_pane(_as_agent(a))[1:] == " [10s 0↑ 0↓ 0↟ 0↡ $0.00]"


@pytest.mark.usefixtures("patched_loop_time")
def test_active_wait_reason_per_turn_not_session_total() -> None:
    """The 15s wait-reason threshold applies to the CURRENT turn only.

    After many short turns, cumulative ``elapsed_seconds`` will exceed
    the threshold even when the fresh turn just started -- the user
    would see "waiting on model." from second 0 of every subsequent
    turn. Correct behavior: hide the reason until the CURRENT call has
    been in flight for >=15s, regardless of session-total time.
    """
    # 100s of prior session activity already banked; current turn 1s in.
    a = _agent(
        active=True,
        elapsed_seconds=100.0,
        current_call_start=9.0,
    )
    s = render_status_pane(_as_agent(a))
    assert "waiting on model." not in s, (
        "wait reason fired on cumulative session time, not current-turn time;"
        f" got {s!r}"
    )


@pytest.mark.usefixtures("patched_loop_time")
def test_tools_wait_reason() -> None:
    a = _agent(active=True, current_call_start=-6.0)
    a.runtime.running_tools["c1"] = object()
    assert (
        render_status_pane(_as_agent(a))[1:]
        == " [16s 0↑ 0↓ 0↟ 0↡ $0.00; waiting on tools.]"
    )


@pytest.mark.usefixtures("patched_loop_time")
def test_auth_gate_reason_is_immediate() -> None:
    a = _agent(active=True, current_call_start=9.0)
    a.runtime.inbox.gate_armed = True
    assert (
        render_status_pane(_as_agent(a))[1:]
        == " [1s 0↑ 0↓ 0↟ 0↡ $0.00; waiting for input.]"
    )


@pytest.mark.usefixtures("patched_loop_time")
def test_service_suspension_countdown_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``service_suspended_until`` is a wall-clock epoch; the countdown reads
    # ``time.time()`` (patched here), not the monotonic loop clock.
    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)
    a = _agent(
        active=True,
        current_call_start=0.0,
        service_suspended_until=1_800_000_065.0,
    )
    assert (
        render_status_pane(_as_agent(a))[1:]
        == " [10s 0↑ 0↓ 0↟ 0↡ $0.00; retrying in 1m 5s.]"
    )


@pytest.mark.usefixtures("patched_loop_time")
def test_service_suspension_uses_wall_clock_not_loop_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Issue#316 RUNTIME-003: ``service_suspended_until`` is a WALL-CLOCK epoch
    # (``ModelServiceSuspended.retry_at`` == ``time.time() + delay``). The
    # countdown must compare against ``time.time()``, not the monotonic loop
    # clock (patched to 10.0 above), or a 2-minute suspension renders as a
    # ~57-year wait.
    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)
    a = _agent(
        active=True,
        current_call_start=0.0,
        service_suspended_until=1_800_000_120.0,
    )
    assert "retrying in 2m 0s." in render_status_pane(_as_agent(a))


@pytest.mark.usefixtures("patched_loop_time")
def test_compaction_outranks_gate_armed_reason() -> None:
    """``compacting.`` must win over ``waiting for input.`` during compaction.

    During compaction the inbox gate can be armed (the compactor itself
    waits on a model response), but the user-visible state is
    "compacting" -- no input is actually being accepted. Reporting
    "waiting for input" would invite the user to type when typing won't
    advance anything.
    """
    a = _agent(
        current_compact_start=5.0,
        elapsed_seconds=0.0,
        input_tokens=10,
        output_tokens=20,
        total_cost_usd=0.05,
    )
    a.runtime.inbox.gate_armed = True
    s = render_status_pane(_as_agent(a))
    assert "compacting." in s, f"compaction state must outrank gate-armed; got {s!r}"
    assert "waiting for input." not in s, (
        f"gate-armed reason must not fire during compaction; got {s!r}"
    )


def test_real_agent_cost_tracker_is_compatible() -> None:
    """``render_status_pane`` accepts a real ``CostTracker``/``ActivityTracker``."""

    class _Dummy:
        activity = ActivityTracker(elapsed_seconds=2.0)
        cost_tracker = CostTracker()
        budget = AgentSettings(
            max_request_tokens=1000,
            max_response_tokens=100,
            buffer_tokens=10,
        )

    s = render_status_pane(cast(Agent, _Dummy()))
    assert s.startswith("[2s ")


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
