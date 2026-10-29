"""Tests for the status line: the session's sums, the spinner, and what it waits on."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import asyncio

import pytest

from sagent.model.capability import ModelCapability
from sagent.model.cost import TokenCost, TokenCount
from sagent.model.scripted import ScriptedProvider, reply
from sagent.runtime import types
from sagent.runtime.runtime import Await, Runtime
from sagent.session.types import UserMessage
from sagent.tui import status
from sagent.tui.status import Status, count, elapsed


@dataclass(slots=True, kw_only=True)
class _Clock:
    """Both clocks the status line reads, set by the test."""

    mono: float = 100.0
    wall: float = 1_800_000_000.0

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return self.wall


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(status, "time", fake)
    return fake


def _status(*, wait_after_sec: float = 15.0) -> Status:
    runtime = Runtime(model=ScriptedProvider()["scripted"])
    return Status(runtime, wait_after_sec=wait_after_sec, spinner="abcd")


def _complete(
    tokens: TokenCount | None = None,
    spend: float = 0.0,
) -> types.ModelResponseComplete:
    return types.ModelResponseComplete(
        response=reply(
            "x",
            tokens=tokens if tokens is not None else TokenCount(),
            spend=TokenCost(request=spend),
        ),
    )


def _call(
    line: Status,
    clock: _Clock,
    *,
    sec: float,
    event: types.RuntimeEvent,
) -> None:
    line(types.ModelCallStarted())
    clock.mono += sec
    line(event)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "0s"),
        (12.9, "12s"),
        (60, "1m 0s"),
        (83, "1m 23s"),
        (3_600, "1h 0m 0s"),
        (2 * 3_600 + 17 * 60 + 4, "2h 17m 4s"),
        (86_400, "1d 0h 0m 0s"),
        (3 * 86_400 + 3_600 + 125, "3d 1h 2m 5s"),
    ],
)
def test_elapsed_always_shows_seconds(seconds: float, text: str) -> None:
    assert elapsed(seconds) == text


@pytest.mark.parametrize(
    ("n", "text"),
    [
        (0, "0"),
        (412, "412"),
        (9_999, "9999"),
        (10_000, "10K"),
        (12_400, "12K"),
        (12_505, "13K"),
        (999_499, "999K"),
        (999_500, "1.0M"),
        (1_840_000, "1.8M"),
        (1_850_000, "1.9M"),
    ],
)
def test_count_abbreviates_large_numbers(n: int, text: str) -> None:
    assert count(n) == text


def test_nothing_shows_before_the_first_call() -> None:
    line = _status()
    assert line.frame() == ""
    line(UserMessage(content="hi"))
    assert line.frame() == ""


def test_a_resumed_session_s_spend_shows_from_the_start(clock: _Clock) -> None:
    runtime = Runtime(model=ScriptedProvider()["scripted"])
    line = Status(runtime, spend=1.25)
    assert line.frame() == "[0s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $1.25]"
    _call(line, clock, sec=2.0, event=_complete(spend=0.5))
    assert line.frame() == "[2s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $1.75]"


def test_an_ended_call_shows_the_sums_without_a_spinner(clock: _Clock) -> None:
    line = _status()
    tokens = TokenCount(
        request=12_000,
        response=3_400,
        cache_write=6_000,
        cache_write_1h=3_000,
        cache_read=210_000,
    )
    _call(line, clock, sec=83.0, event=_complete(tokens, spend=0.42))
    assert line.frame() == "[1m 23s 12K↑ 3400↓ 9000↟ 210K↡ (91%↡ 117%↔) $0.42]"


def test_the_cache_s_share_and_the_window_s_fill_follow_the_sums(
    clock: _Clock,
) -> None:
    """42% of prompt tokens came from cache; the window holds 71K of 200K, 36%."""
    line = _status()
    first = TokenCount(request=10, cache_write=50_000, response=1_000)
    _call(line, clock, sec=1.0, event=_complete(first))
    second = TokenCount(
        request=12,
        cache_write=20_000,
        cache_read=50_000,
        response=1_144,
    )
    _call(line, clock, sec=1.0, event=_complete(second))
    assert line.frame() == "[2s 22↑ 2144↓ 70K↟ 50K↡ (42%↡ 36%↔) $0.00]"


@pytest.mark.parametrize(
    ("end", "held"),
    [
        (types.CompactComplete(token_after=10_000), " 5%↔)"),
        (types.ClearComplete(), " 0%↔)"),
    ],
    ids=["compacted", "cleared"],
)
def test_the_window_s_fill_drops_when_it_is_compacted_or_cleared(
    clock: _Clock,
    end: types.RuntimeEvent,
    held: str,
) -> None:
    line = _status()
    _call(line, clock, sec=1.0, event=_complete(TokenCount(cache_write=150_000)))
    line(end)
    assert held in line.frame()


def test_a_context_size_chosen_is_the_window_shown(clock: _Clock) -> None:
    line = _status()
    model = line.runtime.model
    model.settings = replace(
        model.settings,
        context_size=model.capability.prices["auto", 100_000],
    )
    _call(line, clock, sec=1.0, event=_complete(TokenCount(cache_write=25_000)))
    assert " 25%↔)" in line.frame()


def test_a_model_with_no_window_shows_no_fill(clock: _Clock) -> None:
    runtime = Runtime(model=ScriptedProvider(capability=ModelCapability())["s"])
    line = Status(runtime)
    _call(line, clock, sec=1.0, event=_complete(TokenCount(request=5)))
    assert line.frame() == "[1s 5↑ 0↓ 0↟ 0↡ (0%↡) $0.00]"


def test_calls_add_up(clock: _Clock) -> None:
    line = _status()
    _call(line, clock, sec=10.0, event=_complete(TokenCount(request=5), spend=0.25))
    _call(line, clock, sec=5.0, event=_complete(TokenCount(request=7), spend=0.5))
    assert line.frame() == "[15s 12↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.75]"


def test_a_cancelled_or_failed_call_adds_its_time_but_no_tokens(clock: _Clock) -> None:
    line = _status()
    _call(line, clock, sec=4.0, event=types.ModelResponseCancelled())
    _call(
        line,
        clock,
        sec=3.0,
        event=types.ModelResponseError(exception=RuntimeError("x")),
    )
    assert line.frame() == "[7s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00]"


def test_an_ending_without_a_start_adds_no_time() -> None:
    line = _status()
    line(_complete(TokenCount(request=1)))
    assert line.frame() == ""


def test_text_streamed_before_any_call_starts_is_ignored() -> None:
    line = _status()
    line(types.ModelResponsePartial(text="early"))
    assert line.frame() == ""
    line(types.ModelCallStarted())
    assert line.frame() == "a [0s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00]"


def test_a_running_call_spins_and_counts_its_time(clock: _Clock) -> None:
    line = _status()
    _call(line, clock, sec=10.0, event=_complete())
    line(types.ModelCallStarted())
    clock.mono += 0.5
    assert line.frame() == "c [10s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00]"
    clock.mono += 2.7
    assert line.frame() == "a [13s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00]"


def test_a_running_call_estimates_what_it_streamed() -> None:
    line = _status()
    line(types.ModelCallStarted())
    line(types.ModelResponsePartial(text="x" * 40))
    line(types.ModelResponsePartial(text="x" * 4))
    assert line.frame() == "a [0s 0↑ 11↓ 0↟ 0↡ (0%↡ 0%↔) $0.00]"


def test_an_ended_call_counts_what_it_was_billed_not_what_it_streamed(
    clock: _Clock,
) -> None:
    line = _status()
    line(types.ModelCallStarted())
    line(types.ModelResponsePartial(text="x" * 400))
    clock.mono += 1.0
    line(_complete(TokenCount(response=100)))
    assert line.frame() == "[1s 0↑ 100↓ 0↟ 0↡ (0%↡ 0%↔) $0.00]"


def test_a_new_call_starts_its_estimate_from_zero() -> None:
    line = _status()
    line(types.ModelCallStarted())
    line(types.ModelResponsePartial(text="x" * 400))
    line(types.ModelCallStarted())
    line(types.ModelResponsePartial(text="xxx"))
    assert line.frame() == "a [0s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00]"


def test_the_defaults_wait_fifteen_seconds_and_spin_braille(clock: _Clock) -> None:
    line = Status(Runtime(model=ScriptedProvider()["scripted"]))
    line(types.ModelCallStarted())
    clock.mono += 14.5
    assert line.frame() == "⠹ [14s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00]"
    clock.mono += 0.5
    assert line.frame() == "⠴ [15s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00; waiting on model.]"


def test_what_is_charged_to_the_agent_is_added_to_the_spend(clock: _Clock) -> None:
    """A child's calls and a compaction's arrive as charges, as the cap counts them."""
    line = _status()
    _call(line, clock, sec=1.0, event=_complete(spend=0.5))
    line(types.Charged(usd=1.25, by="Agent_0"))
    line(types.Charged(usd=0.25, by=""))
    assert line.frame() == "[1s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $2.00]"


def test_a_finished_child_adds_nothing_its_charges_did_not(clock: _Clock) -> None:
    line = _status()
    _call(line, clock, sec=1.0, event=_complete(spend=0.5))
    line(types.ChildDoneEvent(label="Agent_0", elapsed=9.0, tokens=100, cost=1.25))
    assert line.frame() == "[1s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.50]"


def test_the_wait_shows_only_once_the_call_runs_past_the_threshold(
    clock: _Clock,
) -> None:
    line = _status(wait_after_sec=15.0)
    _call(line, clock, sec=100.0, event=_complete())
    line(types.ModelCallStarted())
    clock.mono += 14.5
    assert line.frame().endswith("$0.00]")
    clock.mono += 0.5
    assert line.frame().endswith("$0.00; waiting on model.]")


@pytest.mark.asyncio
async def test_running_tools_are_the_wait() -> None:
    line = _status(wait_after_sec=0.0)
    line(types.ModelCallStarted())
    line.runtime.cohort.add("c1")
    assert line.frame() == "a [0s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00; waiting on tools.]"
    line.runtime.cohort.clear()
    task = asyncio.create_task(asyncio.sleep(0))
    line.runtime.running_tools["c2"] = task
    assert line.frame().endswith("; waiting on tools.]")
    await task


def test_tools_running_after_the_reply_spin_and_name_the_wait(clock: _Clock) -> None:
    line = _status(wait_after_sec=15.0)
    _call(line, clock, sec=2.0, event=_complete())
    line(types.CohortStarted())
    line.runtime.cohort.add("c1")
    clock.mono += 0.5
    assert line.frame() == "c [2s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00]"
    clock.mono += 15.0
    assert line.frame().endswith("$0.00; waiting on tools.]")
    line.runtime.cohort.clear()
    assert line.frame() == "[2s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00]"


@dataclass(frozen=True, slots=True, kw_only=True)
class _Job:
    """A job as ``Runtime.jobs`` lists it, in the state the test sets."""

    state: Literal["running", "sleeping", "detached"]


def test_calls_in_the_background_are_counted(
    clock: _Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    line = _status()
    _call(line, clock, sec=1.0, event=_complete())
    jobs = {"a": _Job(state="detached"), "b": _Job(state="sleeping")}

    def held(runtime: Runtime) -> dict[str, _Job]:
        del runtime
        return jobs

    monkeypatch.setattr(Runtime, "jobs", property(held))
    assert line.frame() == "[1s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00; 2 in background.]"
    jobs["c"] = _Job(state="running")
    assert line.frame() == "[1s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00; 2 in background.]"
    line.runtime.inbox.push_front(Await(types=(UserMessage,)))
    assert line.frame().endswith("; waiting for input. 2 in background.]")


def test_an_armed_gate_waits_for_input_at_once() -> None:
    line = _status()
    line(types.ModelCallStarted())
    line.runtime.inbox.push_front(Await(types=(UserMessage,)))
    assert line.frame() == "a [0s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00; waiting for input.]"


def test_an_armed_gate_shows_while_idle(clock: _Clock) -> None:
    line = _status()
    _call(line, clock, sec=2.0, event=_complete())
    line.runtime.inbox.push_front(Await(types=(UserMessage,)))
    assert line.frame() == "[2s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00; waiting for input.]"


def test_a_suspension_counts_down_on_the_wall_clock(clock: _Clock) -> None:
    line = _status()
    line(types.ModelCallStarted())
    clock.mono += 10.0
    line.runtime.service_suspended_until = clock.wall + 65.0
    line.runtime.inbox.push_front(Await(types=(UserMessage,)))
    assert line.frame() == "c [10s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00; retrying in 1m 5s.]"
    clock.wall += 65.0
    assert line.frame() == "c [10s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00; waiting for input.]"


def test_compaction_spins_and_outranks_an_armed_gate(clock: _Clock) -> None:
    line = _status()
    _call(line, clock, sec=3.0, event=_complete())
    line(types.CompactStarted())
    line.runtime.inbox.push_front(Await(types=(UserMessage,)))
    clock.mono += 0.3
    assert line.frame() == "b [3s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00; compacting.]"


@pytest.mark.parametrize(
    "end",
    [
        types.CompactComplete(),
        types.CompactFailed(exception=RuntimeError(), tape_len=0),
    ],
)
def test_compaction_ends_on_either_outcome(
    end: types.RuntimeEvent,
) -> None:
    line = _status()
    line(types.CompactStarted())
    assert line.frame() == "a [0s 0↑ 0↓ 0↟ 0↡ (0%↡ 0%↔) $0.00; compacting.]"
    line(end)
    assert line.frame() == ""


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
