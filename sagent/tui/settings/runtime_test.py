"""Tests for the runtime as settings: when long calls wake the model, how a rate limit waits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import re

import pytest

from sagent.model.scripted import ScriptedProvider
from sagent.runtime.runtime import Runtime
from sagent.runtime.types import DetachSchedule, WakeSchedule
from sagent.settings.harvest import harvest
from sagent.tools.bash import Bash
from sagent.tools.read import Read
from sagent.tui.settings.runtime import RuntimeSettings


if TYPE_CHECKING:
    from sagent.settings.types import Setting


def _settings(runtime: Runtime) -> dict[str, Setting]:
    return {
        ".".join(s.name): s
        for s in harvest([(("runtime",), RuntimeSettings(runtime=runtime))])
    }


def _runtime() -> Runtime:
    return Runtime(model=ScriptedProvider()["scripted"], tools=[Bash(), Read()])


@pytest.mark.asyncio
async def test_detach_sets_when_every_tool_wakes_the_model() -> None:
    runtime = _runtime()
    setting = _settings(runtime)["runtime.detach"]
    assert setting.value is not None

    assert setting.value() == "15s 1 300s (default)"
    assert await setting.apply("10 0.5 120") == "10s 0.5 120s"
    assert runtime.detach_schedule.default == WakeSchedule(
        first_sec=10.0,
        growth=0.5,
        max_gap_sec=120.0,
    )
    assert await setting.apply("5s,2,60s") == "5s 2 60s"
    assert await setting.apply("off") == "off"
    assert runtime.detach_schedule.default is None
    assert await setting.apply("default") == "15s 1 300s (default)"
    assert runtime.detach_schedule == DetachSchedule()


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ("10 1", "'10 1' is not FIRST GROWTH MAX_GAP, off, or default"),
        ("a b c", "'a b c' is not FIRST GROWTH MAX_GAP, off, or default"),
        ("0 1 300", "first_sec must be a number of seconds above 0, got 0.0."),
        ("10 -1 300", "growth must be a number at least 0, got -1.0."),
    ],
)
@pytest.mark.asyncio
async def test_detach_refuses_what_is_not_a_schedule(bad: str, message: str) -> None:
    runtime = _runtime()
    settings = _settings(runtime)
    for name in ("runtime.detach", "runtime.detach.Bash"):
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            await settings[name].apply(bad)
    assert runtime.detach_schedule == DetachSchedule()


@pytest.mark.asyncio
async def test_a_tool_s_own_detach_wins_over_the_default_until_given_up() -> None:
    runtime = _runtime()
    settings = _settings(runtime)
    bash = settings["runtime.detach.Bash"]
    assert bash.value is not None
    assert bash.value() == "default"
    assert await bash.apply("off") == "off"
    assert runtime.detach_schedule.of("Bash") is None
    assert runtime.detach_schedule.of("Read") == WakeSchedule()
    assert await settings["runtime.detach"].apply("30 1 600") == "30s 1 600s"
    assert runtime.detach_schedule.of("Bash") is None
    assert await bash.apply("1 0 5") == "1s 0 5s"
    assert runtime.detach_schedule.of("Bash") == WakeSchedule(
        first_sec=1.0,
        growth=0.0,
        max_gap_sec=5.0,
    )
    assert await bash.apply("default") == "default"
    assert runtime.detach_schedule.of("Bash") == WakeSchedule(
        first_sec=30.0,
        max_gap_sec=600.0,
    )


@pytest.mark.asyncio
async def test_retry_waits_out_a_rate_limit_or_ends_the_turn() -> None:
    runtime = _runtime()
    setting = _settings(runtime)["runtime.retry"]
    assert setting.value is not None

    assert setting.value() == "off"
    assert await setting.apply("persistent") == "persistent"
    assert runtime.persistent_retry
    with pytest.raises(ValueError, match="'maybe' is not one of: persistent, off"):
        await setting.apply("maybe")
    assert await setting.apply("off") == "off"
    assert not runtime.persistent_retry


def test_each_setting_says_how_it_is_typed_what_it_takes_and_what_it_does() -> None:
    settings = _settings(_runtime())
    assert list(settings) == [
        "runtime.detach",
        "runtime.detach.Bash",
        "runtime.detach.Read",
        "runtime.retry",
    ]
    detach, bash, retry = (
        settings["runtime.detach"],
        settings["runtime.detach.Bash"],
        settings["runtime.retry"],
    )
    assert (detach.help_short, detach.usage, detach.options(), detach.cli) == (
        "when a long call wakes the model, and detaches",
        "FIRST GROWTH MAX_GAP|off|default",
        ("off", "default"),
        True,
    )
    assert detach.help_long == (
        "The first wake is FIRST seconds in; each gap after is GROWTH times how"
        " long the call has run, at least FIRST, at most MAX_GAP and the prompt"
        " cache's lifetime. The first wake answers the calls with what they wrote"
        " and lets them run on; each later one shows the model what is new."
    )
    assert (bash.help_short, bash.usage, bash.options(), bash.cli) == (
        "when a long Bash call wakes the model",
        "FIRST GROWTH MAX_GAP|off|default",
        ("off", "default"),
        True,
    )
    assert (retry.help_short, retry.usage, retry.options(), retry.cli) == (
        "wait out a long rate limit, or end the turn on it",
        "persistent|off",
        ("persistent", "off"),
        True,
    )
    assert retry.help_long == ""


def test_a_new_runtime_wakes_on_the_default_schedule() -> None:
    assert _runtime().detach_schedule == DetachSchedule()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
