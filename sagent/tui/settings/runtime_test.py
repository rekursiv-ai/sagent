"""Tests for the runtime as settings: when a call detaches, and how a rate limit waits."""

from __future__ import annotations

import re

import pytest

from sagent.model.scripted import ScriptedProvider
from sagent.runtime.runtime import Runtime
from sagent.settings.harvest import harvest
from sagent.tui.settings.runtime import RuntimeSettings


@pytest.mark.asyncio
async def test_detach_is_off_until_given_seconds() -> None:
    runtime = Runtime(model=ScriptedProvider()["scripted"])
    (setting,) = (
        s
        for s in harvest([(("runtime",), RuntimeSettings(runtime=runtime))])
        if s.name == ("runtime", "detach")
    )
    assert setting.value is not None

    assert setting.value() == "off (default)"
    assert await setting.apply("90") == "90s"
    assert runtime.detach_after_sec == 90.0
    assert await setting.apply("1.5s") == "1.5s"
    assert await setting.apply("off") == "off (default)"
    assert runtime.detach_after_sec is None
    for bad in ("0", "-3", "soon", "nan", "inf"):
        message = f"{bad!r} is not off or a number of seconds above 0"
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            await setting.apply(bad)
    assert runtime.detach_after_sec is None


@pytest.mark.asyncio
async def test_retry_waits_out_a_rate_limit_or_ends_the_turn() -> None:
    runtime = Runtime(model=ScriptedProvider()["scripted"])
    (setting,) = (
        s
        for s in harvest([(("runtime",), RuntimeSettings(runtime=runtime))])
        if s.name == ("runtime", "retry")
    )
    assert setting.value is not None

    assert setting.value() == "off"
    assert await setting.apply("persistent") == "persistent"
    assert runtime.persistent_retry
    with pytest.raises(ValueError, match="'maybe' is not one of: persistent, off"):
        await setting.apply("maybe")
    assert await setting.apply("off") == "off"
    assert not runtime.persistent_retry


def test_each_setting_says_how_it_is_typed_what_it_takes_and_what_it_does() -> None:
    runtime = Runtime(model=ScriptedProvider()["scripted"])
    detach, retry = harvest([(("runtime",), RuntimeSettings(runtime=runtime))])
    assert (detach.help_short, detach.usage, detach.options(), detach.cli) == (
        "detach a call the model waits on after this long",
        "off|SECONDS",
        ("off",),
        True,
    )
    assert detach.help_long == (
        "A detached call runs on; its result arrives later as a message, and the"
        " model works meanwhile."
    )
    assert (retry.help_short, retry.usage, retry.options(), retry.cli) == (
        "wait out a long rate limit, or end the turn on it",
        "persistent|off",
        ("persistent", "off"),
        True,
    )
    assert retry.help_long == ""


@pytest.mark.asyncio
async def test_detach_takes_default_as_off_and_any_time_above_zero() -> None:
    runtime = Runtime(model=ScriptedProvider()["scripted"])
    detach, _ = harvest([(("runtime",), RuntimeSettings(runtime=runtime))])
    await detach.apply("5")
    assert await detach.apply("default") == "off (default)"
    assert await detach.apply("0.5") == "0.5s"
    assert await detach.apply("1") == "1s"


def test_a_new_runtime_waits_on_every_call() -> None:
    assert Runtime(model=ScriptedProvider()["scripted"]).detach_after_sec is None


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
