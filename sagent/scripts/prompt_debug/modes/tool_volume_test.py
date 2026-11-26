"""Tests for the ``tool_volume`` mode, on scripted models."""

from __future__ import annotations

import asyncio

import pytest

from sagent.journal.types import ToolMessage
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.scripts.prompt_debug.harness import Case, Run
from sagent.scripts.prompt_debug.modes.tool_volume import ToolVolumeMode


def test_the_mode_reads_its_bound() -> None:
    mode = ToolVolumeMode.parse({"max_tool_calls": 2}, {}, where="c.md")

    assert mode == ToolVolumeMode(max_tool_calls=2)


@pytest.mark.parametrize(
    ("most", "error", "said"),
    [
        (None, TypeError, "max_tool_calls must be an integer."),
        (True, TypeError, "max_tool_calls must be an integer."),
        (0, ValueError, "tool_volume mode requires max_tool_calls >= 1"),
    ],
)
def test_the_mode_needs_a_bound_of_one_or_more(
    most: object,
    error: type[Exception],
    said: str,
) -> None:
    meta = {} if most is None else {"max_tool_calls": most}

    with pytest.raises(error, match=f"^c\\.md: {said}$"):
        _ = ToolVolumeMode.parse(meta, {}, where="c.md")


@pytest.mark.parametrize(("calls", "flagged"), [(1, False), (2, True)])
def test_a_trial_working_in_its_folder_flags_calls_past_the_bound(
    calls: int,
    flagged: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = ScriptedProvider(
        replies=[
            reply(calls=[call("List", f"c{i}", path=".") for i in range(calls)]),
            reply("yes"),
        ],
    )
    model = provider["m"]
    mode = ToolVolumeMode(max_tool_calls=1)
    case = Case(
        name="c",
        n=1,
        max_flag_rate=0.0,
        mode=mode,
        producer_task="Can we?",
        files={"widget.py": "PLACEHOLDER = 1\n"},
    )
    run = Run(model=model, judge_model=model, system="S", cli_context=False)
    caplog.set_level(
        "INFO",
        logger="sagent.scripts.prompt_debug.modes.tool_volume",
    )

    assert asyncio.run(mode.trial(case, run)) is flagged

    assert caplog.messages == [f"tool_volume trial: {calls} calls (max 1)"]
    listed = provider.requests[1].records[-1]
    assert isinstance(listed, ToolMessage)
    assert "widget.py" in listed.text()
    tools = {t.name for t in provider.requests[0].tools}
    assert tools == {"Read", "Grep", "Glob", "List", "Bash", "Skill"}


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
