"""Tests for the ``text`` mode, on scripted models."""

from __future__ import annotations

import asyncio

import pytest

from sagent.journal.types import ToolMessage, UserMessage
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.scripts.prompt_debug.harness import Case, Run
from sagent.scripts.prompt_debug.modes.text import TextMode


def _mode(meta: dict[str, object], rubric: str = "R") -> TextMode:
    return TextMode.parse(meta, {"judge_rubric": f"\n{rubric}\n"}, where="c.md")


def test_the_mode_reads_its_rubric_and_tools() -> None:
    assert _mode({}) == TextMode(judge_rubric="R")
    assert _mode({"use_tools": True}) == TextMode(judge_rubric="R", use_tools=True)


@pytest.mark.parametrize(
    ("meta", "rubric", "error", "said"),
    [
        ({}, " ", ValueError, "c.md: text mode requires a non-empty judge_rubric"),
        ({"use_tools": 1}, "R", TypeError, "c.md: use_tools must be a boolean."),
    ],
)
def test_the_mode_refuses_a_missing_rubric_or_a_bad_flag(
    meta: dict[str, object],
    rubric: str,
    error: type[Exception],
    said: str,
) -> None:
    with pytest.raises(error, match=f"^{said}$"):
        _ = _mode(meta, rubric)


@pytest.mark.parametrize(
    ("verdict", "flagged"),
    [
        ("VERDICT: YES", True),
        ("verdict: yes", True),
        ("I'd say VERDICT: YES if verbose, but no.\nVERDICT: NO", False),
        ("VERDICT: NO\nVERDICT: YES", True),
    ],
)
def test_a_blind_judge_scores_the_answer_by_its_last_verdict(
    verdict: str,
    flagged: bool,
) -> None:
    provider = ScriptedProvider(replies=[reply("the answer"), reply(verdict)])
    producer, judge = provider["producer"], provider["judge"]
    run = Run(model=producer, judge_model=judge, system="S", cli_context=False)
    case = Case(name="c", n=1, max_flag_rate=0.0, mode=_mode({}), producer_task="Why?")

    assert asyncio.run(case.mode.trial(case, run)) is flagged

    asked, judged = provider.requests
    assert (asked.system[0], [t.name for t in asked.tools]) == ("S", [])
    assert "strict classifier" in judged.system[0]
    (said,) = [r for r in judged.records if isinstance(r, UserMessage)]
    assert said.content == "RUBRIC:\nR\n\nANSWER:\nthe answer"


def test_a_producer_with_tools_may_use_them_and_a_silent_judge_is_an_error() -> None:
    provider = ScriptedProvider(
        replies=[
            reply(calls=[call("List", path=".")]),
            reply("ok"),
            reply("no verdict"),
        ],
    )
    run = Run(
        model=provider["m"],
        judge_model=provider["m"],
        system="S",
        cli_context=False,
    )
    mode = _mode({"use_tools": True})
    case = Case(name="c", n=1, max_flag_rate=0.0, mode=mode, producer_task="Look.")

    with pytest.raises(
        ValueError,
        match=r"^judge emitted no VERDICT line: 'no verdict'$",
    ):
        _ = asyncio.run(mode.trial(case, run))

    tools = {t.name for t in provider.requests[0].tools}
    assert tools == {"Read", "Grep", "Glob", "List", "Bash"}
    assert any(isinstance(r, ToolMessage) for r in provider.requests[1].records)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
