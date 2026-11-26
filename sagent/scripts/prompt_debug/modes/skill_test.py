"""Tests for the ``skill`` mode, on scripted models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio

import pytest

from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.scripts.prompt_debug.harness import Case, Run
from sagent.scripts.prompt_debug.modes.skill import SkillMode


if TYPE_CHECKING:
    from sagent.journal.types import ToolCall


def test_the_mode_reads_the_skill_it_expects() -> None:
    mode = SkillMode.parse({"expect_skill": "debug"}, {}, where="c.md")

    assert mode == SkillMode(expect_skill="debug")


@pytest.mark.parametrize("skill", [None, "", 1])
def test_the_mode_needs_a_skill_named(skill: object) -> None:
    meta = {} if skill is None else {"expect_skill": skill}

    with pytest.raises(
        ValueError,
        match=r"^c\.md: skill mode requires a non-empty expect_skill$",
    ):
        _ = SkillMode.parse(meta, {}, where="c.md")


@pytest.mark.parametrize(
    ("called", "flagged"),
    [
        ([call("Skill", skill="debug")], False),
        ([call("Skill", skill="tdd")], True),
        ([call("Glob", pattern="*.py")], True),
        ([], True),
    ],
    ids=["the skill", "another skill", "another tool", "nothing"],
)
def test_a_trial_flags_when_the_skill_never_fired(
    called: list[ToolCall],
    flagged: bool,
) -> None:
    replies = [reply(calls=called), reply("done")] if called else [reply("done")]
    provider = ScriptedProvider(replies=replies)
    model = provider["m"]
    mode = SkillMode(expect_skill="debug")
    case = Case(name="c", n=1, max_flag_rate=0.0, mode=mode, producer_task="Fix it.")
    run = Run(model=model, judge_model=model, system="S", cli_context=False)

    assert asyncio.run(mode.trial(case, run)) is flagged
    tools = {t.name for t in provider.requests[0].tools}
    assert tools == {"Read", "Edit", "Glob", "Grep", "Skill"}


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
