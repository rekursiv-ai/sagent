"""Tests for the prune-note experiment's arms, with no live model."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, override

import asyncio
import logging

import pytest

from sagent.agent.agent import Agent, NoShrinker
from sagent.agent.compaction.pruner import Pruner
from sagent.journal.types import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from sagent.model.scripted import ScriptedProvider
from sagent.scripts.prompt_debug.experiments import prune_note
from sagent.scripts.prompt_debug.harness import CaseResult, load_cases
from sagent.scripts.prompt_debug.modes.prune import PruneMode


if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

    from sagent.scripts.prompt_debug.harness import Case


def _ran(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Case, dict[str, object]]]:
    """Stub the experiment's ``run_case``; return what each call was given."""
    seen: list[tuple[Case, dict[str, object]]] = []

    async def run_case(case: Case, **kwargs: object) -> CaseResult:
        seen.append((case, kwargs))
        return CaseResult(case=case, verdicts=(True, False))

    monkeypatch.setattr(prune_note, "run_case", run_case)
    return seen


@pytest.mark.parametrize("arm", ["auto", "note", "old"])
def test_each_arm_is_the_guard_case_but_its_pruner_ask_and_description(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    seen = _ran(monkeypatch)
    model = ScriptedProvider()["scripted"]
    guard = next(c for c in load_cases() if c.name == "prune-keeps-rules")
    assert isinstance(guard.mode, PruneMode)

    result = asyncio.run(prune_note.run(arm, n=2, model=model))

    ((case, kwargs),) = seen
    assert result.flagged == 1
    assert kwargs == {
        "model": model,
        "max_concurrency": 2,
        "progress": prune_note._progress,
        "stop_early": False,
    }
    assert (case.name, case.n) == (f"prune-note-{arm}", 2)
    assert (case.files, case.setup_turn) == (guard.files, guard.setup_turn)
    assert isinstance(case.mode, PruneMode)
    assert case.mode.flag_regex == guard.mode.flag_regex
    asked = arm == "old"
    assert case.producer_task == (
        "Your context is getting large. Prune the tool results you no longer need,"
        " then tell me which function in {DIR}/audit.py weights its rows the most,"
        " and by how much?"
        if asked
        else guard.producer_task
    )
    assert case.mode.session_description == (
        prune_note._old_description() if asked else ""
    )
    pruner, cold = case.mode.pruner, guard.mode.pruner
    if asked:
        assert pruner is NoShrinker
        return
    assert isinstance(pruner, partial)
    assert isinstance(cold, partial)
    kind = {"auto": prune_note._OldPruner, "note": Pruner}[arm]
    assert (pruner.func, pruner.keywords["policy"]) == (kind, cold.keywords["policy"])


def test_a_guard_that_is_no_prune_case_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = next(c for c in load_cases() if c.name == "adjudicate-user-proposal")

    def parse_case(path: Traversable) -> Case:
        return replace(text, name=path.name)

    monkeypatch.setattr(prune_note, "parse_case", parse_case)

    with pytest.raises(
        TypeError,
        match=r"^prune_keeps_rules\.md is not a prune case\.$",
    ):
        _ = asyncio.run(prune_note.run("note", n=1, model=ScriptedProvider()["s"]))


def test_an_unknown_arm_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _ran(monkeypatch)

    with pytest.raises(
        ValueError,
        match=r"^arm must be auto, note, or old, got 'new'\.$",
    ):
        _ = asyncio.run(prune_note.run("new", n=1, model=ScriptedProvider()["s"]))
    assert seen == []


class _Picking(prune_note._OldPruner):
    """The old pruner, its cost rule picking the first tool result."""

    @override
    def _plan(self) -> list[int]:
        context = self._agent.runtime.tape.context().messages
        return [next(i for i, e in enumerate(context) if isinstance(e, ToolMessage))]


def test_the_old_pruner_drops_what_its_rule_picks_and_puts_off_nothing() -> None:
    agent = Agent(model=ScriptedProvider()["scripted"], pruner=_Picking)
    agent.session.append(
        UserMessage(content="go"),
        ToolCall(call_id="c0", name="Read"),
        ToolMessage(call_id="c0", state="done", stdout=("x" * 400,)),
        AssistantMessage(content="read"),
    )
    pruner = agent.session.pruner
    assert isinstance(pruner, Pruner)

    assert not pruner.prune()

    (result,) = [r for r in agent.session if isinstance(r, ToolMessage)]
    assert result.text() == agent.runtime.stubs.pruned


def test_progress_logs_each_trial(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=prune_note.__name__)

    prune_note._progress(1, 8, True)

    assert caplog.messages == ["1/8 flagged=True"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
