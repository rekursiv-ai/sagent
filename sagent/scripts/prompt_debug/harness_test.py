"""Tests for the prompt-regression harness, on scripted models."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING

import asyncio
import sys

import pytest

from sagent.agent.agent import NoShrinker
from sagent.agent.agents_md import AgentsMdConfig
from sagent.agent.compaction.pruner import Pruner
from sagent.journal.types import UserMessage
from sagent.model.exceptions import DisconnectedError
from sagent.model.scripted import ScriptedProvider, reply
from sagent.scripts.prompt_debug import harness
from sagent.scripts.prompt_debug.harness import (
    Case,
    CaseResult,
    Run,
    converse,
    load_cases,
    parse_case,
    producer,
    run_case,
    workspace,
)
from sagent.scripts.prompt_debug.modes.code import CodeMode
from sagent.scripts.prompt_debug.modes.text import TextMode
from sagent.tools.read import Read
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from sagent.model.types import Response
    from sagent.providers.types import Model


_TEXT = """\
---
name = "verb"
n = 4
max_flag_rate = 0.25
layer = "ignored by every mode"
---
preamble, dropped
## setup_turn
First, in {DIR}.

## producer_task
Why is X
---
happening?

## judge_rubric
Score VERBOSE YES/NO.

## file: notes/a.txt
  indented
"""


@dataclass(slots=True, kw_only=True)
class _Scripted:
    """A mode whose trials return, or raise, what it was given, in turn."""

    outcomes: list[bool | Exception]
    seen: list[Run] = field(default_factory=list)
    live: int = 0
    peak: int = 0

    @classmethod
    def parse(
        cls,
        meta: Mapping[str, object],
        sections: Mapping[str, str],
        *,
        where: str,
    ) -> _Scripted:
        del sections, where
        return cls(outcomes=[meta["flag"] is True])

    async def trial(self, case: Case, run: Run) -> bool:
        del case
        self.seen.append(run)
        self.live += 1
        self.peak = max(self.peak, self.live)
        await asyncio.sleep(0)
        self.live -= 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "case.md"
    _ = path.write_text(text, encoding="utf-8")
    return path


def _model(*replies: Response) -> Model:
    return ScriptedProvider(replies=replies)["scripted"]


def _case(mode: _Scripted, *, n: int = 4, max_flag_rate: float = 0.25) -> Case:
    return Case(
        name="c",
        n=n,
        max_flag_rate=max_flag_rate,
        mode=mode,
        producer_task="t",
    )


def test_a_case_file_parses_into_its_turns_files_and_mode(tmp_path: Path) -> None:
    case = parse_case(_write(tmp_path, _TEXT))

    assert case == Case(
        name="verb",
        n=4,
        max_flag_rate=0.25,
        mode=TextMode(judge_rubric="Score VERBOSE YES/NO."),
        producer_task="Why is X\n---\nhappening?",
        setup_turn="First, in {DIR}.",
        files={"notes/a.txt": "indented\n"},
    )


def test_a_mode_named_by_its_class_is_imported(tmp_path: Path) -> None:
    text = _TEXT.replace("n = 4", f'n = 4\nflag = true\nmode = "{__name__}:_Scripted"')

    case = parse_case(_write(tmp_path, text))

    assert case.mode == _Scripted(outcomes=[True])


@pytest.mark.parametrize(
    ("before", "after", "said"),
    [
        (_TEXT, "no front", "case.md: missing TOML frontmatter."),
        (_TEXT, '---\nname = "x"\n', "case.md: missing frontmatter terminator."),
        ('name = "verb"', "", "case.md: name is required."),
        ('name = "verb"', 'name = " "', "case.md: name is required."),
        ("n = 4", "", "case.md: n must be an integer."),
        ("n = 4", "n = true", "case.md: n must be an integer."),
        ("max_flag_rate = 0.25", "", "case.md: max_flag_rate must be numeric."),
        ("max_flag_rate = 0.25", "max_flag_rate = true", "case.md: max_flag_rate must"),
        ("## producer_task", "## task", "case.md: producer_task section is required."),
        ("n = 4", "n = 4\nmode = 1", "case.md: mode must be a string."),
        ("n = 4", 'n = 4\nmode = "cdoe"', "case.md: mode must be text, code, skill,"),
        ("n = 4", 'n = 4\nmode = "no.such:Mode"', "case.md: mode 'no.such:Mode' not"),
        (
            "n = 4",
            f'n = 4\nmode = "{__name__}:_None"',
            f"case.md: mode '{__name__}:_None",
        ),
    ],
)
def test_a_malformed_case_names_its_file_and_fault(
    tmp_path: Path,
    before: str,
    after: str,
    said: str,
) -> None:
    with pytest.raises((TypeError, ValueError)) as caught:
        _ = parse_case(_write(tmp_path, _TEXT.replace(before, after, 1)))
    assert str(caught.value).startswith(said)


def test_the_built_in_cases_load() -> None:
    cases = {case.name: case for case in load_cases()}

    assert isinstance(cases["over-abstraction-delegators"].mode, CodeMode)
    assert cases["prune-keeps-rules"].files["AGENTS.md"].startswith("# House rules")


def test_cases_load_from_any_package_and_must_not_share_a_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "more_cases"
    package.mkdir()
    _ = (package / "__init__.py").write_text("", encoding="utf-8")
    _ = (package / "a.md").write_text(_TEXT, encoding="utf-8")
    _ = (package / "notes.txt").write_text("not a case", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "more_cases", raising=False)
    assert [case.name for case in load_cases("more_cases")] == ["verb"]
    _ = (package / "b.md").write_text(_TEXT, encoding="utf-8")

    with pytest.raises(ValueError, match=r"^Duplicate case name\(s\): verb\.$"):
        _ = load_cases("more_cases")


@pytest.mark.parametrize(
    ("verdicts", "rate", "passed"),
    [
        ((), 0.0, True),
        ((True, False, False, False), 0.25, True),
        ((True, True, False, False), 0.5, False),
        ((False,) * 3, 0.0, True),
    ],
)
def test_a_result_is_scored_against_n_not_the_trials_run(
    verdicts: tuple[bool, ...],
    rate: float,
    passed: bool,
) -> None:
    result = CaseResult(case=_case(_Scripted(outcomes=[])), verdicts=verdicts)

    assert (result.flagged, result.rate, result.passed) == (sum(verdicts), rate, passed)


def test_a_run_on_the_cli_prompt_tells_its_context_and_judges_on_its_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harness, "default_system_prompt", lambda: "CLI")
    mode = _Scripted(outcomes=[False])
    model = _model()

    _ = asyncio.run(run_case(_case(mode, n=1), model=model))

    assert mode.seen == [
        Run(model=model, judge_model=model, system="CLI", cli_context=True),
    ]


def test_a_run_on_a_given_prompt_tells_nothing_more() -> None:
    mode = _Scripted(outcomes=[False])
    model, judge = _model(), _model()

    _ = asyncio.run(
        run_case(_case(mode, n=1), model=model, judge_model=judge, producer_system="S"),
    )

    assert mode.seen == [
        Run(model=model, judge_model=judge, system="S", cli_context=False),
    ]


@pytest.mark.parametrize(
    ("outcomes", "ran"),
    [([True] * 8, 3), ([False] * 8, 6), ([True, False] * 4, 5)],
    ids=["certain fail", "certain pass", "mixed"],
)
def test_a_run_stops_once_its_outcome_is_certain(
    outcomes: list[bool | Exception],
    ran: int,
) -> None:
    events: list[tuple[int, int, bool]] = []
    mode = _Scripted(outcomes=list(outcomes))
    case = _case(mode, n=8)

    def progress(done: int, total: int, verdict: bool) -> None:
        events.append((done, total, verdict))

    result = asyncio.run(
        run_case(
            case,
            model=_model(),
            producer_system="S",
            max_concurrency=1,
            progress=progress,
        ),
    )

    assert result.verdicts == tuple(outcomes[:ran])
    assert events == [(i + 1, 8, v) for i, v in enumerate(result.verdicts)]
    full = asyncio.run(
        run_case(
            replace(case, mode=_Scripted(outcomes=list(outcomes))),
            model=_model(),
            producer_system="S",
            stop_early=False,
        ),
    )
    assert len(full.verdicts) == 8


def test_a_run_holds_its_trials_to_its_concurrency() -> None:
    bounded, default = (
        _Scripted(outcomes=[False] * 20),
        _Scripted(outcomes=[False] * 20),
    )

    _ = asyncio.run(
        run_case(
            _case(bounded, n=20, max_flag_rate=1.0),
            model=_model(),
            producer_system="S",
            stop_early=False,
            max_concurrency=3,
        ),
    )
    _ = asyncio.run(
        run_case(
            _case(default, n=20, max_flag_rate=1.0),
            model=_model(),
            producer_system="S",
            stop_early=False,
        ),
    )

    assert (bounded.peak, default.peak) == (3, 4)


def test_a_retryable_failure_retries_after_a_doubling_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def sleep(sec: float) -> None:
        slept.append(sec)

    outcomes: list[bool | Exception] = [
        DisconnectedError("a"),
        DisconnectedError("b"),
        True,
    ]
    monkeypatch.setattr(asyncio, "sleep", sleep)

    result = asyncio.run(harness._with_retries(partial(_next, outcomes), attempts=4))

    assert (result, slept, outcomes) == (True, [1.0, 2.0], [])


@pytest.mark.parametrize("attempts", [0, 1, 3])
def test_the_last_attempt_s_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
    attempts: int,
) -> None:
    slept: list[float] = []

    async def sleep(sec: float) -> None:
        slept.append(sec)

    errors: list[bool | Exception] = [DisconnectedError(str(i)) for i in range(4)]
    monkeypatch.setattr(asyncio, "sleep", sleep)

    with pytest.raises(DisconnectedError, match=f"^{max(attempts, 1) - 1}$"):
        _ = asyncio.run(
            harness._with_retries(partial(_next, errors), attempts=attempts),
        )
    assert slept == [2.0**k for k in range(attempts - 1)]


def test_any_other_failure_is_not_retried() -> None:
    mode = _Scripted(outcomes=[RuntimeError("broken"), True])

    with pytest.raises(RuntimeError, match=r"^broken$"):
        _ = asyncio.run(run_case(_case(mode, n=1), model=_model(), producer_system="S"))
    assert len(mode.outcomes) == 1


def test_a_producer_on_the_cli_prompt_gets_the_cli_context(tmp_path: Path) -> None:
    run = _run(cli_context=True)

    agent = producer(run, tools=[Read()], cwd=tmp_path, pruner=Pruner)

    assert (agent.agents_md, agent.environment) == (AgentsMdConfig(), True)
    assert (agent.model, agent.system_prompt) == (run.model, "S")
    assert list(agent.tools) == ["Read"]
    assert agent.runtime.tool_context.cwd == tmp_path
    assert isinstance(agent.session.pruner, Pruner)
    bare = producer(_run(), tools=[])
    assert (bare.agents_md.tiers, bare.environment) == (frozenset(), False)
    assert bare.runtime.tool_context.cwd == ToolContext().cwd
    assert isinstance(bare.session.pruner, NoShrinker)


def test_a_workspace_holds_the_case_s_files_until_it_closes() -> None:
    case = replace(
        _case(_Scripted(outcomes=[])),
        files={"a.py": "x = 1\n", "deep/er/b.md": "# b\n"},
    )

    with workspace(case) as root:
        assert root.name.startswith("prompt_debug_")
        found = {
            str(p.relative_to(root)): p.read_text(encoding="utf-8")
            for p in root.rglob("*")
            if p.is_file()
        }
    assert found == {"a.py": "x = 1\n", "deep/er/b.md": "# b\n"}
    assert not root.exists()


@pytest.mark.parametrize(("setup", "turns"), [("Set {DIR} up.", 2), ("", 1)])
def test_converse_sends_the_setup_then_the_task_naming_the_folder(
    tmp_path: Path,
    setup: str,
    turns: int,
) -> None:
    case = replace(
        _case(_Scripted(outcomes=[])),
        setup_turn=setup,
        producer_task="Do {DIR}/a.py.",
    )
    run = replace(_run(), model=_model(reply("set"), reply("done")))

    async def talk() -> tuple[str, list[str]]:
        async with producer(run, tools=[]) as agent:
            said = await converse(agent, case, tmp_path)
            return said, [
                r.content or ""
                for r in agent.session.history()
                if isinstance(r, UserMessage)
            ]

    said, sent = asyncio.run(talk())

    assert said == ("done" if setup else "set")
    assert sent == [f"Set {tmp_path} up.", f"Do {tmp_path}/a.py."][2 - turns :]


async def _next(outcomes: list[bool | Exception]) -> bool:
    outcome = outcomes.pop(0)
    if isinstance(outcome, Exception):
        raise outcome
    return outcome


def _run(*, cli_context: bool = False) -> Run:
    model = _model()
    return Run(model=model, judge_model=model, system="S", cli_context=cli_context)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
