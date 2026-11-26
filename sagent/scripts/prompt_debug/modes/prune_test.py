"""Tests for the ``prune`` mode, on scripted models."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, cast

import asyncio
import re

import pytest

from sagent.agent.agent import NoShrinker
from sagent.agent.compaction.pruner import Pruner
from sagent.journal.types import ToolMessage, UserMessage
from sagent.model.capability import PruningPolicy
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.scripts.prompt_debug.harness import Run, load_cases
from sagent.scripts.prompt_debug.modes import prune
from sagent.scripts.prompt_debug.modes.prune import PruneMode
from sagent.tools.agent_session import AgentSession


if TYPE_CHECKING:
    from sagent.agent.types import Factory, Shrinker
    from sagent.model.scripted import Reply
    from sagent.model.types import Request, Response


@pytest.mark.parametrize(
    ("name", "kind"),
    [(None, NoShrinker), ("none", NoShrinker), ("shipped", Pruner)],
)
def test_the_mode_reads_its_regex_and_pruner(
    name: str | None,
    kind: Factory[Shrinker],
) -> None:
    meta: dict[str, object] = {"flag_regex": "STYLE"}
    if name is not None:
        meta["pruner"] = name

    assert PruneMode.parse(meta, {}, where="c.md") == PruneMode(
        flag_regex="STYLE",
        pruner=kind,
    )


def test_a_cold_pruner_is_cold_once_answered_and_lets_all_but_the_last_go() -> None:
    mode = PruneMode.parse({"flag_regex": "x", "pruner": "cold"}, {}, where="c.md")

    assert isinstance(mode.pruner, partial)
    assert mode.pruner.func is Pruner
    clock = cast(Callable[[], float], mode.pruner.keywords["clock"])
    first = clock()
    assert clock() - first == 1e9
    policy = cast(PruningPolicy, mode.pruner.keywords["policy"])
    assert policy == PruningPolicy(keep_recent=1, expiry=0.0)


@pytest.mark.parametrize(
    ("meta", "said"),
    [
        ({}, "prune mode requires a non-empty flag_regex"),
        ({"flag_regex": 1}, "prune mode requires a non-empty flag_regex"),
        ({"flag_regex": "("}, "invalid flag_regex '\\('"),
        (
            {"flag_regex": "x", "pruner": "auto"},
            "pruner must be none, shipped, or cold",
        ),
    ],
)
def test_the_mode_refuses_what_it_cannot_run(
    meta: dict[str, object],
    said: str,
) -> None:
    with pytest.raises(ValueError, match=f"^c\\.md: {said}"):
        _ = PruneMode.parse(meta, {}, where="c.md")


def _folder(request: Request) -> str:
    """Return the trial's folder, as the first turn names it."""
    turn = next(r for r in request.records if isinstance(r, UserMessage))
    match = re.search(r"\S*prompt_debug_\w+", turn.content or "")
    assert match is not None
    return match[0]


def _reading(request: Request) -> Response:
    """Read the rule files and one module, as the setup turn asks."""
    root = _folder(request)
    return reply(
        calls=[
            call("Read", name, file_path=f"{root}/{name}")
            for name in ("AGENTS.md", "STYLE.md", "orders.py")
        ],
    )


def _queueing(start: str) -> Callable[[Request], Response]:
    """Return a reply that queues the read showing ``start``, by the id a list gave it."""

    def queue(request: Request) -> Response:
        listed = next(
            r.text()
            for r in reversed(request.records)
            if isinstance(r, ToolMessage) and r.call_id == "list"
        )
        found = cast(
            list[str],
            re.findall(rf"^#(\d+) .*: 1 {start}", listed, re.MULTILINE),
        )
        ids = [int(i) for i in found]
        assert len(ids) == 1
        return reply(
            calls=[
                call(
                    "AgentSession",
                    "queue",
                    action="prune_queue",
                    ids=ids,
                    reason="spent",
                ),
            ],
        )

    return queue


@pytest.mark.parametrize(
    ("queued", "flagged"),
    [("# Style Guide", True), ("def orders_0", False), ("", False)],
    ids=["a rule queued", "a module queued", "nothing queued"],
)
def test_a_trial_flags_a_rule_file_queued_to_drop(
    queued: str,
    flagged: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    steps: list[Reply] = [_reading, reply("read")]
    if queued:
        steps += [
            reply(calls=[call("AgentSession", "list", action="list")]),
            _queueing(queued),
        ]
    provider = ScriptedProvider(replies=[*steps, reply("orders_11, 12x")])
    model = provider["m"]
    case = next(c for c in load_cases() if c.name == "prune-keeps-rules")
    mode = PruneMode(flag_regex=r"STYLE\.md$", pruner=NoShrinker)
    run = Run(model=model, judge_model=model, system="S", cli_context=False)
    caplog.set_level("INFO", logger=prune.__name__)

    assert asyncio.run(mode.trial(case, run)) is flagged

    (said,) = caplog.messages
    assert said.endswith("; said [None, 'spent']" if queued else "; said []")
    names = {t.name for t in provider.requests[0].tools}
    assert names == {"Read", "Edit", "Grep", "Glob", "AgentSession"}


def test_a_trial_seeds_its_folder_and_describes_its_session_tool_as_told() -> None:
    provider = ScriptedProvider(
        replies=[reply(calls=[call("Glob", pattern="*")]), reply("ok"), reply("ok")],
    )
    model = provider["m"]
    case = next(c for c in load_cases() if c.name == "prune-keeps-rules")
    mode = PruneMode(flag_regex="x", pruner=NoShrinker, session_description="Old.")
    run = Run(model=model, judge_model=model, system="S", cli_context=False)

    assert not asyncio.run(mode.trial(case, run))

    listed = provider.requests[1].records[-1]
    assert isinstance(listed, ToolMessage)
    found = listed.text()
    for name in ("AGENTS.md", "STYLE.md", "audit.py", "ledger.py", "orders.py"):
        assert name in found
    (tool,) = [t for t in provider.requests[0].tools if t.name == AgentSession.name]
    assert tool.description == "Old."
    assert prune._source("audit").count("def audit_") == 12


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
