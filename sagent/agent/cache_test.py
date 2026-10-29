"""The prompt cache holds: within a window, each request extends the one before.

A provider caches a request's prefix: its tools, its system prompt, then its
records in order. A request that changes any of these, rather than adding to
the end, pays to write the whole prompt again. The scripted provider keeps every
request, so a session that moves, reads below its root, changes its gate, its
reminders and its model settings is checked request against request.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import itertools
import shlex

from treekle import PlainTree, json_dumps

import pytest

from sagent.agent.agent import Agent
from sagent.agent.agents_md import AgentsMdConfig
from sagent.agent.compactor import SummaryCompactor
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.session.types import (
    AssistantMessage,
    ToolCall,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.bash import Bash
from sagent.tools.display import ToolDisplay
from sagent.tools.read import Read


if TYPE_CHECKING:
    from pathlib import Path

    from sagent.model.types import Request, Response
    from sagent.tools.types import Tool, ToolContext


@dataclass(slots=True, kw_only=True)
class _Reminding:
    """A tool whose reminder the test changes between rounds."""

    told: str = "first reminder"
    name: str = "Note"
    description: str = "Take a note."
    schema: Mapping[str, PlainTree] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": dict[str, PlainTree](),
        },
    )
    display: ToolDisplay = field(default_factory=ToolDisplay)

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return self.name

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    def reminder(self, context: ToolContext) -> str:
        del context
        return self.told

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        del context, arguments
        return UncategorizedToolResult(call_id="", content="noted")


def _offered(tools: Sequence[Tool]) -> list[str]:
    """Return each tool as the wire would carry it."""
    return [
        json_dumps({"name": t.name, "description": t.description, "schema": t.schema})
        for t in tools
    ]


def _then(
    response: Response,
    *,
    before: Sequence[Callable[[], None]],
) -> Callable[[Request], Response]:
    """Return a reply that runs ``before``'s steps as the model is asked."""

    def answer(request: Request) -> Response:
        del request
        for step in before:
            step()
        return response

    return answer


def _extends(requests: Sequence[Request]) -> None:
    """Assert each request keeps the one before as its prefix: tools, system, records."""
    for before, after in itertools.pairwise(requests):
        assert after.system == before.system
        assert _offered(after.tools) == _offered(before.tools)
        assert list(after.records[: len(before.records)]) == list(before.records)


@pytest.mark.asyncio
async def test_each_request_in_a_window_extends_the_one_before(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("root rule\n")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "AGENTS.md").write_text("pkg rule\n")
    (pkg / "a.py").write_text("x = 1\n")
    note = _Reminding()
    agent: Agent
    provider = ScriptedProvider()
    root = shlex.quote(str(tmp_path))

    def gate() -> None:
        agent.allowed = frozenset({"Bash", "Read"})

    def ungate() -> None:
        agent.allowed = None

    def remind_again() -> None:
        note.told = "second reminder"

    def withdraw() -> None:
        note.told = ""

    def think_harder() -> None:
        efforts = sorted(agent.model.capability.thinking_effort)
        agent.model.settings = replace(
            agent.model.settings,
            thinking_effort=efforts[-1],
        )

    provider.replies.extend(
        [
            reply(calls=[call("Read", "r1", file_path="pkg/a.py")]),
            _then(
                reply(calls=[call("Bash", "b1", command="cd pkg && echo in")]),
                before=[gate, remind_again],
            ),
            _then(
                reply(calls=[call("Bash", "b2", command=f"cd {root} && echo out")]),
                before=[withdraw, think_harder],
            ),
            _then(
                reply(calls=[call("Bash", "b3", command=f"cd {root} && cat pkg/a.py")]),
                before=[ungate],
            ),
            reply("first answer"),
            reply(calls=[call("Note", "n1")]),
            reply("second answer"),
            reply("<summary>the gist</summary>"),
            # The new window ends in the summary, which the model answers.
            reply("continuing"),
            reply(calls=[call("Read", "r2", file_path="pkg/a.py")]),
            reply("third answer"),
        ],
    )
    agent = Agent(
        model=provider["scripted"],
        tools=[Bash(peers=(Read(),)), Read(), note],
        system_prompt="S",
        cwd=tmp_path,
        agents_md=AgentsMdConfig(system_dir=tmp_path / "s", user_dir=tmp_path / "u"),
        compactor=SummaryCompactor,
    )
    compactor = agent.session.compactor
    assert isinstance(compactor, SummaryCompactor)

    async with agent:
        assert await agent.send("go") == "first answer"
        assert await agent.send("again") == "second answer"
        await compactor.run()
        assert await agent.send("after") == "third answer"

    requests = provider.requests
    assert len(requests) == 11
    # The summary is asked of the window's last request with one message more,
    # so it reads that request's cache; only the window after it starts afresh.
    first, summary, second = requests[:7], requests[7], requests[8:]
    _extends([*first, summary])
    _extends(second)
    asked = summary.records[len(first[-1].records) :]
    assert all(isinstance(r, (UserMessage, AssistantMessage)) for r in asked)
    assert not any(isinstance(r, ToolCall) for r in second[0].records)

    calls = [r.name for r in first[-1].records if isinstance(r, ToolCall)]
    assert calls == [
        "Read",
        "Bash",
        "Bash",
        "Read",
        "Note",
    ], "the cd to where the shell stood is dropped, and its cat read"
    added = [str(r) for r in first[-1].records]
    assert any("pkg rule" in r for r in added), "a rule read below the root is added"
    assert not any("pkg rule" in s for s in first[-1].system)
    again = [str(r) for r in second[-1].records]
    assert any("pkg rule" in r for r in again), "a new window reads it again"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
