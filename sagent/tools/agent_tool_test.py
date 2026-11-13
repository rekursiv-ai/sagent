"""Tests for agent tools targeting another agent with the caller's authority."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, override
from unittest.mock import MagicMock, Mock

import asyncio

import pytest

from sagent.agent.agent import Agent
from sagent.agent.compactor import SummaryCompactor
from sagent.agent.team import Team
from sagent.model.capability import ModelCapability
from sagent.model.scripted import ScriptedModel, ScriptedProvider, call, reply
from sagent.providers.catalog import ModelCatalog
from sagent.runtime import types
from sagent.runtime.context import validate_context
from sagent.session.types import ToolResult, UncategorizedToolResult, UserMessage
from sagent.tools.agent_self import AgentSelf
from sagent.tools.agent_session import AgentSession
from sagent.tools.agent_tool import target_agent
from sagent.tools.types import ToolContext, settled


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from treekle import PlainTree

    from sagent.model.types import Request, Response, TextChunk


def _setup(tmp_path: Path) -> tuple[Team, AgentSelf, AgentSession]:
    catalog = ModelCatalog(
        factories={
            "scripted": partial(
                ScriptedProvider,
                capability=ModelCapability(
                    model_id="scripted",
                    thinking_effort=frozenset({"none", "high"}),
                ),
            ),
        },
    )
    team = Team(catalog=catalog)
    control, session = AgentSelf(team=team), AgentSession(team=team)
    for label, parent in (
        ("root", None),
        ("child", "root"),
        ("grand", "child"),
        ("sibling", "child"),
        ("peer", None),
    ):
        agent = Agent(
            model=catalog["scripted"]["scripted"],
            tools=[control, session] if label == "root" else [],
            tool_context=ToolContext(cwd=tmp_path / label),
            compactor=SummaryCompactor,
            environment=False,
        )
        team.join(agent, label=label, parent=parent)
        agent.session.append(UserMessage(content=f"{label}'s words"))
    return team, control, session


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["", "root", "child", "grand"])
async def test_state_targets_self_or_descendants_without_granting_tools(
    tmp_path: Path,
    target: str,
) -> None:
    team, control, _ = _setup(tmp_path)
    result = await settled(
        control.call(
            team["root"].runtime.tool_context,
            {"agent": target, "effort": "high", "diagnostics": True},
        ),
    )
    label = target or "root"
    assert f"label: {label}" in result.text()
    assert f"cwd: {tmp_path / label}" in result.text()
    assert [a.model.settings.thinking_effort for a in team.values()] == [
        "high" if name == label else "none" for name in team
    ]
    assert list(team["grand"].tools) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["list", "read", "search"])
@pytest.mark.parametrize("target", ["peer", "parent", "sibling"])
async def test_session_reads_reach_any_team_member(
    tmp_path: Path,
    action: str,
    target: str,
) -> None:
    def resolve(pattern: str, *, by: str) -> list[str]:
        del by
        return {"parent": ["child"], "descendants": list[str]()}.get(pattern, [pattern])

    label = "child" if target == "parent" else target
    record = UserMessage(content=f"{label}'s words")
    contexted = Mock(return_value=Mock(messages=[record]))
    history = Mock(return_value=[record])
    selected = Mock(
        spec=Agent,
        runtime=Mock(context=contexted),
        session=Mock(history=history),
    )
    resolved = Mock(side_effect=resolve)
    fetched = Mock(return_value=selected)
    team = MagicMock(
        spec=Team,
        __contains__=Mock(side_effect={"grand", label}.__contains__),
        resolve=resolved,
        __getitem__=fetched,
    )
    arguments: dict[str, PlainTree] = {
        "agent": target,
        "action": action,
        "ids": [0],
        "text": "words",
    }
    result = await settled(
        AgentSession(team=team).call(
            ToolContext(cwd=tmp_path / "grand", label="grand"),
            arguments,
        ),
    )
    assert f"{label}'s words" in result.text()
    resolved.assert_called_once_with(target, by="grand")
    fetched.assert_called_once_with(label)
    contexted.assert_called_once_with()
    assert history.call_args_list == ([((), {})] if action == "search" else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["clear", "compact", "prune", "estimate"])
async def test_session_changes_refuse_unrelated_agents(
    tmp_path: Path,
    action: str,
) -> None:
    team, _, session = _setup(tmp_path)
    with pytest.raises(ValueError, match=r"^Mutations require self or a descendant\.$"):
        await session.call(
            team["root"].runtime.tool_context,
            {"agent": "peer", "action": action},
        ).result
    assert team["peer"].runtime.inbox.drain_nowait() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["clear", "compact", "prune"])
async def test_session_mutation_queues_on_the_target_only(
    tmp_path: Path,
    action: str,
) -> None:
    team, _, session = _setup(tmp_path)
    target = team["grand"]
    target.session.append(
        ToolResult(
            call_id="old",
            state="done",
            stdout=("old output",),
            result=UncategorizedToolResult(),
        ),
    )
    arguments: dict[str, PlainTree] = {"agent": "grand", "action": action}
    if action == "prune":
        arguments.update({"ids": [1], "reason": "obsolete"})
    await settled(session.call(team["root"].runtime.tool_context, arguments))
    await asyncio.sleep(0)
    (queued,) = target.runtime.inbox.drain_nowait()
    if action == "prune":
        assert isinstance(queued, types.Scheduled)
        await queued.work()
        results = [r for r in target.session if isinstance(r, ToolResult)]
        assert results[0].text().startswith("[pruned:")
    else:
        assert isinstance(queued, types.Clear if action == "clear" else types.Compact)
    assert team["root"].runtime.inbox.drain_nowait() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["peer", "parent", "sibling"])
async def test_state_changes_refuse_ancestors_siblings_and_peers(
    tmp_path: Path,
    target: str,
) -> None:
    team, control, _ = _setup(tmp_path)
    with pytest.raises(ValueError, match=r"^Mutations require self or a descendant\.$"):
        await control.call(
            team["grand"].runtime.tool_context,
            {"agent": target, "effort": "high"},
        ).result
    assert all(a.model.settings.thinking_effort == "none" for a in team.values())


def test_targeting_requires_a_calling_team_member(tmp_path: Path) -> None:
    contains = Mock(side_effect={"child"}.__contains__)
    resolved = Mock(return_value=["child"])
    fetched = Mock(return_value=Mock(spec=Agent))
    team = MagicMock(
        spec=Team,
        __contains__=contains,
        resolve=resolved,
        __getitem__=fetched,
    )
    context = ToolContext(cwd=tmp_path, label="outsider")
    for name in ("AgentSelf", "AgentSession"):
        with pytest.raises(
            ValueError,
            match=rf"^{name} needs a calling agent in the team\.$",
        ):
            target_agent(team, context=context, name=name, agent="child")
    assert contains.call_args_list == [(("outsider",), {}), (("outsider",), {})]
    resolved.assert_not_called()
    fetched.assert_not_called()


@pytest.mark.parametrize("name", ["missing", "descendants"])
def test_targets_must_name_one_agent(tmp_path: Path, name: str) -> None:
    labels: list[str] = [] if name == "missing" else ["child", "grand"]
    resolved = Mock(return_value=labels)
    fetched = Mock()
    team = MagicMock(
        spec=Team,
        __contains__=Mock(return_value=True),
        resolve=resolved,
        __getitem__=fetched,
    )
    context = ToolContext(cwd=tmp_path, label="root")
    for tool in ("AgentSelf", "AgentSession"):
        with pytest.raises(
            ValueError,
            match=rf"^{name!r} names {len(labels)} agents; name one\.$",
        ):
            target_agent(team, context=context, name=tool, agent=name)
    assert resolved.call_args_list == [
        ((name,), {"by": "root"}),
        ((name,), {"by": "root"}),
    ]
    fetched.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["AgentSelf", "AgentSession"])
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("permitted", [False, True])
async def test_runtime_admission_holds_across_the_round_cap(
    tmp_path: Path,
    name: str,
    background: bool,
    permitted: bool,
) -> None:
    team, _, _ = _setup(tmp_path)
    caller = team["root"]
    caller.max_rounds = 1
    caller.allowed = frozenset({name}) if permitted else frozenset()
    provider = caller.model.provider
    assert isinstance(provider, ScriptedProvider)
    arguments: dict[str, PlainTree] = (
        {"effort": "high"} if name == "AgentSelf" else {"action": "list"}
    )
    arguments.update({"agent": "grand", "background": background})
    provider.replies.extend(
        [reply(calls=[call(name, "probe", **arguments)]), *[reply("finished")] * 3],
    )
    async with caller:
        assert await asyncio.wait_for(caller.send("probe"), timeout=5) == "finished"
        arrived = (
            await asyncio.wait_for(caller.runtime.foreground("probe"), timeout=5)
            if background
            else None
        )
    results = [
        r for r in caller.session if isinstance(r, ToolResult) and r.call_id == "probe"
    ]
    result = arrived if arrived is not None else results[-1]
    if permitted:
        assert result.is_error is False, result.text()
        assert (
            "thinking_effort" in result.text()
            if name == "AgentSelf"
            else "grand's words" in result.text()
        )
    else:
        assert "not available" in result.text()
    assert team["grand"].model.settings.thinking_effort == (
        "high" if permitted and name == "AgentSelf" else "none"
    )


@pytest.mark.asyncio
async def test_a_watcher_is_not_a_mutation_target(tmp_path: Path) -> None:
    team, control, session = _setup(tmp_path)
    team.watch("peer", "root")
    with pytest.raises(ValueError, match=r"^Mutations require self or a descendant\.$"):
        await control.call(
            team["root"].runtime.tool_context,
            {"agent": "parent", "effort": "high"},
        ).result
    result = await settled(
        session.call(
            team["root"].runtime.tool_context,
            {"agent": "parent", "action": "list"},
        ),
    )
    assert "peer's words" in result.text()


@pytest.mark.asyncio
async def test_target_status_events_publish_on_the_target_only(tmp_path: Path) -> None:
    team, control, _ = _setup(tmp_path)
    root_events: list[types.RuntimeEvent] = []
    child_events: list[types.RuntimeEvent] = []
    team["root"].runtime.observers.append(root_events.append)
    team["grand"].runtime.observers.append(child_events.append)
    await settled(
        control.call(
            team["root"].runtime.tool_context,
            {"agent": "grand", "status": "Checking the target state"},
        ),
    )
    assert child_events == [types.StatusChanged(text="Checking the target state")]
    assert root_events == []


@pytest.mark.asyncio
async def test_invalid_target_call_does_not_drop_a_sibling_call(tmp_path: Path) -> None:
    team, _, _ = _setup(tmp_path)
    caller = team["root"]
    provider = caller.model.provider
    assert isinstance(provider, ScriptedProvider)
    provider.replies.extend(
        [
            reply(
                calls=[
                    call("AgentSession", "bad", agent="missing", action="list"),
                    call("AgentSession", "good", agent="grand", action="list"),
                ],
            ),
            reply("recovered"),
        ],
    )
    async with caller:
        assert await asyncio.wait_for(caller.send("probe"), timeout=5) == "recovered"
    results = {r.call_id: r for r in caller.session if isinstance(r, ToolResult)}
    assert results["bad"].state == "failed"
    assert "names 0 agents" in results["bad"].text()
    assert "grand's words" in results["good"].text()
    validate_context(caller.runtime.context().messages)


@pytest.mark.asyncio
async def test_invalid_target_patch_leaves_both_agents_unchanged(
    tmp_path: Path,
) -> None:
    team, control, _ = _setup(tmp_path)
    models = [a.model for a in team.values()]
    with pytest.raises(ValueError, match="not offered"):
        await control.call(
            team["root"].runtime.tool_context,
            {"agent": "grand", "model": "scripted", "output": "redacted"},
        ).result
    assert [a.model for a in team.values()] == models


@pytest.mark.asyncio
async def test_peer_messaging_changes_the_target_only(tmp_path: Path) -> None:
    team, control, _ = _setup(tmp_path)
    await settled(
        control.call(
            team["root"].runtime.tool_context,
            {"agent": "grand", "peer_messaging": False},
        ),
    )
    assert team.peer_messaging("root")
    assert not team.peer_messaging("grand")


class _HeldModel(ScriptedModel):
    """A model held inside a live request until the test releases it."""

    def __init__(self, provider: ScriptedProvider) -> None:
        super().__init__(
            provider=provider,
            capability=provider.capability,
            settings=provider.capability.default,
        )
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    @override
    async def stream(self, request: Request) -> AsyncIterator[TextChunk | Response]:
        self.started.set()
        await self.release.wait()
        async for event in super().stream(request):
            yield event

    @override
    async def aclose(self) -> None:
        self.closed = True
        await super().aclose()


@pytest.mark.asyncio
async def test_switching_a_running_descendant_keeps_its_active_request_alive(
    tmp_path: Path,
) -> None:
    team, control, _ = _setup(tmp_path)
    child = team["grand"]
    provider = child.model.provider
    assert isinstance(provider, ScriptedProvider)
    held = _HeldModel(provider)
    child.model = held
    provider.replies.extend([reply("old request"), reply("new request")])
    root_model = team["root"].model
    async with child:
        running = asyncio.create_task(child.send("first"))
        await asyncio.wait_for(held.started.wait(), timeout=5)
        await settled(
            control.call(
                team["root"].runtime.tool_context,
                {"agent": "grand", "model": "scripted", "effort": "high"},
            ),
        )
        assert child.model is not held
        assert child.model.settings.thinking_effort == "high"
        assert not held.closed
        held.release.set()
        assert await asyncio.wait_for(running, timeout=5) == "old request"
        assert await asyncio.wait_for(child.send("second"), timeout=5) == "new request"
    assert held.closed
    assert team["root"].model is root_model


def test_session_target_must_be_local(tmp_path: Path) -> None:
    contains = Mock(side_effect={"root", "child"}.__contains__)
    resolved = Mock(return_value=["far"])
    fetched = Mock()
    team = MagicMock(
        spec=Team,
        __contains__=contains,
        resolve=resolved,
        __getitem__=fetched,
    )
    with pytest.raises(ValueError, match=r"^Agent 'far' is not local to this team\.$"):
        target_agent(
            team,
            context=ToolContext(cwd=tmp_path, label="root"),
            name="AgentSession",
            agent="far",
            reading=True,
        )
    assert contains.call_args_list == [(("root",), {}), (("far",), {})]
    resolved.assert_called_once_with("far", by="root")
    fetched.assert_not_called()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
