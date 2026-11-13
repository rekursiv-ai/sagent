"""Ownership and lifecycle for a live, UI-neutral Sagent session."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Self, cast

from wesearch.fetch import FetchSession

from sagent.agent.agent import Agent, default_system_prompt
from sagent.agent.agents_md import AgentsMdConfig
from sagent.agent.compactor import SummaryCompactor
from sagent.agent.default_tools import build_tools, preload_builtin_tools
from sagent.agent.pruner import Pruner
from sagent.agent.team import Team
from sagent.providers.catalog import ModelCatalog, default_provider
from sagent.session.controller import (
    AgentSessionController,
    Submission,
    SubmissionMode,
)
from sagent.session.jsonl import JsonlJournal


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from sagent.runtime import types
    from sagent.session.types import Attachment


__all__ = ["AgentSession"]


class AgentSession:
    """Own one live native Sagent team and its async resources.

    Frontends own this object while it is open.  The object owns the native
    catalog, team, fetch session, journal, and agent resources; it does not own
    rendering, transport, or frontend state.
    """

    def __init__(
        self,
        *,
        resources: AsyncExitStack,
        catalog: ModelCatalog,
        team: Team,
        agent: Agent,
        path: Path | None,
        controller: AgentSessionController,
        owns_resources: bool,
    ) -> None:
        self._resources = resources
        self.catalog = catalog
        self.team = team
        self.agent = agent
        self.path = path
        self.controller = controller
        self._owns_resources = owns_resources
        self._closed = False

    @classmethod
    async def open(
        cls,
        *,
        cwd: Path,
        path: Path,
        model_name: str,
        allowed_providers: str = "",
        tool_names: Sequence[str] | None = None,
        system_prompt: str | None = None,
        append_system_prompt: str = "",
        skill_directories: Sequence[Path] = (),
    ) -> AgentSession:
        """Open a journal-backed native session and return its owner."""
        preload_builtin_tools()
        resources = AsyncExitStack()
        await resources.__aenter__()
        try:
            catalog = await resources.enter_async_context(ModelCatalog())
            if allowed_providers:
                catalog.allow(_comma_list(allowed_providers))
            team = await resources.enter_async_context(Team(catalog=catalog))
            fetch = await resources.enter_async_context(FetchSession())
            journal = await resources.enter_async_context(JsonlJournal(path))
            model = catalog.model(model_name, current=default_provider())
            system = system_prompt
            if append_system_prompt:
                base = system if system is not None else default_system_prompt()
                system = f"{base}\n\n{append_system_prompt}"

            agent = Agent(
                model=model,
                tools=build_tools(
                    cwd=cwd,
                    team=team,
                    session=fetch,
                    names=cast("Sequence[str] | None", tool_names),
                    skill_directories=skill_directories,
                ),
                journal=journal,
                compactor=SummaryCompactor,
                pruner=Pruner,
                cwd=cwd,
                agents_md=AgentsMdConfig(),
                system_prompt=system,
                persistent_retry=True,
            )
            await agent.recall(catalog)
            await resources.enter_async_context(agent)
            team.join(agent, label="Agent")
            controller = AgentSessionController(team)
            return cls(
                resources=resources,
                catalog=catalog,
                team=team,
                agent=agent,
                path=path,
                controller=controller,
                owns_resources=True,
            )
        except BaseException:
            await resources.aclose()
            raise

    async def close(self) -> None:
        """Release native resources and remove controller observers."""
        if self._closed:
            return
        self._closed = True
        self.controller.unsubscribe_all()
        if self._owns_resources:
            await self._resources.aclose()

    async def __aenter__(self) -> Self:
        """Return this already-open native session."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the session at the end of an async context."""
        await self.close()

    @property
    def agents(self) -> Mapping[str, Agent]:
        """Return the live native agents without exposing controller state."""
        return self.controller.agents

    def submit(
        self,
        text: str,
        *,
        agent_label: str = "Agent",
        mode: SubmissionMode = "immediate",
        attachments: tuple[Attachment, ...] = (),
    ) -> Submission:
        """Submit user input using native runtime inbox semantics."""
        return self.controller.send(
            text,
            agent_label=agent_label,
            mode=mode,
            attachments=attachments,
        )

    def halt(self, *, agent_label: str = "Agent") -> bool:
        """Halt one native agent and report whether it was active."""
        return self.controller.halt(agent_label=agent_label)

    def subscribe(
        self,
        observer: Callable[[str, types.RuntimeEvent], None],
        *,
        agent_label: str | None = None,
    ) -> Callable[[], None]:
        """Subscribe to native events with agent labels."""
        return self.controller.subscribe(observer, agent_label=agent_label)

    def send_agent_message(
        self,
        *,
        sender: str,
        recipient: str,
        content: str,
    ) -> None:
        """Send an agent-to-agent message through native team bookkeeping."""
        self.team.send(sender, recipient, content)

    @classmethod
    def attach(
        cls,
        *,
        catalog: ModelCatalog,
        team: Team,
        agent: Agent,
        path: Path | None = None,
    ) -> AgentSession:
        """Adapt an already-configured native session for another frontend.

        ``attach`` is deliberately non-owning.  This is for frontends such as
        the TUI that configure extra native resources—MCP servers, custom tool
        sets, or ephemeral sessions—before exposing the agent to the shared
        session surface.  The caller remains responsible for closing those
        resources.
        """
        return cls(
            resources=AsyncExitStack(),
            catalog=catalog,
            team=team,
            agent=agent,
            path=path,
            controller=AgentSessionController(team),
            owns_resources=False,
        )

    def _agent(self, label: str) -> Agent:
        try:
            return self.team[label]
        except KeyError:
            raise ValueError(f"unknown agent {label!r}") from None


def _comma_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]
