"""AgentSpawn: start a child agent on a task, and return what it says.

A child inherits its parent's model, settings, tools, system prompt,
compactor, and directory, each overridable, and waits out a long rate limit
when its parent does. It joins the team under its
parent, so ``AgentSend`` reaches it and the depth cap holds. It runs its prompt
to idle, whose last text is the spawn's result, then stays, its inbox open,
until the team stops it; a message its parent sends it later is owed a reply.

Its events reach the parent's observers as ``ChildEvent``s, so the parent's
screen shows its work; a finished one ends with a ``ChildDoneEvent``. What its
model calls cost is charged to its parent and each agent above, journaled, and
held to their spend caps; it may spend no more than the least of them has left.

A child of a parent with a journal file stands in that journal until it is
stopped; ``resume`` starts such children again, each from its own
journal, when the parent's session is resumed. With ``watch``, the call starts
the caller's watcher instead, ``prompt`` its first request.

Typical usage example:

  spawn = AgentSpawn(team=team)
  result = await spawn.call(context, {"prompt": "list the TODOs in src/", "tools": ["Grep"]})
  await spawn.resume("Agent")          # After resuming the root's session.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from functools import partial
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import asyncio
import logging
import time

from sagent.agent.agent import Agent
from sagent.agent.team import RESERVED, ChildSpec, Team
from sagent.model.capability import ThinkingBudget, ThinkingEffort, ThinkingOutput
from sagent.model.types import Request
from sagent.providers.catalog import successor
from sagent.runtime import types
from sagent.runtime.exceptions import describe
from sagent.session.jsonl import JsonlJournal
from sagent.session.projects import child_journal, session_file
from sagent.session.types import (
    AgentStatusResult,
    AgentToAgentMessage,
    ToolCall,
    ToolResult,
)
from sagent.tools.display import ToolDisplay
from sagent.tools.skill import Skill
from sagent.tools.types import Arg, parse, schema_of
from sagent.tools.watch import Feed


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from treekle import PlainTree

    from sagent.providers.types import Model
    from sagent.tools.types import ToolContext


__all__ = [
    "AgentSpawn",
]


# The owners of slash settings: ``/<label>.<setting>`` must not read as one of them.
_TAKEN = RESERVED | {"model", "tool", "runtime", "screen", "sagent"}


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSpawn:
    """Start a child agent on a task."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        prompt: Annotated[str, "Self-contained instruction for the child."]
        system: Annotated[
            str | None,
            "System prompt for the child; omit to inherit yours.",
        ] = None
        provider: Annotated[
            str | None,
            "Catalog provider, such as openai-sub; omit to inherit yours.",
        ] = None
        model: Annotated[
            str | None,
            "Model name or role (best, default, everyday, utility), with any"
            " +CONTEXT or +EFFORT tag; omit to inherit yours, window included.",
        ] = None
        account: Annotated[
            str | None,
            "Named sign-in account for the child's provider, such as work; omit"
            " to inherit yours.",
        ] = None
        effort: Annotated[
            ThinkingEffort | None,
            "Reasoning effort; omit to inherit yours.",
        ] = None
        budget: Annotated[
            ThinkingBudget | None,
            "Reasoning budget: none, auto, or fixed; omit to inherit yours.",
        ] = None
        output: Annotated[
            ThinkingOutput | None,
            "What of the reasoning comes back: none, text, or redacted; omit to"
            " inherit yours.",
        ] = None
        tools: Annotated[
            tuple[str, ...] | None,
            "Which of your tools the child gets; omit for all, [] for none.",
        ] = None
        max_tool_call_rounds: Annotated[
            int | None,
            "Cap on the child's tool-call rounds.",
            Arg(minimum=1),
        ] = None
        max_depth: Annotated[
            int | None,
            "Generations the child may spawn below it; 0 makes it a leaf.",
            Arg(minimum=0),
        ] = None
        label: Annotated[
            str,
            "Label for the child; omit for <your label>_<n>.",
        ] = ""
        watch: Annotated[
            bool,
            "Start your watcher instead of a child: prompt is its policy, its"
            " first request.",
        ] = False

    name = "AgentSpawn"
    description = (files(__package__) / "assets" / "agentspawn.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    team: Team = field(repr=False)
    """Who spawns, and where the child joins."""

    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``AgentSpawn <label>: <first words>``."""
        label = str(arguments.get("label", "") or "")
        prompt = " ".join(str(arguments.get("prompt", "")).split())
        head = f"AgentSpawn{f' {label}' if label else ''}"
        return f"{head}: {prompt}" if prompt else head

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``parent`` for a watcher's start, as AgentSend's; else ``None``."""
        del context
        return "parent" if arguments.get("watch") is True else None

    def reminder(self, context: ToolContext) -> str:
        """Return the caller's children, who owes whom, and whether it is a leaf.

        Args:
          context: The calling agent's state.

        Returns:
          text: Such as ``Your children: reviewer (owes you a reply), kid.``;
            ``""`` outside a team, or with no children and spawning allowed.

        """
        me = context.label
        if me not in self.team:
            return ""
        mine = self.team.owes(me)
        kids = [
            label
            + (" (owes you a reply)" if me in self.team.owes(label) else "")
            + (" (you owe it a reply)" if label in mine else "")
            for label in self.team.children(me)
        ]
        return " ".join(
            part
            for part in (
                f"Your children: {', '.join(kids)}." if kids else "",
                ""
                if self.team.can_spawn(me)
                else "Spawn budget: you are a leaf agent; AgentSpawn will refuse.",
            )
            if part
        )

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> AgentStatusResult:
        """Build the child, join it to the team, and run or start it.

        Args:
          context: The calling agent's state; its label names the parent.
          arguments: What the model passed, as ``AgentSpawn.Arguments``.

        Returns:
          result: The child's last text, or why it failed; with ``watch``, the
            caller's new watcher, sent ``prompt`` as a request after the
            ``watch`` skill.

        Raises:
          ValueError: The caller is in no team or may not spawn, a tool or model
            is unknown, or the label is taken or reserved; with ``watch``, the
            prompt is empty or the caller is no unwatched root.

        """
        a = parse(arguments, AgentSpawn.Arguments)
        parent_label = context.label
        if parent_label not in self.team:
            raise ValueError("AgentSpawn needs a calling agent in the team.")
        if a.watch:
            if not a.prompt.strip():
                raise ValueError("A watcher needs a policy in prompt.")
            label = await self.start_watcher(parent_label)
            self.team.send(parent_label, recipient=label, content=a.prompt)
            return AgentStatusResult(
                call_id="",
                agent_id=label,
                agent_kind="watcher",
                state="running",
                content=f"Started your watcher, {label}; it owes you a reply.",
            )
        if not self.team.can_spawn(parent_label):
            raise ValueError(
                "Your depth cap is reached: you are a leaf and cannot spawn.",
            )
        if a.label and a.label in self.team:
            raise ValueError(f"An agent is already labelled {a.label!r}.")
        if a.label in _TAKEN:
            raise ValueError(
                f"{a.label!r} is reserved: {', '.join(sorted(_TAKEN))}.",
            )
        label = a.label or self.team.label_for(parent_label)
        parent = self.team[parent_label]
        allowed = _allowed(parent, a.tools)
        model = await self._model(
            parent,
            provider=a.provider,
            name=a.model,
            account=a.account,
            effort=a.effort,
            budget=a.budget,
            output=a.output,
        )
        path = parent.session.journal.path
        journal = child_journal(path, label) if path is not None else None
        spec = (
            ChildSpec(
                label=label,
                journal=str(journal),
                tools=tuple(sorted(allowed if allowed is not None else parent.tools)),
                max_rounds=a.max_tool_call_rounds,
                max_depth=a.max_depth,
            )
            if journal is not None
            else None
        )
        told = a.account if a.account is not None else self.team.catalog.account
        child = await self._start(
            parent,
            label=label,
            parent_label=parent_label,
            model=model,
            account=None if told in ("", "default") else told,
            allowed=allowed,
            system=a.system,
            journal=journal,
            max_rounds=a.max_tool_call_rounds,
            max_depth=a.max_depth,
            spec=spec,
            recall=False,
        )
        return await self._run(child, label=label, parent=parent, a=a)

    async def watch(
        self,
        watched: str,
        *,
        label: str = "",
        journal: Path | None = None,
        before: str = "",
    ) -> str:
        """Start a watcher of the root ``watched``: its ``parent`` by name.

        It runs on ``watched``'s model, settings, tools, and system prompt, so
        its requests open with the same cached prefix, and is sent what
        ``watched`` does: at once ``before``, then each time it idles.

        Args:
          watched: A root of the team.
          label: The watcher's; ``""`` is ``watcher``, or ``watcher_<n>`` once taken.
          journal: The watcher's journal to resume; ``None`` starts one beside
            ``watched``'s, if it keeps one.
          before: What the first message leads with, such as how to watch.

        Returns:
          label: The watcher's.

        Raises:
          ValueError: ``watched`` is no root, or is watched already.

        """
        target = self.team[watched]
        if self.team.above(watched) is not None:
            raise ValueError(f"{watched!r} is no unwatched root.")
        label = label or (
            "watcher" if "watcher" not in self.team else self.team.label_for("watcher")
        )
        resumed = journal is not None
        path = target.session.journal.path
        if journal is None and path is not None:
            journal = child_journal(path, label)
        model = await self._model(
            target,
            provider=None,
            name=None,
            account=None,
            effort=None,
            budget=None,
            output=None,
        )
        stack = AsyncExitStack()
        opened = JsonlJournal(journal) if journal is not None else None
        agent = Agent(
            model=model,
            tools=list(target.tools.values()),
            allowed=target.allowed,
            system_prompt=target.system_prompt,
            journal=opened,
            compactor=target.compactor_factory,
            cwd=target.runtime.tool_context.cwd,
            root=target.root,
            agents_md=target.agents_md,
            max_result_chars=target.runtime.tool_context.max_result_chars,
            environment=target.environment,
            account=target.account,
            persistent_retry=target.runtime.persistent_retry,
        )
        agent.model_cache = dict(target.model_cache)
        self.team.join(agent, label=label, closing=stack)
        try:
            if opened is not None:
                await stack.enter_async_context(opened)
            if resumed:
                await agent.recall(self.team.catalog)
            await stack.enter_async_context(agent)
        except BaseException:
            await self.team.stop(label)
            await model.aclose()
            raise
        self.team.restore(label)
        self.team.watch(
            label,
            watched,
            journal=str(journal) if journal is not None else None,
        )
        feed = Feed(team=self.team, watched=watched, watcher=label)
        if resumed:
            feed.sent = len(list(target.session.history()))
        target.runtime.observers.append(feed)
        feed.send(before)
        return label

    async def start_watcher(self, watched: str) -> str:
        """Start the watcher of the root ``watched``, first sent the ``watch`` skill.

        The TUI's ``/send parent`` and ``AgentSend(to="parent")`` from a root
        both start one so.

        Args:
          watched: A root of the team.

        Returns:
          label: The watcher's.

        Raises:
          ValueError: ``watched`` is no root, or is watched already.

        """
        target = self.team[watched]
        skill = target.tools.get("Skill")
        loader = skill if isinstance(skill, Skill) else Skill()
        how = await loader.call(
            target.runtime.tool_context,
            {"skill": "watch", "args": watched},
        )
        return await self.watch(watched, before=how.text())

    async def resume(self, parent_label: str) -> list[str]:
        """Start again each persistent descendant the journals hold as running.

        Each resumes its own journal, on the model, settings, and prompt it
        stood there, then its own children from that journal, and waits for a
        message; one whose model, tools, or journal can no longer be had is
        skipped, with its children, and a warning.

        Args:
          parent_label: The resumed agent, already in the team.

        Returns:
          labels: Each descendant started, a parent before its children, under
            its old label, or a new one when that is taken.

        """
        parent = self.team[parent_label]
        started: list[str] = []
        for spec in self.team.standing(parent_label):
            label = (
                spec.label
                if spec.label not in self.team
                else self.team.label_for(parent_label)
            )
            try:
                allowed = _allowed(parent, spec.tools)
                model = (
                    self.team.catalog.model(
                        spec.model,
                        current=parent.model.provider.name,
                    )
                    if spec.model
                    else await self._model(
                        parent,
                        provider=None,
                        name=None,
                        account=None,
                        effort=None,
                        budget=None,
                        output=None,
                    )
                )
                await self._start(
                    parent,
                    label=label,
                    parent_label=parent_label,
                    model=model,
                    account=self.team.catalog.account,
                    allowed=allowed,
                    system=None,
                    journal=session_file(Path(spec.journal).parent),
                    max_rounds=spec.max_rounds,
                    max_depth=spec.max_depth,
                    spec=spec,
                    recall=True,
                )
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning("not resuming agent %r: %s", spec.label, e)
                continue
            started.append(label)
            started.extend(await self.resume(label))
        for record in parent.session.standing():
            label, journal = record.payload.get("label"), record.payload.get("journal")
            if record.kind != "watcher" or not isinstance(label, str):
                continue
            try:
                started.append(
                    await self.watch(
                        parent_label,
                        label=label if label not in self.team else "",
                        journal=Path(journal) if isinstance(journal, str) else None,
                    ),
                )
            except (OSError, RuntimeError, ValueError) as e:
                logger.warning("not resuming watcher %r: %s", label, e)
        return started

    # The child joins before anything is awaited, so a label chosen for it cannot be
    # taken by a spawn running beside this one. It is offered every tool its parent
    # is, so its requests open with the parent's cached prefix; ``allowed`` gates
    # what it may call.
    # A child given ``recall`` takes the model, settings, and prompt its journal stood,
    # once the journal is open and before the child opens.
    async def _start(
        self,
        parent: Agent,
        *,
        label: str,
        parent_label: str,
        model: Model,
        account: str | None,
        allowed: frozenset[str] | None,
        system: str | None,
        journal: Path | None,
        max_rounds: int | None,
        max_depth: int | None,
        spec: ChildSpec | None,
        recall: bool,
    ) -> Agent:
        """Join a child of ``parent`` to the team, then open it and its journal."""
        stack = AsyncExitStack()
        opened = JsonlJournal(journal) if journal is not None else None
        payers = self._payers(parent_label)
        left = [
            payer.max_spend_usd - payer.spend_usd
            for payer in payers
            if payer.max_spend_usd is not None
        ]
        child = Agent(
            model=model,
            tools=list(parent.tools.values()),
            allowed=allowed,
            system_prompt=system if system is not None else parent.system_prompt,
            journal=opened,
            compactor=parent.compactor_factory,
            cwd=parent.runtime.tool_context.cwd,
            root=parent.root,
            max_rounds=max_rounds,
            max_spend_usd=min(left, default=None),
            agents_md=parent.agents_md,
            max_result_chars=parent.runtime.tool_context.max_result_chars,
            environment=parent.environment,
            account=account,
            persistent_retry=parent.runtime.persistent_retry,
        )
        child.model_cache = dict(parent.model_cache)
        # One conversation to vendors: the child keeps its own id, from its
        # journal, and takes its root's.
        above, below = parent.runtime, child.runtime
        below.agent = below.conversation
        below.conversation = above.conversation
        below.parent = above.agent or above.conversation
        below.parent_turn = above.turns
        below.root_prompt = (
            above.root_prompt or Request(agent=below.parent, turn=above.turns).prompt_id
        )
        below.name = f"{above.name}/{label}"
        self.team.join(
            child,
            label=label,
            parent=parent_label,
            max_depth=max_depth,
            closing=stack,
            spec=spec,
        )
        try:
            if opened is not None:
                await stack.enter_async_context(opened)
            if recall:
                await child.recall(self.team.catalog)
            await stack.enter_async_context(child)
            # Its journal's standing records are read as it opens.
            self.team.restore(label)
        except BaseException:
            await self.team.stop(label)
            # The child never opened, so it will not close the model it owns.
            await model.aclose()
            raise
        child.runtime.observers.extend(
            (_Forward(label=label, parent=parent), _Charge(label=label, payers=payers)),
        )
        return child

    async def _run(
        self,
        child: Agent,
        *,
        label: str,
        parent: Agent,
        a: AgentSpawn.Arguments,
    ) -> AgentStatusResult:
        """Run ``child`` on the prompt to idle; it stays, its inbox open."""
        started = time.monotonic()
        tally = _Tally()
        child.runtime.observers.append(tally)
        self.team.awaited.add(label)
        try:
            reply, state = await _answer(child, a.prompt)
        except asyncio.CancelledError:
            # The spawn is cancelled, not the child: it stops its turn and stays.
            child.runtime.halt()
            child.runtime.push(types.Kill())
            raise
        finally:
            self.team.awaited.discard(label)
            if tally in child.runtime.observers:
                child.runtime.observers.remove(tally)
            parent.runtime.publish(
                types.ChildDoneEvent(
                    label=label,
                    elapsed=time.monotonic() - started,
                    tokens=tally.tokens,
                    cost=tally.cost,
                ),
            )
        return AgentStatusResult(
            call_id="",
            is_error=state != "completed",
            agent_id=label,
            agent_kind="persistent",
            prompt=a.prompt,
            content=reply,
            model=child.model.capability.model_id,
            state=state,
            tokens=tally.tokens,
            duration_sec=time.monotonic() - started,
            tool_calls=tally.calls,
            output_file=str(child.session.journal.path)
            if child.session.journal.path
            else None,
        )

    def _payers(self, label: str) -> tuple[Agent, ...]:
        """Return ``label``'s agent and each above it, whose caps a child spends against."""
        payers: list[Agent] = []
        above: str | None = label
        while above is not None:
            payers.append(self.team[above])
            above = self.team.parent(above)
        return tuple(payers)

    # A named ``account`` signs in the provider named, else the parent's, as that
    # account; the catalog's own account stays as it was.
    async def _model(
        self,
        parent: Agent,
        *,
        provider: str | None,
        name: str | None,
        account: str | None,
        effort: ThinkingEffort | None,
        budget: ThinkingBudget | None,
        output: ThinkingOutput | None,
    ) -> Model:
        """Return a model of its own for the child, the parent's choices kept."""
        current = parent.model
        catalog = self.team.catalog
        build: Callable[[str], Model]
        if account is not None:
            signed_in = catalog.signed_in(
                provider or current.provider.name,
                account=account,
            )
            build = signed_in.__getitem__
        elif provider is not None:
            build = catalog[provider].__getitem__
        else:
            build = partial(catalog.model, current=current.provider.name)
        model = await successor(
            build,
            name if name is not None else current.capability.model_id,
            current=current,
            model_cache=parent.model_cache,
        )
        try:
            settings = model.settings
            if effort is not None:
                settings = replace(settings, thinking_effort=effort)
            if budget is not None:
                settings = replace(settings, thinking_budget=budget)
            if output is not None:
                settings = replace(settings, thinking_output=output)
            model.settings = settings
        except ValueError:
            await model.aclose()
            raise
        return model


def _allowed(parent: Agent, names: tuple[str, ...] | None) -> frozenset[str] | None:
    """Return what a child of ``parent`` may call: ``names``, or the parent's own."""
    if names is None:
        return parent.allowed
    mine = parent.allowed if parent.allowed is not None else frozenset(parent.tools)
    unknown = sorted(set(names) - mine)
    if unknown:
        raise ValueError(
            f"Unknown tools {', '.join(unknown)}; yours: {', '.join(sorted(mine))}",
        )
    return frozenset(names) if set(names) != set(parent.tools) else None


async def _answer(
    child: Agent,
    prompt: str,
) -> tuple[str, Literal["completed", "failed"]]:
    """Return what ``child`` replies to ``prompt``, or why it failed, and which."""
    try:
        return await child.send(prompt), "completed"
    except Exception as e:  # noqa: BLE001 -- A child's failure is its spawn's result, not the parent's.
        return describe(e), "failed"


@dataclass(slots=True, kw_only=True)
class _Tally:
    """What a child's model calls cost, counted as they finish."""

    tokens: int = 0
    cost: float = 0.0
    calls: int = 0

    def __call__(self, event: types.RuntimeEvent) -> None:
        """Count a finished model call."""
        if isinstance(event, types.ModelResponseComplete):
            self.tokens += event.response.tokens.total
            self.cost += event.response.spend.total
            self.calls += sum(isinstance(r, ToolCall) for r in event.response.records)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Forward:
    """Passes a child's events to its parent's observers, wrapped with its label."""

    label: str
    parent: Agent

    def __call__(self, event: types.RuntimeEvent) -> None:
        """Republish ``event`` on the parent's runtime."""
        if isinstance(
            event,
            (
                types.ModelResponseComplete,
                types.ModelResponseError,
                types.ModelCallSuspended,
                types.CompactComplete,
                types.DetachedResult,
                AgentToAgentMessage,
                ToolResult,
            ),
        ):
            self.parent.runtime.publish(types.ChildEvent(label=self.label, inner=event))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Charge:
    """Charges what a child's model calls cost to its parent and each agent above.

    Its replies, and what it charges itself, as its compactions; what its own
    children charge it, their observers charge each payer already.
    """

    label: str
    payers: tuple[Agent, ...]

    def __call__(self, event: types.RuntimeEvent) -> None:
        """Charge a finished model call to each payer."""
        if isinstance(event, types.ModelResponseComplete):
            usd = event.response.spend.total
        elif isinstance(event, types.Charged) and event.by == self.label:
            usd = event.usd
        else:
            return
        if usd:
            for payer in self.payers:
                payer.charge(usd, by=self.label)
