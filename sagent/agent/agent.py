"""The agent: a model, its tools, and the session it keeps, run by the runtime.

Typical usage example:

  async with ModelCatalog() as catalog:
      opus = catalog["anthropic"]["default"]
      async with JsonlJournal(path) as journal, Agent(
          model=opus,
          tools=[Read(), Bash()],
          journal=journal,
      ) as agent:
          reply = await agent.send("fix the failing test")
          async for words in agent.stream("now run the tests"):
              print(words, end="")
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, override

import asyncio
import uuid

from treekle import from_plain

from sagent.agent.agents_md import AgentsMdConfig, Directives
from sagent.agent.environment import environment
from sagent.model.cost import TokenCost, TokenCount
from sagent.model.types import Response
from sagent.providers.catalog import CachedModel, ModelMemory, cache_key, recall
from sagent.providers.table import ModelTags
from sagent.runtime import types
from sagent.runtime.context import resolve_context
from sagent.runtime.exceptions import BudgetExhaustedError, RoundLimitError
from sagent.runtime.runtime import Compactor, Pruning, Runtime, Sizing
from sagent.runtime.tape import ContextSplice, ReferrableTapeEvent
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    ContextClear,
    SystemMessage,
    Thinking,
    TokenUsage,
    ToolCall,
    ToolResult,
    TurnContext,
    UncategorizedRecord,
    UncategorizedToolResult,
    UserMessage,
)
from sagent.tools.types import Briefing, Reminding, ToolContext
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterator,
        Collection,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )

    from sagent.agent.types import (
        Agent as AgentProtocol,
        Factory,
        Journal,
        Session,
        Shrinker,
    )
    from sagent.providers.types import Model, ModelCatalog
    from sagent.runtime.tape import TapeRecord
    from sagent.session.types import Attachment, SessionRecord
    from sagent.tools.types import Tool


__all__ = [
    "Agent",
    "NoJournal",
    "NoShrinker",
    "RuntimeSession",
    "default_system_prompt",
    "spent",
]


class NoJournal:
    """A journal that keeps nothing: the default, for an agent nothing resumes."""

    @property
    def path(self) -> None:
        """No file."""
        return

    def append(self, *records: SessionRecord) -> None:
        """Drop ``records``."""
        del records

    def __iter__(self) -> Iterator[SessionRecord]:
        """Yield nothing: nothing was kept."""
        return iter(())

    def history(self) -> Iterator[SessionRecord]:
        """Yield nothing: nothing was kept."""
        return iter(())

    def flush(self) -> None:
        """Do nothing; nothing is kept."""


class NoShrinker:
    """A compactor or pruner that never shrinks: the default.

    Args:
      agent: The agent it would shrink; unused.

    """

    def __init__(self, agent: AgentProtocol) -> None:
        del agent

    def should_run(self) -> bool:
        """Return ``False``: never due."""
        return False

    async def run(self, instructions: str = "") -> None:
        """Refuse to shrink.

        Args:
          instructions: Unused.

        Raises:
          NotImplementedError: Always; this shrinker never shrinks.

        """
        del instructions
        raise NotImplementedError("This agent has no compactor or pruner.")


class Agent(AsyncResource):
    """A model, its tools, and the session it keeps; ``async with`` runs it.

    ``sagent.agent.types.Agent`` draws the tree of its parts.

    Args:
      model: What answers; assign ``agent.model`` to switch.
      tools: What the model may call; names must differ.
      system_prompt: What the model is told it is; ``None`` uses the default.
      journal: Where the session is written; nowhere when omitted. A
        journal that opens, such as ``JsonlJournal``, is opened by the caller.
      compactor: Builds what starts the session over from a summary.
      pruner: Builds what drops old tool results.
      cwd: Where the tools start; the current directory when omitted.
      root: Where the session started, which AGENTS.md and the skills are read
        from and each new window returns to; ``cwd`` when omitted.
      max_rounds: Tool-call rounds a turn may run; ``None`` for no cap.
      max_spend_usd: What the agent may spend; ``None`` for no cap.
      persistent_retry: Whether a long rate limit is waited out, not raised.
      agents_md: Where the AGENTS.md files are looked for; the defaults when
        omitted.
      allowed: The tools the model may call, by name; ``None`` for all. The
        rest are still offered, so every agent's tools cache alike.
      max_result_chars: How much text one tool result may hold; the tool
        context's default when omitted.
      environment: Whether the model is told where it runs: its directory, the
        git state, platform and shell. Off for an agent serving people the
        host's details are not for.
      account: The named login ``model``'s provider signs in as; ``None`` for
        its default. Whoever switches the model keeps it current.

    Raises:
      ValueError: ``allowed`` names a tool not in ``tools``.

    """

    def __init__(
        self,
        *,
        model: Model,
        tools: Sequence[Tool] = (),
        system_prompt: str | None = None,
        journal: Journal | None = None,
        compactor: Factory[Shrinker] = NoShrinker,
        pruner: Factory[Shrinker] = NoShrinker,
        cwd: Path | None = None,
        root: Path | None = None,
        max_rounds: int | None = None,
        max_spend_usd: float | None = None,
        persistent_retry: bool = False,
        agents_md: AgentsMdConfig | None = None,
        allowed: Collection[str] | None = None,
        max_result_chars: int | None = None,
        environment: bool = True,
        account: str | None = None,
    ) -> None:
        self.agents_md: AgentsMdConfig = (
            agents_md if agents_md is not None else AgentsMdConfig()
        )
        start = cwd if cwd is not None else Path.cwd()
        kept = journal.path if journal is not None else None
        self.runtime: Runtime = Runtime(
            model=model,
            tools=tools,
            tool_context=ToolContext(
                cwd=start,
                root=root if root is not None else start,
                spill_dir=kept.parent if kept is not None else None,
            ),
        )
        # Named by its journal, which a resume reopens, so vendors see one
        # conversation across restarts; an agent with none is new each time.
        self.runtime.conversation = str(
            uuid.uuid4()
            if kept is None
            else uuid.uuid5(uuid.NAMESPACE_URL, kept.resolve().as_uri()),
        )
        if max_result_chars is not None:
            self.runtime.tool_context.max_result_chars = max_result_chars
        self.tools: Mapping[str, Tool] = self.runtime.tools
        self.environment = environment
        self.account = account
        self.allowed = frozenset(allowed) if allowed is not None else None
        self._system_prompt = system_prompt
        self._session = RuntimeSession(
            self,
            journal=journal if journal is not None else NoJournal(),
            compactor=compactor,
            pruner=pruner,
        )
        self._session.frame()
        self.session: Session = self._session
        # What a child it spawns compacts with, so the child compacts as it does.
        self.compactor_factory: Factory[Shrinker] = compactor
        self.runtime.context_updates = self._session.updates
        # A compactor the runtime can call compacts on ``Compact``, and when due,
        # and sizes each reply to the room the window leaves.
        summarizer = self._session.compactor
        if isinstance(summarizer, Compactor):
            self.runtime.compactor = summarizer
            self.runtime.compact_due = summarizer.should_run
            if isinstance(summarizer, Sizing):
                self.runtime.reply_room = summarizer.reply_room
        # A pruner the runtime can ask drops old results before a call, when that pays.
        dropper = self._session.pruner
        if isinstance(dropper, Pruning):
            self.runtime.prune = dropper.prune
        self._loop: asyncio.Task[None] | None = None
        self.max_rounds = max_rounds
        self.max_spend_usd = max_spend_usd
        self.runtime.persistent_retry = persistent_retry
        # What the model calls have cost, a resumed session's included.
        self.spend_usd = 0.0
        # The rounds of the turn so far.
        self._rounds = 0
        # The tools allowed before the round cap took them all; ``None`` while uncapped.
        self._capped: tuple[frozenset[str] | None] | None = None
        # Whether no turn is under way: no model call since the runtime went idle.
        self._idle = True
        # Models assigned away from, closed at the runtime's next safe point.
        self._retired: list[Model] = []
        # The settings each model was left with, by ``cache_key``: a switch back
        # restores them.
        self.model_cache: dict[str, CachedModel] = {}
        # The cap a halt was for, raised by the turn it stopped.
        self._limit: Exception | None = None

    @property
    def model(self) -> Model:
        """What answers; the next model call uses whatever is assigned."""
        return self.runtime.model

    # The agent owns the models assigned to it. One assigned away from may still be
    # streaming, so it closes once the runtime is between turns; with no loop yet,
    # when one starts.
    @model.setter
    def model(self, model: Model) -> None:
        old, self.runtime.model = self.runtime.model, model
        if old is model:
            return
        key = cache_key(old)
        if key == cache_key(model):
            self.model_cache.pop(key, None)
        else:
            self.model_cache[key] = CachedModel(
                capability=old.capability,
                settings=old.settings,
            )
        self._retired.append(old)
        if self._loop is not None:
            self._close_retired_soon()

    @property
    def running(self) -> bool:
        """Whether the runtime's loop serves turns: opened, and not stopped or died."""
        return self._loop is not None and not self._loop.done()

    @property
    def system_prompt(self) -> str | None:
        """What the model is told it is; ``None`` uses the default."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, prompt: str | None) -> None:
        self._system_prompt = prompt
        self._session.frame()

    @property
    def allowed(self) -> frozenset[str] | None:
        """The tools the model may call, by name; ``None`` for all."""
        return self.runtime.allowed

    # Every tool stays offered, so a change of gate keeps the cached prefix; the
    # next round states the new gate to the model.
    @allowed.setter
    def allowed(self, names: frozenset[str] | None) -> None:
        if names is not None:
            unknown = sorted(names - set(self.tools))
            if unknown:
                raise ValueError(f"Unknown tools: {', '.join(unknown)}.")
        self.runtime.allowed = names

    def charge(self, usd: float, *, by: str) -> None:
        """Count spend no reply of this agent's own brought; halt the turn past the cap.

        The spend is journaled, so a resumed session counts it too, and told to the
        runtime's observers as ``Charged``.

        Args:
          usd: What one model call cost.
          by: The label of the agent that spent it: one below this one, or this
            one's own for a compaction.

        """
        self.spend_usd += usd
        self._session.journal.append(
            TokenUsage(timestamp=_now(), info={"spend_usd": usd, "agent": by}),
        )
        self.runtime.publish(types.Charged(usd=usd, by=by))
        if (over := self._over_budget()) is not None:
            self._limit = over
            self.runtime.halt()

    async def recall(self, catalog: ModelCatalog) -> None:
        """Take the model, its account, the model cache, and the prompt last stood.

        Called before the agent opens, on a journal already open; what was
        never stood is left as it is.

        Args:
          catalog: Where the providers of the models stood are.

        Raises:
          ValueError: A model stood is no longer offered as it was.

        """
        standing: dict[str, UncategorizedRecord] = {}
        for record in self._session.journal:
            _apply_standing(standing, record)
        stood = {
            key: {k: v for k, v in r.payload.items() if k != "standing"}
            for key, r in standing.items()
        }
        if "model" in stood:
            memory = from_plain(stood["model"], ModelMemory)
            self.model, cache = await recall(memory, catalog=catalog)
            self.model_cache = cache
            self.account = memory.account
        if "system_prompt" in stood:
            self.system_prompt = from_plain(stood["system_prompt"], _Prompt).text

    @override
    async def aopen(self) -> None:
        self._session.frame()
        self._session.resume()
        self.runtime.observers.append(self._observe)
        self._loop = asyncio.create_task(self.runtime.run_forever())
        if self._retired:
            self._close_retired_soon()

    @override
    async def aclose(self) -> None:
        loop, self._loop = self._loop, None
        try:
            if loop is not None:
                self.runtime.push(types.Quit())
                await loop
        finally:
            # What the session holds is written even when the loop died raising.
            # An agent closed twice, or never opened, has no observer to remove.
            if self._observe in self.runtime.observers:
                self.runtime.observers.remove(self._observe)
            self._session.persist()
            self.session.journal.flush()
            self._retired.append(self.runtime.model)
            await self._close_retired()

    async def send(
        self,
        text: str,
        *,
        attachments: Sequence[Attachment] = (),
    ) -> str:
        """Send ``text``; return the reply once the agent is idle.

        Args:
          text: What the model reads.
          attachments: Images and documents sent with it.

        Returns:
          reply: What the model said last in the turn.

        Raises:
          BudgetExhaustedError: The spend reached ``max_spend_usd``, before or
            during the turn.
          RoundLimitError: The turn reached ``max_rounds``.
          asyncio.CancelledError: Someone halted the turn.
          RuntimeError: The agent is not open, or its runtime stopped.

        """
        reply = ""
        async for event in self._turn(text, attachments=attachments):
            if isinstance(event, types.ModelResponseComplete):
                said = "".join(
                    r.content or ""
                    for r in event.response.records
                    if isinstance(r, AssistantMessage)
                )
                if said:
                    reply = said
        return reply

    async def stream(
        self,
        text: str,
        *,
        attachments: Sequence[Attachment] = (),
    ) -> AsyncIterator[str]:
        """Send ``text``; yield the reply's words as they are written, until idle.

        Args:
          text: What the model reads.
          attachments: Images and documents sent with it.

        Yields:
          words: The next few words of the reply.

        Raises:
          BudgetExhaustedError: The spend reached ``max_spend_usd``, before or
            during the turn.
          RoundLimitError: The turn reached ``max_rounds``.
          asyncio.CancelledError: Someone halted the turn.
          RuntimeError: The agent is not open, or its runtime stopped.

        """
        async for event in self._turn(text, attachments=attachments):
            if isinstance(event, types.ModelResponsePartial):
                yield event.text

    async def _turn(
        self,
        text: str,
        *,
        attachments: Sequence[Attachment],
    ) -> AsyncIterator[types.RuntimeEvent]:
        """Send ``text``; yield every event of the turn, until the agent is idle."""
        # A spent cap refuses the turn before a model call could spend more.
        if (over := self._over_budget()) is not None:
            raise over
        queue: asyncio.Queue[types.RuntimeEvent] = asyncio.Queue()
        observer = queue.put_nowait
        # A cap tripped outside a turn, by a message no turn sent, is not this one's.
        self._limit = None
        self.runtime.observers.append(observer)
        try:
            self.runtime.push(UserMessage(content=text, attachments=tuple(attachments)))
            while True:
                event = await self._next(queue)
                yield event
                if isinstance(event, types.AgentIdle):
                    return
                if isinstance(event, types.ModelResponseError):
                    raise event.exception
                if isinstance(event, types.Halt):
                    limit, self._limit = self._limit, None
                    if limit is not None:
                        raise limit
                    raise asyncio.CancelledError
        finally:
            self.runtime.observers.remove(observer)

    # The loop raises what it cannot recover from; waiting on its events alone would
    # then wait forever.
    async def _next(
        self,
        queue: asyncio.Queue[types.RuntimeEvent],
    ) -> types.RuntimeEvent:
        """Return the next event, or raise why the runtime's loop ended first."""
        loop = self._loop
        if loop is None:
            raise RuntimeError("The agent is not open: use it with 'async with'.")
        get = asyncio.ensure_future(queue.get())
        await asyncio.wait((get, loop), return_when=asyncio.FIRST_COMPLETED)
        if get.done():
            return get.result()
        get.cancel()
        loop.result()
        raise RuntimeError("The agent's runtime stopped.")

    def _close_retired_soon(self) -> None:
        """Close the models assigned away from at the runtime's next safe point."""
        self.runtime.push(
            types.Scheduled(
                work=self._close_retired,
                done=asyncio.get_running_loop().create_future(),
            ),
        )

    async def _close_retired(self) -> None:
        """Close each model assigned away from, once."""
        retired = self._retired
        self._retired = []
        for model in retired:
            await model.aclose()

    def _observe(self, event: types.RuntimeEvent) -> None:
        """Write the session, start each window afresh, and hold the caps."""
        if isinstance(event, types.SaveSession):
            self._session.persist()
        elif isinstance(event, (types.CompactComplete, types.ClearComplete)):
            self.runtime.tool_context.files.forget()
            self._return_home()
            self._session.frame()
        elif isinstance(event, UserMessage) or (
            # A persistent child's turns start with another agent's message; one
            # arriving mid-turn, such as a child's idle ping, starts none.
            isinstance(event, AgentToAgentMessage) and self._idle
        ):
            self._rounds = 0
            self._uncap()
        elif isinstance(event, types.ModelResponseComplete):
            self._charge(event.response)
        if isinstance(event, (types.AgentIdle, types.ModelCallStarted)):
            self._idle = isinstance(event, types.AgentIdle)

    # The turn that reaches the round cap leaves the model no tool to call, so the next
    # answers with what it has; a call then halts the turn.
    def _charge(self, response: Response) -> None:
        """Count a model turn against the caps; halt the turn past either."""
        self.spend_usd += response.spend.total
        # Replying is never capped: a debtor at the cap could otherwise never pay.
        called = any(
            isinstance(r, ToolCall) and r.name != "AgentSend" for r in response.records
        )
        if (over := self._over_budget()) is not None:
            self._limit = over
        elif called and self._capped is not None:
            self._limit = RoundLimitError(
                rounds=self._rounds,
                max_rounds=self.max_rounds or self._rounds,
            )
        elif called:
            self._rounds += 1
            if self.max_rounds is not None and self._rounds >= self.max_rounds:
                was = self.runtime.allowed
                self._capped = (was,)
                self.runtime.allowed = frozenset({"AgentSend"}) & frozenset(
                    self.tools if was is None else was,
                )
        if self._limit is not None:
            self.runtime.halt()

    def _over_budget(self) -> BudgetExhaustedError | None:
        """Return the error a spend at or past ``max_spend_usd`` raises; ``None`` under it."""
        if self.max_spend_usd is None or self.spend_usd < self.max_spend_usd:
            return None
        return BudgetExhaustedError(
            total_cost_usd=self.spend_usd,
            max_budget_usd=self.max_spend_usd,
        )

    def _uncap(self) -> None:
        """Give back the tools the round cap took."""
        if self._capped is not None:
            (self.runtime.allowed,) = self._capped
            self._capped = None

    # A new window starts where the session did: its prompt holds that directory's
    # AGENTS.md, and the model, which no longer sees where it had moved, is told.
    def _return_home(self) -> None:
        """Move the tools back to the root, and say where they were."""
        context = self.runtime.tool_context
        root = self.root
        if context.cwd == root:
            return
        moved, context.cwd = context.cwd, root
        self._session.append(
            SystemMessage(
                timestamp=_now(),
                content=f"Working directory reset to {root}; it was {moved}.",
                subtype="moved",
            ),
        )

    @property
    def root(self) -> Path:
        """Where the session started; each new window returns there."""
        return self.runtime.tool_context.origin


class RuntimeSession:
    """The conversation the model sees, kept by the runtime, written to a journal.

    Args:
      agent: Whose runtime holds the conversation.
      journal: Where it is written, and what a resume reads.
      compactor: Builds what starts the session over from a summary.
      pruner: Builds what drops old tool results.

    """

    def __init__(
        self,
        agent: Agent,
        *,
        journal: Journal,
        compactor: Factory[Shrinker],
        pruner: Factory[Shrinker],
    ) -> None:
        path = journal.path
        self.id: str = path.parent.name if path is not None else uuid.uuid4().hex
        self.journal: Journal = journal
        self.compactor: Shrinker = compactor(agent)
        self.pruner: Shrinker = pruner(agent)
        self._agent = agent
        self._written = 0
        # What the journal's window reads as, so a merge can write only its part.
        self._lines: list[SessionRecord] = []
        # Where the model works and what it has seen, when the window's AGENTS.md
        # files were last looked for.
        self._seen: tuple[Path, tuple[Path, ...]] | None = None
        self._standing: dict[str, UncategorizedRecord] = {}
        # The model and prompt last stood, so each stands again only when it changes.
        self._memory: ModelMemory | None = None
        self._prompt: str | None = None
        self._directives = Directives(root=agent.root, config=agent.agents_md)

    # A window's prompt is read once, at its start, and never again until the next:
    # any byte it changed mid-window would rewrite the whole cached conversation. What
    # changes later is appended, by ``updates`` or as the call that changed it.
    def frame(self) -> None:
        """Build the system prompt the window keeps: the prompt, then what stands.

        Two blocks, each its own cache mark: the prompt, the same for every agent
        given it; then the root's AGENTS.md and each tool's briefing, the same
        for every agent started there.
        """
        agent = self._agent
        context = agent.runtime.tool_context
        self._directives = Directives(root=agent.root, config=agent.agents_md)
        self._seen = None
        standing = [
            self._directives.start(),
            *(
                t.briefing(context)
                for t in agent.tools.values()
                if isinstance(t, Briefing)
            ),
        ]
        agent.runtime.system = tuple(
            block
            for block in (
                _prompt(agent.system_prompt),
                "\n\n".join(s for s in standing if s),
            )
            if block
        )

    def __iter__(self) -> Iterator[SessionRecord]:
        """Yield what the model sees: the last clear onward."""
        return iter(_records(self._agent.runtime.context().messages))

    def history(self) -> Iterator[SessionRecord]:
        """Yield the whole session from its first record: every window, compacted ones too.

        What the runtime added since the last write is written first; a session
        whose journal keeps nothing yields what the model sees.

        Returns:
          records: The session's records, oldest first.

        """
        self.persist()
        records: list[SessionRecord] = list(self.journal.history())
        if not records:
            records = list(self)
        return iter(records)

    def append(self, *records: SessionRecord) -> None:
        """Add ``records`` to the end; the journal writes them after the loop's pass."""
        for entry in _entries(records):
            self._agent.runtime.append_history(entry)

    async def clear(self) -> None:
        """Schedule a start over with nothing kept, and wait; the journal keeps the past.

        The new window reads its system prompt afresh, as the agent frames it.
        """
        cleared = asyncio.Event()

        def observe(event: types.RuntimeEvent) -> None:
            if isinstance(event, types.ClearComplete):
                cleared.set()

        self._agent.runtime.observers.append(observe)
        try:
            self._agent.runtime.push(types.Clear())
            await cleared.wait()
        finally:
            self._agent.runtime.observers.remove(observe)
        self._agent.runtime.tool_context.files.forget()

    def stand(self, key: str, record: UncategorizedRecord | None) -> None:
        """Keep ``record`` in force under ``key``: written now and at each window's start.

        A resume reads only the last window, so what must outlive compaction,
        such as a persistent child to restart, is written again with each.

        Args:
          key: What the record is about; a later one under it replaces it.
          record: The record; ``None`` withdraws ``key``, which is written too,
            so a resume of this window does not read the old record.

        """
        # The key rides in the payload, so a resume finds the record again.
        stored = (
            UncategorizedRecord(
                kind=record.kind,
                payload={**record.payload, "standing": key},
            )
            if record is not None
            else UncategorizedRecord(kind="withdrawn", payload={"standing": key})
        )
        self.journal.append(stored)
        _apply_standing(self._standing, stored)

    def standing(self) -> list[UncategorizedRecord]:
        """Return the records in force, each as ``stand`` was given it."""
        return [
            replace(r, payload={k: v for k, v in r.payload.items() if k != "standing"})
            for r in self._standing.values()
        ]

    def resume(self) -> None:
        """Load the journal's last window into the runtime, or start one."""
        window = list(self.journal)
        if window:
            self.append(*_restated(window))
            for record in window:
                _apply_standing(self._standing, record)
            stood = {r.kind: r.payload for r in self.standing()}
            if "model" in stood:
                self._memory = from_plain(stood["model"], ModelMemory)
            if "system_prompt" in stood:
                self._prompt = from_plain(stood["system_prompt"], _Prompt).text
            self._agent.spend_usd = spent(window)
            self._lines = list(self)
            # Turn and window ids are counts, so a resume goes on counting from
            # the journal or would reuse an earlier turn's.
            runtime = self._agent.runtime
            for record in self.journal.history():
                if isinstance(record, (UserMessage, AgentToAgentMessage)):
                    runtime.turns += 1
                elif isinstance(record, ContextClear) and record.history:
                    runtime.compactions += 1
        else:
            self._start()
        self._written = len(self._agent.runtime.tape)
        self._agent.runtime.settle_detached()
        self._agent.runtime.hold_for_user()
        self.persist()

    def persist(self) -> None:
        """Write what the runtime added since the last write.

        A result restated in place, as a prune stubs it, is written as a
        ``restated`` record a resume applies; any other edit of earlier history,
        such as a compaction, writes the whole context again as a new window.
        Each splice is written against the context it changed, so what it hid
        is on disk before it, even taped in the same pass. A write that fails
        is tried again by the next.
        """
        tape = self._agent.runtime.tape
        # What came before a splice, then the splice alone.
        ends = [
            end
            for i in range(self._written, len(tape))
            if isinstance(tape[i], ContextSplice)
            for end in (i, i + 1)
        ]
        for end in (*ends, len(tape)):
            if end > self._written:
                self._write(tape[:end], start=self._written)
                self._written = end
        self._remember()

    def _remember(self) -> None:
        """Stand the model, its cache, and the system prompt, each when it changed."""
        agent = self._agent
        memory = ModelMemory.from_model(
            agent.model,
            agent.model_cache,
            account=agent.account,
        )
        if memory != self._memory:
            self.stand(
                "model",
                UncategorizedRecord(
                    kind="model",
                    payload={"model": memory.model, "cache": list(memory.cache)},
                ),
            )
            self._memory = memory
        prompt = agent.system_prompt
        if prompt != self._prompt:
            self.stand(
                "system_prompt",
                None
                if prompt is None
                else UncategorizedRecord(
                    kind="system_prompt",
                    payload={"text": prompt},
                ),
            )
            self._prompt = prompt

    def _write(self, tape: Sequence[TapeRecord], *, start: int) -> None:
        """Write ``tape[start:]``, what the runtime taped since the last write."""
        added = tape[start:]
        entries = [
            record.event
            for record in added
            if isinstance(record, ReferrableTapeEvent)
            and not isinstance(
                record.event,
                (types.CompactStarted, types.CompactComplete, types.CompactFailed),
            )
        ]
        if not any(isinstance(record, ContextSplice) for record in added):
            self.journal.append(*_journaled(entries))
            self._lines.extend(_records(entries))
            return
        current = _records(resolve_context(tape).messages)
        grown = _grown(self._lines, current)
        if grown is None:
            self._start(history=tuple(current))
            return
        restated, added_records = grown
        if restated:
            self.journal.append(
                UncategorizedRecord(kind="restated", payload={"results": restated}),
            )
        self.journal.append(
            *added_records,
            *(r for r in _journaled(entries) if isinstance(r, TokenUsage)),
        )
        self._lines = current

    def _start(self, *, history: tuple[SessionRecord, ...] = ()) -> None:
        """Write a window's start: the prompt, ``history``, the spend, what stands."""
        self.journal.append(
            ContextClear(
                timestamp=_now(),
                system_prompt="\n\n".join(self._agent.runtime.system),
                history=history,
                # What was spent before this window, for a resume to go on from.
                extra={"spend_usd": self._agent.spend_usd},
            ),
            *self._standing.values(),
        )
        self._lines = list(history)

    def updates(self) -> list[TurnContext | SystemMessage]:
        """Return what to restate before the next model call, if it changed.

        The runtime asks once per model round. The window's first states every
        setting; each later one only the settings changed since, and the time
        once the hour turns. An AGENTS.md file is given once a window: one of a
        directory the model works in or reads from, as it gets there, and a
        rule whose ``paths:`` match a file the model has seen, as it sees one. A
        tool's reminder is told whenever it differs from the last it gave, and
        withdrawn when it empties.

        Returns:
          updates: A ``TurnContext``, a ``SystemMessage`` of AGENTS.md files, and
            reminders, each only when there is something new to say.

        """
        context = self._agent.runtime.context().messages
        out: list[TurnContext | SystemMessage] = []
        turn = self._turn(context)
        if turn is not None:
            out.append(turn)
        tool_context = self._agent.runtime.tool_context
        seen = (tool_context.cwd, tuple(tool_context.files.recent()))
        if seen != self._seen:
            self._seen = seen
            cwd, files = seen
            given = {
                key
                for entry in context
                if isinstance(entry, SystemMessage) and entry.subtype == "rules"
                for key in from_plain(entry.extra.get("rules"), list[str], default=[])
            }
            text, matched = self._directives.found(
                [cwd, *(f.parent for f in files)],
                files,
                given=given,
            )
            if text:
                out.append(
                    SystemMessage(
                        timestamp=_now(),
                        content=text,
                        subtype="rules",
                        extra={"rules": sorted(matched)},
                    ),
                )
        out.extend(self._reminders(context))
        return out

    # One that empties is withdrawn, so the model stops relying on what it said.
    def _reminders(
        self,
        context: Sequence[types.ModelContextEvent],
    ) -> list[SystemMessage]:
        """Return each tool's reminder that differs from the last it gave this window."""
        said: dict[str, str] = {
            str(entry.extra.get("tool")): ""
            if entry.extra.get("withdrawn") is True
            else entry.content or ""
            for entry in context
            if isinstance(entry, SystemMessage) and entry.subtype == "reminder"
        }
        tool_context = self._agent.runtime.tool_context
        allowed = self._agent.allowed
        out: list[SystemMessage] = []
        for name, tool in self._agent.tools.items():
            # Offered for the cache but gated off, a tool's reminder would ask for
            # calls the model may not make, such as a reply by AgentSend.
            if not isinstance(tool, Reminding) or (
                allowed is not None and name not in allowed
            ):
                continue
            text = tool.reminder(tool_context)
            last = said.get(name, "")
            if text == last:
                continue
            out.append(
                SystemMessage(
                    timestamp=_now(),
                    content=text or f"No longer in force: {last}",
                    subtype="reminder",
                    extra={"tool": name, **({} if text else {"withdrawn": True})},
                ),
            )
        return out

    def _turn(self, context: Sequence[types.ModelContextEvent]) -> TurnContext | None:
        """Return the settings changed since the window last stated them, or ``None``."""
        stated = [entry for entry in context if isinstance(entry, TurnContext)]
        agent = self._agent
        cwd = agent.runtime.tool_context.cwd
        model = agent.model
        settings = model.settings
        offered = frozenset(agent.tools)
        current = _Stated(
            cwd=str(cwd) if agent.environment else None,
            environment=environment(
                cwd,
                knowledge_cutoff=model.capability.knowledge_cutoff,
            )
            if agent.environment
            else None,
            model=ModelTags.from_settings(model.capability, settings).name,
            tools=_gate(agent.allowed, offered=offered),
        )
        now = _now()
        delta = current
        if stated:
            last = _last_stated(stated, offered=offered)
            delta = _Stated(
                cwd=current.cwd if current.cwd != last.cwd else None,
                environment=current.environment
                if current.environment != last.environment
                else None,
                model=current.model if current.model != last.model else None,
                # A gate lifted names every tool: ``None`` would say nothing.
                tools=(
                    current.tools
                    if current.tools is not None
                    else tuple(sorted(offered))
                )
                if current.tools != last.tools
                else None,
            )
            if delta == _Stated() and _hour(stated[-1].timestamp) == _hour(now):
                return None
        return TurnContext(
            timestamp=now,
            cwd=delta.cwd,
            environment=delta.environment,
            model=delta.model,
            tools=delta.tools,
        )


def default_system_prompt() -> str:
    """Return the system prompt an agent without one is given."""
    return (files(__package__) / "assets" / "system_prompt.md").read_text(
        encoding="utf-8",
    )


def spent(records: Iterable[SessionRecord]) -> float:
    """Return what an agent's journal says it spent, from its start or one window's.

    A ``ContextClear`` states what was spent before it, compactions included, and each
    ``TokenUsage`` adds a model turn's spend.

    Args:
      records: The journal's records, oldest first: all of them, or a window.

    Returns:
      usd: What the agent had spent by the last record.

    """
    usd = 0.0
    for record in records:
        if isinstance(record, ContextClear):
            usd = from_plain(record.extra.get("spend_usd"), float, default=0.0)
        elif isinstance(record, TokenUsage):
            usd += from_plain(record.info.get("spend_usd"), float, default=0.0)
    return usd


@dataclass(frozen=True, slots=True, kw_only=True)
class _Prompt:
    """The system prompt an agent was given, as it stands in its journal."""

    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _Stated:
    """The settings a ``TurnContext`` states; ``None`` for each it leaves as was."""

    cwd: str | None = None
    environment: str | None = None
    model: str | None = None
    tools: tuple[str, ...] | None = None
    """The tools the model may call; ``None`` for every one offered."""


def _gate(
    allowed: Collection[str] | None,
    *,
    offered: frozenset[str],
) -> tuple[str, ...] | None:
    """Return the tools ``allowed``, sorted; ``None`` when that is every one ``offered``."""
    if allowed is None or frozenset(allowed) == offered:
        return None
    return tuple(sorted(allowed))


def _last_stated(
    stated: Iterable[TurnContext],
    *,
    offered: frozenset[str],
) -> _Stated:
    """Return the settings in force after ``stated``, each the last any of them said."""
    last = _Stated()
    for turn in stated:
        last = _Stated(
            cwd=turn.cwd if turn.cwd is not None else last.cwd,
            environment=turn.environment
            if turn.environment is not None
            else last.environment,
            model=turn.model if turn.model is not None else last.model,
            tools=_gate(turn.tools, offered=offered)
            if turn.tools is not None
            else last.tools,
        )
    return last


def _now() -> str:
    """Return the time now, local, with its UTC offset, to the second."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _prompt(prompt: str | None) -> str:
    """Return ``prompt``; ``None`` is the default prompt."""
    return prompt if prompt is not None else default_system_prompt()


def _hour(timestamp: str | None) -> str | None:
    """Return an ISO 8601 time to the hour, such as ``2026-10-24T13``."""
    return timestamp[:13] if timestamp is not None else None


def _records(entries: Iterable[types.ModelContextEvent]) -> list[SessionRecord]:
    """Return the history's entries as records, each model turn as its own."""
    return [
        record
        for entry in entries
        for record in (entry.records if isinstance(entry, Response) else (entry,))
    ]


def _journaled(entries: Iterable[types.ModelContextEvent]) -> list[SessionRecord]:
    """Return entries as records to journal, each model turn followed by its usage."""
    records: list[SessionRecord] = []
    for entry in entries:
        if not isinstance(entry, Response):
            records.append(entry)
            continue
        records.extend(entry.records)
        tokens, spend = entry.tokens, entry.spend
        records.append(
            TokenUsage(
                timestamp=_now(),
                info={
                    "request": tokens.request,
                    "response": tokens.response,
                    "cache_write": tokens.cache_write,
                    "cache_write_1h": tokens.cache_write_1h,
                    "cache_read": tokens.cache_read,
                    "spend_usd": spend.total,
                },
            ),
        )
    return records


def _entries(records: Iterable[SessionRecord]) -> list[types.ModelContextEvent]:
    """Return records as history entries; a run of the model's records is one turn."""
    entries: list[types.ModelContextEvent] = []
    turn: list[AssistantMessage | Thinking | ToolCall] = []
    for record in _flatten(records):
        if isinstance(record, (AssistantMessage, Thinking, ToolCall)):
            turn.append(record)
            continue
        if turn:
            entries.append(_response(turn))
            turn = []
        if isinstance(
            record,
            (UserMessage, AgentToAgentMessage, SystemMessage, TurnContext, ToolResult),
        ):
            entries.append(record)
    if turn:
        entries.append(_response(turn))
    return entries


def _response(records: Sequence[AssistantMessage | Thinking | ToolCall]) -> Response:
    """Return a stored model turn as the ``Response`` it came in."""
    return Response(
        records=tuple(records),
        tokens=TokenCount(),
        spend=TokenCost(),
        stop_reason="",
    )


def _flatten(records: Iterable[SessionRecord]) -> Iterator[SessionRecord]:
    """Yield ``records``, a ``ContextClear`` replaced by its summary and history."""
    for record in records:
        if isinstance(record, ContextClear):
            if record.summary is not None:
                yield UserMessage(content=record.summary)
            yield from _flatten(record.history)
        else:
            yield record


def _apply_standing(
    standing: dict[str, UncategorizedRecord],
    record: SessionRecord,
) -> None:
    """Put ``record`` in force in ``standing`` if it stands, or withdraw its key."""
    if not isinstance(record, UncategorizedRecord):
        return
    key = record.payload.get("standing")
    if not isinstance(key, str):
        return
    if record.kind == "withdrawn":
        standing.pop(key, None)
    else:
        standing[key] = record


# A splice that leaves the history before it as it was, such as an empty note, or
# only restates results in place, as a prune does, appends; anything else rewrites
# the window.
# Returns ``None`` when ``current`` changes ``lines`` any other way.
def _grown(
    lines: Sequence[SessionRecord],
    current: Sequence[SessionRecord],
) -> tuple[dict[str, str], list[SessionRecord]] | None:
    """Return the results ``current`` restates in ``lines``, by call, and what it adds."""
    n = len(lines)
    if len(current) < n:
        return None
    restated: dict[str, str] = {}
    for old, new in zip(lines, current[:n], strict=True):
        if old == new:
            continue
        if not (
            isinstance(old, ToolResult)
            and isinstance(new, ToolResult)
            and new == _restate(old, new.text())
        ):
            return None
        restated[old.call_id] = new.text()
    return restated, list(current[n:])


def _restated(records: Iterable[SessionRecord]) -> list[SessionRecord]:
    """Return ``records``, flat, each result a later ``restated`` record names restated."""
    out: list[SessionRecord] = []
    for record in _flatten(records):
        if not (isinstance(record, UncategorizedRecord) and record.kind == "restated"):
            out.append(record)
            continue
        texts = from_plain(record.payload.get("results"), dict[str, str], default={})
        out = [
            _restate(r, texts[r.call_id])
            if isinstance(r, ToolResult) and r.call_id in texts
            else r
            for r in out
        ]
    return out


def _restate(result: ToolResult, text: str) -> UncategorizedToolResult:
    """Return ``result`` reading ``text``, as a prune leaves it."""
    return UncategorizedToolResult(
        call_id=result.call_id,
        content=text,
        is_error=result.is_error,
    )
