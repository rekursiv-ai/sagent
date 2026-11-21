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

from collections import Counter
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, override

import asyncio
import re
import uuid

from treekle import PlainTree, from_plain

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
from sagent.runtime.tape import (
    ContextSplice,
    ReferrableTapeEvent,
    parts_model_turns,
)
from sagent.session.recording import RecordingJournal
from sagent.session.types import (
    AgentIdle,
    AgentMessage,
    AssistantMessage,
    Cancelled,
    ContextClear,
    Failed,
    ModelIdle,
    RequestMeasurement,
    SessionContext,
    Succeeded,
    SystemMessage,
    Thinking,
    TokenUsage,
    ToolCall,
    ToolMessage,
    UncategorizedRecord,
    UserMessage,
    now_timestamp,
)
from sagent.tools.types import Briefing, Reminding, Skipping, ToolContext
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
    from datetime import date, datetime, timedelta

    from sagent.agent.types import (
        Agent as AgentProtocol,
        Factory,
        Journal,
        Session,
        Shrinker,
    )
    from sagent.model.token_accounting import TokenAccounting
    from sagent.providers.types import Model, ModelCatalog
    from sagent.runtime.tape import TapeRecord
    from sagent.session.types import Attachment, SessionRecord
    from sagent.tools.types import Tool


__all__ = [
    "Agent",
    "NoShrinker",
    "RuntimeSession",
    "default_system_prompt",
    "spent",
    "spent_by_model",
]


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

    async def run(self, instructions: str = "") -> None:  # pragma: no mutate -- unread
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
      journal: Where the session is written; in memory when omitted, so
        nothing resumes it. A journal that opens, such as ``JsonlJournal``,
        is opened by the caller.
      compactor: Builds what starts the session over from a summary.
      pruner: Builds what drops old tool results.
      tool_context: What the tools start with: where they work, where the session
        started (``cwd`` when unset), how much a result may hold, the width
        output shows in, and the fetch session; ``ToolContext()`` when omitted.
        The agent works on a copy, so a ``cd`` never moves the one given, but
        the copy's fields are the given's own objects: two agents given one
        context share its file tracker.
      max_rounds: Tool-call rounds a turn may run; ``None`` for no cap.
      max_spend_usd: What the agent may spend; ``None`` for no cap.
      persistent_retry: Whether a long rate limit is waited out, not raised.
      agents_md: Where the AGENTS.md files are looked for; the defaults when
        omitted.
      allowed: The tools the model may call, by name; ``None`` for all. The
        rest are still offered, so every agent's tools cache alike.
      environment: Whether the model is told where it runs: its directory, the
        git state, platform and shell. Off for an agent serving people the
        host's details are not for.
      account: The named login ``model``'s provider signs in as; ``None`` for
        its default. Whoever switches the model keeps it current.
      user: Whose session it is, which the model is told: a person's message
        that names no sender is theirs. ``None`` tells the model no name.

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
        tool_context: ToolContext | None = None,
        max_rounds: int | None = None,
        max_spend_usd: float | None = None,
        persistent_retry: bool = False,
        agents_md: AgentsMdConfig | None = None,
        allowed: Collection[str] | None = None,
        environment: bool = True,
        account: str | None = None,
        user: str | None = None,
    ) -> None:
        self.agents_md: AgentsMdConfig = (
            agents_md if agents_md is not None else AgentsMdConfig()
        )
        given = tool_context if tool_context is not None else ToolContext()
        kept = journal.path if journal is not None else None
        self.runtime: Runtime = Runtime(
            model=model,
            tools=tools,
            # Where it started is fixed now: a ``cd`` must not move where its
            # AGENTS.md and skills are read from.
            tool_context=replace(
                given,
                root=given.origin,
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
        self.tools: Mapping[str, Tool] = self.runtime.tools
        self.environment = environment
        self.account = account
        self.user = user
        self.max_rounds = max_rounds
        self._rounds = 0
        self._allowed = frozenset(allowed) if allowed is not None else None
        self.allowed = self._allowed
        self._system_prompt = system_prompt
        self._session = RuntimeSession(
            self,
            journal=journal if journal is not None else RecordingJournal(),
            compactor=compactor,
            pruner=pruner,
        )
        self.session: Session = self._session
        # What a child it spawns compacts with, so the child compacts as it does.
        self.compactor_factory: Factory[Shrinker] = compactor
        self.runtime.context_updates = self._session.updates
        # A compactor the runtime can call compacts on ``SessionCompact``, and when due,
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
        self.max_spend_usd = max_spend_usd
        self.runtime.persistent_retry = persistent_retry
        # What the model calls have cost, a resumed session's included.
        self.spend_usd = 0.0
        # Models assigned away from, closed once no call can still be on them.
        self._retired: list[Model] = []
        self._closing: set[asyncio.Task[None]] = set()
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
    # streaming, so it closes once the context naming its successor is in force: the
    # runtime puts it there just before it calls the new one.
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
        """Configured tool permissions, inherited by children; ``None`` for all.

        ``runtime.allowed`` additionally applies the current turn's round cap.
        """
        return self._allowed

    # Every tool stays offered, so a change of gate keeps the cached prefix; the
    # next round states the new gate to the model.
    @allowed.setter
    def allowed(self, names: frozenset[str] | None) -> None:
        if names is not None:
            unknown = sorted(names - set(self.tools))
            if unknown:
                raise ValueError(f"Unknown tools: {', '.join(unknown)}.")
        self._allowed = names
        self._gate_tools()

    def charge(
        self,
        spend: TokenCost,
        *,
        label: str,
        measurement: RequestMeasurement | None = None,
    ) -> None:
        """Count spend no reply of this agent's own brought; halt the turn past the cap.

        The spend is journaled, so a resumed session counts it too, and told to the
        runtime's observers as ``SessionSpendCharged``.

        Args:
          spend: What one model call cost.
          label: The label of the agent that spent it: one below this one, or
            this one's own for a compaction.
          measurement: The exact request's measurement for a side call, if captured.

        """
        self.spend_usd += spend.total
        self._session.journal.append(
            TokenUsage(
                timestamp=now_timestamp(),
                info={"spend_usd": spend.total, "agent": label},
                measurement=measurement,
            ),
        )
        self.runtime.publish(types.SessionSpendCharged(spend=spend, label=label))
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
        await self._session.aresume()
        await self._close_retired()
        self.runtime.observers.append(self._observe)
        self._loop = asyncio.create_task(self.runtime.run_forever())

    @override
    async def aclose(self) -> None:
        loop, self._loop = self._loop, None
        try:
            if loop is not None:
                self.runtime.push(types.SessionQuit())
                await loop
        finally:
            # What the session holds is written even when the loop died raising.
            # An agent closed twice, or never opened, has no observer to remove.
            if self._observe in self.runtime.observers:
                self.runtime.observers.remove(self._observe)
            self._session.persist()
            self.session.journal.flush()
            self._retired.append(self.runtime.model)
            await asyncio.gather(*self._closing)
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
            if isinstance(event, types.ModelResponseCompleted) and isinstance(
                event.outcome,
                Succeeded,
            ):
                said = "".join(
                    r.content or ""
                    for r in event.outcome.value.records
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
            if isinstance(event, types.ModelResponseTextStreamed):
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
        queue: asyncio.Queue[tuple[types.RuntimeEvent, bool]] = asyncio.Queue()
        # A cap tripped outside a turn, by a message no turn sent, is not this one's.
        self._limit = None
        ended = self.runtime.submit(
            UserMessage(content=text, attachments=tuple(attachments)),
        )

        # The message ends just before its ending is told, so the event that tells
        # it is the first heard with it ended: read later, that is still so.
        def observer(event: types.RuntimeEvent) -> None:
            queue.put_nowait((event, ended.done()))

        answered = False  # pragma: no mutate[truthiness]
        self.runtime.observers.append(observer)
        try:
            while True:
                event, ended_by_then = await self._next(queue)
                yield event
                if isinstance(event, AgentIdle):
                    return
                # The message's own end, whatever became of it: a turn that stopped
                # before it was taken in is the one that ran before it. Once it is
                # answered, a stop before the agent rests is the turn's too, as when
                # the reply that answers it reaches a cap.
                if ended_by_then and not answered:
                    answered = True
                    outcome = ended.result()
                elif answered and isinstance(event, ModelIdle):
                    outcome = event.outcome
                else:
                    continue
                if isinstance(outcome, Failed):
                    raise outcome.exception
                if isinstance(outcome, Cancelled):
                    limit, self._limit = self._limit, None
                    if limit is not None:
                        raise limit
                    # A runtime that quit ends its loop: the wait for its next
                    # event raises that.
                    if not self.runtime.has_quit:
                        raise asyncio.CancelledError
        finally:
            self.runtime.observers.remove(observer)

    # The loop raises what it cannot recover from; waiting on its events alone would
    # then wait forever.
    async def _next[T](self, queue: asyncio.Queue[T]) -> T:
        """Return the next of ``queue``, or raise why the runtime's loop ended first."""
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

    async def _close_retired(self) -> None:
        """Close each model assigned away from, once."""
        retired = self._retired
        self._retired = []
        for model in retired:
            await model.aclose()

    # The runtime publishes a halt before it stops anything, so the turn it will stop
    # is still under way here, and the record precedes what the stop writes. What is
    # at work says less: nothing is, in a wait to call again. The record is the
    # journal's alone: a resume never reads it, so the model is not told.
    def _observe(self, event: types.RuntimeEvent) -> None:
        """Write the session, note a halt that stops a turn, start windows, hold caps."""
        if isinstance(event, types.SessionSaveDue):
            self._session.persist()
        # A call on the model now, or none at all: no call is still on a retired one.
        elif (
            isinstance(event, types.ModelResponseStarted | AgentIdle) and self._retired
        ):
            closing = asyncio.create_task(self._close_retired())
            self._closing.add(closing)
            closing.add_done_callback(self._closing.discard)
        elif isinstance(event, types.ModelHalt) and self.runtime.turn_under_way:
            self._session.persist()
            self._session.journal.append(
                UncategorizedRecord(timestamp=now_timestamp(), kind="interrupted"),
            )
        elif isinstance(
            event,
            types.SessionClearCompleted | types.SessionCompactCompleted,
        ) and isinstance(event.outcome, Succeeded):
            self.runtime.tool_context.files.forget()
            self._return_home()
            self._session.frame()
        # The next turn starts afresh: a persistent child's with another agent's
        # message, which mid-turn, as a child's idle ping, starts none.
        elif isinstance(event, (UserMessage, AgentIdle)):
            self._new_turn()
        elif isinstance(event, types.ModelResponseCompleted) and isinstance(
            event.outcome,
            Succeeded,
        ):
            self._charge(event.outcome.value)

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
        elif called and self.max_rounds is not None and self._rounds >= self.max_rounds:
            self._limit = RoundLimitError(rounds=self._rounds, max_rounds=self._rounds)
        elif called:
            self._rounds += 1
            self._gate_tools()
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

    def _new_turn(self) -> None:
        """Count the rounds from none, with the tools the round cap took given back."""
        self._rounds = 0
        self._gate_tools()

    def _gate_tools(self) -> None:
        """Apply configured permissions and the turn's round cap to future calls."""
        self.runtime.allowed = (
            frozenset({"AgentSend"})
            & frozenset(self.tools if self.allowed is None else self.allowed)
            if self.max_rounds is not None and self._rounds >= self.max_rounds
            else self.allowed
        )

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
                timestamp=now_timestamp(),
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
        # How much of the tape is written: none, until a resume writes what it read.
        self._written = 0
        # What the journal's window reads as, so a merge can write only its part.
        self._lines: list[SessionRecord] = []
        self._standing: dict[str, UncategorizedRecord] = {}
        # The model and prompt last stood, so each stands again only when it changes.
        self._memory: ModelMemory | None = None
        self._prompt: str | None = None
        # The last context this session stated, and the skipped files it meant:
        # its ``()`` says either "as before" or "none now", which it alone knows.
        self._stated_skipped: tuple[SessionContext, tuple[Path, ...]] | None = None
        self.frame()

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
        # Where the model works and what it has seen, when the window's AGENTS.md
        # files were last looked for: at its start, from the root with nothing seen.
        self._seen: tuple[Path, tuple[Path, ...]] = (agent.root, ())
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
        return iter(_records(self._agent.runtime.tape.context().messages))

    def history(self, start: int = 0) -> Iterator[SessionRecord]:
        """Yield the session from record ``start``: every window, compacted ones too.

        What the runtime added since the last write is written first.

        Args:
          start: The number of the first record; those before it are not read.

        Returns:
          records: The session's records from ``start``, oldest first, read as
            they are iterated.

        """
        self.persist()
        return self.journal.history(start)

    def append(self, *records: SessionRecord) -> None:
        """Add ``records`` to the end; the journal writes them after the loop's pass."""
        for entry in _entries(records):
            self._agent.runtime.tape.append_history(entry)

    async def clear(self) -> None:
        """Schedule a start over with nothing kept, and wait; the journal keeps the past.

        The new window reads its system prompt afresh, as the agent frames it.

        Raises:
          asyncio.CancelledError: The runtime quit before it cleared.

        """
        if isinstance(
            await self._agent.runtime.ask(types.SessionClear()),
            Cancelled,
        ):
            raise asyncio.CancelledError

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

    # Turn and window ids are counts, so a resume goes on counting from the journal
    # or would reuse an earlier turn's. The whole journal is read for them, on a
    # worker thread: a long session's takes seconds.
    async def aresume(self) -> None:
        """Resume, then go on counting turns and windows from what the journal held."""
        held = self.journal.history()
        self.resume()
        turns, compactions = await asyncio.to_thread(
            _counted,
            _restored(held, accounting=self._agent.runtime.token_accounting),
        )
        self._agent.runtime.turns += turns
        self._agent.runtime.compactions += compactions

    def resume(self) -> None:
        """Load the journal's last window into the runtime, or start one."""
        window = list(self.journal)
        if window:
            self.append(*_restated(window))
            for record in _restored(
                window,
                accounting=self._agent.runtime.token_accounting,
            ):
                _apply_standing(self._standing, record)
            stood = {r.kind: r.payload for r in self.standing()}
            if "model" in stood:
                self._memory = from_plain(stood["model"], ModelMemory)
            if "system_prompt" in stood:
                self._prompt = from_plain(stood["system_prompt"], _Prompt).text
            self._agent.spend_usd = spent(window)
            self._lines = list(self)
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
        # What came before each splice, then the splice with what follows it.
        ends = [
            i
            for i, record in enumerate(tape[self._written :], self._written)
            if isinstance(record, ContextSplice)
        ]
        for end in (*ends, len(tape)):
            if end != self._written:
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
                (types.SessionCompactStarted, types.SessionCompactCompleted),
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
                UncategorizedRecord(
                    kind="restated",
                    payload={"messages": [list(each) for each in restated]},
                ),
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
                timestamp=now_timestamp(),
                system_prompt="\n\n".join(self._agent.runtime.system),
                history=history,
                # What was spent before this window, for a resume to go on from.
                extra={"spend_usd": self._agent.spend_usd},
            ),
            *self._standing.values(),
        )
        self._lines = list(history)

    def updates(self) -> list[types.SessionContextUpdate | SystemMessage]:
        """Return what to restate before the next model call, if it changed.

        The runtime asks once per model round. The window's first states every
        setting; each later one only the settings changed since, and the time
        once the hour turns. An AGENTS.md file is given once a window: one of a
        directory the model works in or reads from, as it gets there, and a
        rule whose ``paths:`` match a file the model has seen, as it sees one. A
        tool's reminder is told whenever it differs from the last it gave, and
        withdrawn when it empties.

        Returns:
          updates: A context to put in force, a ``SystemMessage`` of AGENTS.md
            files, and reminders, each only when there is something new to say.

        """
        context = self._agent.runtime.tape.context().messages
        out: list[types.SessionContextUpdate | SystemMessage] = []
        turn = self._turn(context)
        if turn is not None:
            out.append(
                types.SessionContextUpdate(timestamp=turn.timestamp, context=turn),
            )
        tool_context = self._agent.runtime.tool_context
        seen = (tool_context.cwd, tuple(tool_context.files.recent()))
        if seen != self._seen:
            self._seen = seen
            cwd, files = seen
            given = {
                key
                for entry in context
                if isinstance(entry, SystemMessage) and entry.subtype == "rules"
                for key in from_plain(entry.extra.get("rules"), list[str])
            }
            text, matched = self._directives.found(
                [cwd, *(f.parent for f in files)],
                files,
                given=given,
            )
            if text:
                out.append(
                    SystemMessage(
                        timestamp=now_timestamp(),
                        content=text,
                        subtype="rules",
                        extra={"rules": sorted(matched)},
                    ),
                )
        out.extend(self._reminders(context))
        return out

    def _skipped_instruction_files(self) -> tuple[Path, ...]:
        """Return each AGENTS.md, rule, and skill file left unread, sorted."""
        tool_context = self._agent.runtime.tool_context
        return tuple(
            sorted(
                {
                    *self._directives.skipped,
                    *(
                        path
                        for tool in self._agent.tools.values()
                        if isinstance(tool, Skipping)
                        for path in tool.skipped(tool_context)
                    ),
                },
            ),
        )

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
        allowed = self._agent.runtime.allowed
        out: list[SystemMessage] = []
        for name, tool in self._agent.tools.items():
            # Offered for the cache but gated off, a tool's reminder would ask for
            # calls the model may not make, such as a reply by AgentSend.
            if not isinstance(tool, Reminding):
                continue
            text = (
                tool.reminder(tool_context)
                if allowed is None or name in allowed
                else ""
            )
            last = said.get(name, "")
            if text == last:
                continue
            out.append(
                SystemMessage(
                    timestamp=now_timestamp(),
                    content=text or f"No longer in force: {last}",
                    subtype="reminder",
                    extra={"tool": name, **({} if text else {"withdrawn": True})},
                ),
            )
        return out

    def _turn(
        self,
        context: Sequence[types.ModelContextEvent],
    ) -> SessionContext | None:
        """Return the settings changed since the window last stated them, or ``None``."""
        stated = [entry for entry in context if isinstance(entry, SessionContext)]
        agent = self._agent
        cwd = agent.runtime.tool_context.cwd
        model = agent.model
        settings = model.settings
        offered = frozenset(agent.tools)
        skipped = self._skipped_instruction_files()
        current = _Stated(
            user=agent.user,
            cwd=cwd if agent.environment else None,
            environment=environment(
                cwd,
                knowledge_cutoff=model.capability.knowledge_cutoff,
            )
            if agent.environment
            else None,
            model=ModelTags.from_settings(model.capability, settings).name,
            tools=_gate(agent.runtime.allowed, offered=offered),
            skipped_instruction_files=skipped,
        )
        now = now_timestamp()
        # A window's first context names skipped files only when there are some.
        delta = replace(
            current,
            skipped_instruction_files=current.skipped_instruction_files or None,
        )
        if stated:
            last = _last_stated(stated, offered=offered)
            if (
                self._stated_skipped is not None
                and self._stated_skipped[0] is stated[-1]
            ):
                last = replace(
                    last,
                    skipped_instruction_files=self._stated_skipped[1],
                )
            delta = _Stated(
                user=current.user if current.user != last.user else None,
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
                skipped_instruction_files=current.skipped_instruction_files
                if current.skipped_instruction_files != last.skipped_instruction_files
                else None,
            )
            if delta == _Stated() and _hour(stated[-1].timestamp) == _hour(now):
                return None
        restated = SessionContext(
            timestamp=now,
            user=delta.user,
            cwd=delta.cwd,
            environment=delta.environment,
            model=delta.model,
            tools=delta.tools,
            skipped_instruction_files=delta.skipped_instruction_files or (),
        )
        self._stated_skipped = (restated, skipped)
        return restated


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
    return _tallied(records)[0]


def spent_by_model(records: Iterable[SessionRecord]) -> dict[str, float]:
    """Return what an agent's journal says it spent on each model, from its start.

    Each ``TokenUsage`` counts toward the model the last ``SessionContext`` stated. A
    ``ContextClear`` states what was spent before it: what it states past the sum so
    far counts toward the model then in force, so read from the start the sums add up
    to :func:`spent`'s total.

    Args:
      records: The journal's records, oldest first, from its start.

    Returns:
      usd: Per model, as its ``SessionContext`` names it (``sol-6.1+fullcontext``), what
        was spent; ``""`` for spend before any model was stated.

    """
    return _tallied(records)[1]


def _tallied(records: Iterable[SessionRecord]) -> tuple[float, dict[str, float]]:
    """Return what a journal says was spent, as :func:`spent` reads it, and per model."""
    usd = 0.0
    by_model: dict[str, float] = {}
    model = ""
    for record in records:
        if isinstance(record, SessionContext) and record.model is not None:
            model = record.model
        elif isinstance(record, ContextClear):
            usd = from_plain(record.extra.get("spend_usd"), float, default=0.0)
            if (missing := usd - sum(by_model.values())) > 0:
                by_model[model] = by_model.get(model, 0.0) + missing
        elif isinstance(record, TokenUsage):
            turn = from_plain(record.info.get("spend_usd"), float, default=0.0)
            usd += turn
            by_model[model] = by_model.get(model, 0.0) + turn
    return usd, by_model


@dataclass(frozen=True, slots=True, kw_only=True)
class _Prompt:
    """The system prompt an agent was given, as it stands in its journal."""

    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _Stated:
    """The settings a ``SessionContext`` states; ``None`` for each it leaves as was."""

    user: str | None = None
    cwd: Path | None = None
    environment: str | None = None
    model: str | None = None
    tools: tuple[str, ...] | None = None
    """The tools the model may call; ``None`` for every one offered."""

    skipped_instruction_files: tuple[Path, ...] | None = None


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
    stated: Iterable[SessionContext],
    *,
    offered: frozenset[str],
) -> _Stated:
    """Return the settings in force after ``stated``, each the last any of them said."""
    # A journal from before the field said nothing was skipped.
    last = _Stated(skipped_instruction_files=())
    for turn in stated:
        last = _Stated(
            user=turn.user if turn.user is not None else last.user,
            cwd=turn.cwd if turn.cwd is not None else last.cwd,
            environment=turn.environment
            if turn.environment is not None
            else last.environment,
            model=turn.model if turn.model is not None else last.model,
            tools=_gate(turn.tools, offered=offered)
            if turn.tools is not None
            else last.tools,
            skipped_instruction_files=turn.skipped_instruction_files
            or last.skipped_instruction_files,
        )
    return last


def _prompt(prompt: str | None) -> str:
    """Return ``prompt``; ``None`` is the default prompt."""
    return prompt if prompt is not None else default_system_prompt()


# The offset too: when clocks go back, an hour comes twice.
def _hour(timestamp: datetime | None) -> tuple[date, int, timedelta | None] | None:
    """Return the hour a time falls in, here, where the model is told it."""
    if timestamp is None:
        return None
    here = timestamp.astimezone()
    return here.date(), here.hour, here.utcoffset()


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
                timestamp=now_timestamp(),
                info={
                    "request": tokens.request,
                    "response": tokens.response,
                    "cache_write": tokens.cache_write,
                    "cache_write_1h": tokens.cache_write_1h,
                    "cache_read": tokens.cache_read,
                    "spend_usd": spend.total,
                },
                measurement=entry.measurement,
            ),
        )
    return records


# Only a user turn, a result, or a note the wire sends (one with text, as a wake) ends
# a reply. Claude writes a line per block, each with its usage, so ending a reply at
# usage splits one reply into turns that break role alternation. No provider starts a reply before every call is answered, so a model
# record while a call is open joins the open reply; once the user has spoken, a call
# a killed run left unanswered no longer holds it open.
def _entries(records: Iterable[SessionRecord]) -> list[types.ModelContextEvent]:
    """Return records as history entries; each reply the model gave is one turn."""
    entries: list[types.ModelContextEvent] = []
    turn: list[AssistantMessage | Thinking | ToolCall] = []
    held: list[types.ModelContextEvent] = []
    calling: set[str] = set()
    # A usage after the reply's own is a child's.
    usage: TokenUsage | None = None
    priced = False  # pragma: no mutate[truthiness]
    for record in _flatten(records):
        if isinstance(record, AssistantMessage | Thinking | ToolCall):
            spoke = any(isinstance(e, UserMessage | AgentMessage) for e in held)
            answered = not calling and any(
                isinstance(e, ToolMessage)
                or (
                    isinstance(e, SystemMessage | SessionContext)
                    and parts_model_turns(e)
                )
                for e in held
            )
            if turn and (spoke or answered):
                entries.extend(_reply_entries(turn, held, usage=usage))
                turn.clear()
                held.clear()
                calling.clear()
                usage = None
            turn.append(record)
            priced = True
            if isinstance(record, ToolCall):
                calling.add(record.call_id)
            continue
        if isinstance(record, TokenUsage):
            if priced:
                usage = record
                priced = False  # pragma: no mutate[truthiness]
            continue
        if not isinstance(
            record,
            UserMessage | AgentMessage | SystemMessage | SessionContext | ToolMessage,
        ):
            continue
        priced = False  # pragma: no mutate[truthiness]
        if isinstance(record, ToolMessage):
            calling.discard(record.call_id)
        elif isinstance(record, UserMessage | AgentMessage) and not calling:
            entries.extend(_reply_entries(turn, held, usage=usage))
            turn.clear()
            held.clear()
            usage = None
            entries.append(record)
            continue
        (held if turn else entries).append(record)
    entries.extend(_reply_entries(turn, held, usage=usage))
    return entries


# A provider refuses a user turn between a call and its result.
def _reply_entries(
    turn: Sequence[AssistantMessage | Thinking | ToolCall],
    held: Sequence[types.ModelContextEvent],
    *,
    usage: TokenUsage | None,
) -> list[types.ModelContextEvent]:
    """Return the reply as one turn, then what came while it was open, users last."""
    reply = [_response(turn, usage=usage)] if turn else []
    spoken = [e for e in held if isinstance(e, UserMessage | AgentMessage)]
    rest = [e for e in held if not isinstance(e, UserMessage | AgentMessage)]
    return [*reply, *rest, *spoken]


def _response(
    records: Sequence[AssistantMessage | Thinking | ToolCall],
    *,
    usage: TokenUsage | None = None,
) -> Response:
    """Return a stored model turn as the ``Response`` it came in."""
    info = usage.info if usage is not None else dict[str, PlainTree]()
    return Response(
        records=tuple(records),
        tokens=from_plain(
            {
                key: info[key]
                for key in (
                    "request",
                    "response",
                    "cache_write",
                    "cache_write_1h",
                    "cache_read",
                )
                if key in info
            },
            TokenCount,
        ),
        spend=TokenCost(request=from_plain(info.get("spend_usd"), float, default=0.0)),
        stop_reason="",
        measurement=usage.measurement if usage is not None else None,
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


def _restored(
    records: Iterable[SessionRecord],
    *,
    accounting: TokenAccounting,
) -> Iterator[SessionRecord]:
    """Yield journal records while restoring their request measurements."""
    for record in records:
        if isinstance(record, TokenUsage) and record.measurement is not None:
            accounting.restore(record.measurement)
        yield record


def _counted(records: Iterable[SessionRecord]) -> tuple[int, int]:
    """Return turn and compaction counts from the journal."""
    turns = compactions = 0
    for record in records:
        if isinstance(record, (UserMessage, AgentMessage)):
            turns += 1
        elif isinstance(record, ContextClear) and record.history:
            compactions += 1
    return turns, compactions


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
) -> tuple[list[tuple[str, int, str]], list[SessionRecord]] | None:
    """Return each result ``current`` restates, by call and message, and what it adds."""
    n = len(lines)
    if len(current) < n:
        return None
    restated: list[tuple[str, int, str]] = []
    told = Counter[str]()
    for old, new in zip(lines, current, strict=False):
        if isinstance(old, ToolMessage):
            told[old.call_id] += 1
        if old == new:
            continue
        if not (
            isinstance(old, ToolMessage)
            and isinstance(new, ToolMessage)
            and new == old.restated(new.text())
        ):
            return None
        restated.append((old.call_id, told[old.call_id] - 1, new.text()))
    return restated, list(current[n:])


# A journal from before long calls told their news as results held a wake as a note
# naming its call, and a call's end under ``<id>:detached``: a ``foreground`` call the
# runtime made itself, after a note saying the call completed.
def _as_today(record: SessionRecord) -> SessionRecord | None:
    """Return ``record`` as today's journal holds it; ``None`` if it holds nothing so."""
    match record:
        case SystemMessage(subtype="wake", call_id=str() as call_id):
            return ToolMessage(
                timestamp=record.timestamp,
                call_id=call_id,
                state="pending",
                stdout=(record.content or "",),
                extra=record.extra,
            )
        case UserMessage(content=str() as content) if re.fullmatch(
            r"\[detached tool .+ completed\]",
            content,
        ):
            return None
        case ToolCall(call_id=call_id) if call_id.endswith(":detached"):
            return None
        case ToolMessage(call_id=call_id) if call_id.endswith(":detached"):
            return replace(record, call_id=call_id.removesuffix(":detached"))
        case _:
            return record


# A ``restated`` record names each result a prune stubbed: today as which message of
# its call it is, as a call run apart has several; before, by the id it was written
# under, when a call had one.
def _restated(records: Iterable[SessionRecord]) -> list[SessionRecord]:
    """Return ``records``, flat and as today's, restated as later records say."""
    out: list[SessionRecord] = []
    # The id each result of ``out`` was written under; ``None`` for any other record.
    written: list[str | None] = []
    for record in _flatten(records):
        if not (isinstance(record, UncategorizedRecord) and record.kind == "restated"):
            if (today := _as_today(record)) is not None:
                out.append(today)
                written.append(
                    record.call_id if isinstance(record, ToolMessage) else None,
                )
            continue
        if "messages" in record.payload:
            for call_id, nth, text in from_plain(
                record.payload["messages"],
                list[tuple[str, int, str]],
            ):
                i, result = [
                    (i, r)
                    for i, r in enumerate(out)
                    if isinstance(r, ToolMessage) and r.call_id == call_id
                ][nth]
                out[i] = result.restated(text)
            continue
        texts = from_plain(record.payload.get("results"), dict[str, str])
        out = [
            r.restated(texts[key])
            if isinstance(r, ToolMessage) and (key := written[i]) in texts
            else r
            for i, r in enumerate(out)
        ]
    return out
