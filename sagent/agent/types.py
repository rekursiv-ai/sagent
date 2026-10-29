"""The agent, its session, and what keeps the session small.

Typical usage example:

  async with ModelCatalog() as catalog:
      opus = catalog["anthropic"]["opus"]
      async with Agent(model=opus, tools=tools) as agent:
          reply = await agent.send("fix the failing test")
          print(reply)
          agent.model = catalog["openai"]["gpt-5.6"]
          async for words in agent.stream("now run the tests"):
              print(words, end="")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from sagent.types import Resource


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
    from pathlib import Path

    from sagent.providers.catalog import CachedModel
    from sagent.providers.types import Model
    from sagent.runtime.runtime import Runtime
    from sagent.session.types import (
        Attachment,
        SessionRecord,
        UncategorizedRecord,
    )
    from sagent.tools.types import Tool


__all__ = [
    "Agent",
    "Factory",
    "Journal",
    "Session",
    "Shrinker",
]


class Agent(Resource, Protocol):
    """A model, its tools, and the session it keeps; ``async with`` runs it.

    Each operation lives on the part it acts on. An assignment takes effect at
    the next model call; ``clear``, ``run``, and ``schedule`` wait for the
    runtime's next safe point::

      agent
      ├─ model
      │  ├─ provider
      │  ├─ capability
      │  ├─ settings
      │  ├─ count(request)
      │  ├─ attachment_tokens(attachment)
      │  └─ stream(request)
      ├─ model_cache[key]
      ├─ account
      ├─ system_prompt
      ├─ allowed
      ├─ tools[name]
      │  └─ call(context, arguments)
      ├─ session
      │  ├─ id
      │  ├─ __iter__()
      │  ├─ history()
      │  ├─ append(*records)
      │  ├─ journal
      │  │  ├─ path
      │  │  ├─ append(*records)
      │  │  ├─ __iter__()
      │  │  ├─ history()
      │  │  └─ flush()
      │  ├─ compactor
      │  │  ├─ should_run()
      │  │  └─ run(instructions)
      │  ├─ pruner
      │  │  ├─ should_run()
      │  │  └─ run()
      │  ├─ clear()
      │  ├─ stand(key, record)
      │  └─ standing()
      ├─ runtime
      │  ├─ __aiter__()
      │  ├─ push(event)
      │  ├─ halt()
      │  ├─ schedule(work)
      │  ├─ jobs[id]
      │  │  ├─ call
      │  │  ├─ state
      │  │  └─ cancel()
      │  └─ foreground(call_id)
      ├─ spend_usd
      ├─ charge(usd, by=...)
      ├─ send(text, ...)
      └─ stream(text, ...)

    """

    @property
    def model(self) -> Model:
        """Assign to switch models; the next model call uses it."""
        ...

    @model.setter
    def model(self, model: Model) -> None:
        """Switch to ``model``; the agent owns it, and closes the one before."""
        ...

    @property
    def system_prompt(self) -> str | None:
        """What the model is told it is; ``None`` uses the default."""
        ...

    @system_prompt.setter
    def system_prompt(self, prompt: str | None) -> None:
        """Tell the model it is ``prompt`` from its next call; ``None`` for the default."""
        ...

    tools: Mapping[str, Tool]
    """What the model is offered, by name."""

    model_cache: dict[str, CachedModel]
    """Each model switched away from, by ``cache_key``; a switch back restores its settings."""

    account: str | None
    """The named login the model's provider signs in as; ``None`` for its default."""

    @property
    def allowed(self) -> frozenset[str] | None:
        """The tools the model may call, by name; ``None`` for all."""
        ...

    @allowed.setter
    def allowed(self, names: frozenset[str] | None) -> None:
        """Let the model call only ``names``; every tool is still offered."""
        ...

    session: Session
    """The conversation, what shrinks it, and where it is saved."""

    runtime: Runtime
    """What runs the turns: observe it, halt it, cancel its jobs."""

    spend_usd: float
    """What the model calls have cost, in USD; a resumed session's included."""

    def charge(self, usd: float, *, by: str) -> None:
        """Count spend no reply of its own brought, as a child's; halt past the cap."""
        ...

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
          RetryableError: The model call failed; its subclass says why.
          BudgetExhaustedError: The spend reached the agent's cap.
          RoundLimitError: The turn reached the agent's cap on rounds.
          asyncio.CancelledError: Someone halted the turn.
          RuntimeError: The agent is not open, or its runtime stopped.

        """
        ...

    def stream(
        self,
        text: str,
        *,
        attachments: Sequence[Attachment] = (),
    ) -> AsyncIterator[str]:
        """Send ``text``; yield the reply's words as they are written, until idle.

        Every event, tool calls included, is ``runtime``'s to watch.

        Args:
          text: What the model reads.
          attachments: Images and documents sent with it.

        Yields:
          words: The next few words of the reply.

        Raises:
          RetryableError: The model call failed; its subclass says why.
          BudgetExhaustedError: The spend reached the agent's cap.
          RoundLimitError: The turn reached the agent's cap on rounds.
          asyncio.CancelledError: Someone halted the turn.
          RuntimeError: The agent is not open, or its runtime stopped.

        """
        ...


class Session(Protocol):
    """The conversation the model sees, saved as it grows."""

    id: str
    """The name to resume it by."""

    compactor: Shrinker
    """What starts the session over from a summary."""

    pruner: Shrinker
    """What drops old tool results."""

    journal: Journal
    """Where the records are written."""

    def __iter__(self) -> Iterator[SessionRecord]:
        """Yield what the model sees: the last clear onward."""
        ...

    def history(self) -> Iterator[SessionRecord]:
        """Yield the whole session from its first record: every window, compacted ones too."""
        ...

    def append(self, *records: SessionRecord) -> None:
        """Add ``records`` to the end; the journal writes them after the loop's pass."""
        ...

    async def clear(self) -> None:
        """Schedule a start over with nothing kept, and wait; the journal keeps the past."""
        ...

    def stand(self, key: str, record: UncategorizedRecord | None) -> None:
        """Keep ``record`` in force under ``key``, through every window; ``None`` withdraws it."""
        ...

    def standing(self) -> list[UncategorizedRecord]:
        """Return the records in force, as a resume reads them."""
        ...


class Journal(Protocol):
    """Where a session's records are written."""

    @property
    def path(self) -> Path | None:
        """The file; ``None`` when the records live only in memory."""
        ...

    def append(self, *records: SessionRecord) -> None:
        """Write ``records`` after the last."""
        ...

    def __iter__(self) -> Iterator[SessionRecord]:
        """Yield the records from the last ``ContextClear`` on, as a resume reads them."""
        ...

    def history(self) -> Iterator[SessionRecord]:
        """Yield every record, from the first: each window, compacted ones included."""
        ...

    def flush(self) -> None:
        """Write what is buffered."""
        ...


class Shrinker(Protocol):
    """Keeps its session small: by a summary, by stubbing old results, ..."""

    def should_run(self) -> bool:
        """Return whether the session has grown enough to shrink."""
        ...

    async def run(self, instructions: str = "") -> None:
        """Schedule the shrink at the runtime's next safe point, and wait for it.

        Args:
          instructions: What must survive.

        Raises:
          ValueError: ``instructions`` were given to a shrinker that takes none,
            such as a pruner.
          NotImplementedError: This shrinker never shrinks.

        """
        ...


class Factory[T](Protocol):
    """Builds a part of an agent, bound to that agent; a class is one."""

    def __call__(self, agent: Agent) -> T:
        """Return the part for ``agent``."""
        ...
