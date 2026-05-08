"""Handler protocol and convenience base classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.custom_types import Message


@runtime_checkable
class Handler(Protocol):
    """Descriptor-routed message handler.

    Concrete handlers declare which descriptors they subscribe to and
    whether they run inline (block the dispatch loop) or spawned (as
    an asyncio.Task that posts results back to the inbox).

    A handler with ``descriptors = ()`` is a wildcard that fires on
    every message after specific handlers complete.

    All spawned handlers are abortable -- ``/abort`` cancels every task
    in ``agent.tasks``. Session-lifecycle infrastructure that must
    survive abort (the REPL input pump, daemons) does NOT live as a
    Handler; instead it spawns directly into ``agent.background_tasks``
    with ``hidden=True``. That separation eliminates the prior
    "abortable flag" wart and aligns the registry with bash semantics:
    foreground steps die on abort, background tasks survive until the
    agent itself shuts down.

    Attributes:
      descriptors: Tuple of descriptor strings this handler subscribes to.
      spawn: True to run as a task, False to run inline on the dispatch loop.

    """

    descriptors: tuple[str, ...]
    spawn: bool

    async def handle(self, agent: Agent, msg: Message) -> None:
        """Process one message.

        Args:
          agent: The agent dispatching this message.
          msg: The message to handle.

        """
        ...


class InlineHandler:
    """Base class for inline handlers (run on the dispatch loop).

    Use for ordering-critical work: history append, abort, state
    mutation. Long-running inline handlers freeze the loop; reach for
    :class:`SpawnedHandler` instead when work can take >50 ms.

    Attributes:
      descriptors: Tuple of descriptor strings this handler subscribes to.

    """

    spawn: bool = False
    descriptors: tuple[str, ...] = ()

    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent, msg
        raise NotImplementedError


class SpawnedHandler:
    """Base class for spawned handlers (run as ``asyncio.Task``).

    Use for long ops that should not block the dispatch loop: model
    calls, tool dispatch, subagent runs. The task posts its result(s)
    back to the inbox when done. ``/abort`` cancels every spawned
    handler unconditionally; long-running infra (input pump, daemons)
    must NOT extend this class -- spawn into ``agent.background_tasks``
    with ``hidden=True`` instead.

    Attributes:
      descriptors: Tuple of descriptor strings this handler subscribes to.

    """

    spawn: bool = True
    descriptors: tuple[str, ...] = ()

    async def handle(self, agent: Agent, msg: Message) -> None:
        del agent, msg
        raise NotImplementedError
