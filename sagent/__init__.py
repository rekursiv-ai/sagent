"""sagent -- a Python agent library.

::

    from sagent import tools
    from sagent.agent import Agent
    from sagent.providers import Google
    from sagent.lib.json import json_freeze

    agent = Agent(
        model=Google.from_env().model("gemini-3.1-pro-preview"),
        system="You are a scientist.",
        tools=[tools.Bash(), tools.Read(), tools.Grep()],
    )
    result = await agent.run(json_freeze({"prompt": "analyze ./data/"}))

This module docstring is the authoritative reference for sagent's
cross-cutting design contracts. Submodule docstrings cover
implementation details within their scope.

Agent
-----
An Agent combines four parts:

- **Model** -- the LLM endpoint, from a **Provider**.
- **Tools** -- the agent's capabilities (``run(msg) -> Message``).
- **System prompt** -- static string or dict of named sections.
- **Compactor** -- summarizes conversation when the context window
  fills. Optional; without one the agent runs until it hits the
  window limit.

Inbox spine
-----------
The agent IS its inbox: a per-agent deque of tagged Messages plus a
handler registry. Every signal -- user input, model response, tool
batch, abort, quit, status update -- is a ``Message`` routed through
one dispatch loop to descriptor-keyed handlers.

::

    while True:
        msg = await inbox.get()
        if msg.descriptor == "text/x-quit":
            return
        for h in handlers[msg.descriptor]:
            await h.handle(agent, msg)
        for h in wildcards:
            await h.handle(agent, msg)

Empty deque + no in-flight spawned tasks = idle. There is no
separate ``events`` queue, no flag-passing channel between tools and
the loop -- the deque is the spine.

Priority is insertion order: urgent messages go to the front
(``put_left``), everything else appends at the back.

Handlers
--------
A :class:`Handler` declares ``descriptors: tuple[str, ...]`` and
``spawn: bool``. The dispatch loop pops a message and runs every
handler subscribed to its descriptor (then any wildcards
``descriptors=()``). Inline handlers run on the loop. Spawned
handlers run as ``asyncio.Task``s and post their results back as
follow-up messages.

State ownership: each piece of mutable agent state has exactly one
owning handler. ``HistoryHandler`` is the sole writer of
``agent.history``; ``ActivityHandler`` owns ``agent.activity``;
``ClearHandler`` owns the wipe transition. Other handlers post
request messages.

The standard handler set is :func:`agent.handlers.core_handlers`.
Surface bundles like :func:`repl.repl_handler_set` add render
handlers; tests register stubs for inspection.

Message
-------
The universal unit of exchange. Every input, output, tool call,
tool result, event, and error is a ``Message``. There is no
``ToolResult`` or ``UserTurn`` class -- just ``Message`` with a
``descriptor`` that says what it carries.

::

    type MessageContent = str | bytes | JSON | tuple[Message, ...]

    @dataclass(frozen=True)
    class Message:
        content: MessageContent
        descriptor: str       # MIME-style type tag
        id: int               # process-wide auto-increment
        parent_id: int        # originating Message.id; -1 = unset
        timestamp: int        # nanosecond epoch

``content`` is recursive: a compound message (user message, tool
result) holds ``tuple[Message, ...]`` of atomic parts. Branch on
``descriptor`` to determine what a Message is, never on
``isinstance(content)``.

Descriptors
-----------
The single source of truth for all known descriptors and their
content-type groups is ``lib/descriptors.py``.  See the constants
``TEXT_DESCRIPTORS``, ``IMAGE_DESCRIPTORS``, ``BINARY_DESCRIPTORS``,
``JSON_DESCRIPTORS``, ``MULTIPART_DESCRIPTORS``, and
``ALL_DESCRIPTORS`` defined there.

Tool IDs (``application/x-tool-<name>``) are dynamic -- one per tool
class -- and validated at registration time, not in the registry.

Tool contract
-------------
Message in, Message out. A tool receives a ``Message`` containing
the LLM's directive and returns a ``Message`` containing the
result. That is the entire execution interface.

- ``run(msg) -> Message`` -- execute the directive. Returns a
  ``Message`` with ``text/plain`` on success or ``text/x-error``
  on failure. Never raises -- see error policy below.
- ``directive_schema`` -- JSON Schema describing valid directives.
- ``prompt() -> str | None`` -- per-request system prompt section.
  ``None`` = no change (avoids cache invalidation).
- ``help(msg) -> str`` -- human-readable invocation summary.

Agent is **Tool-shaped but not Tool-conforming**: it has the
metadata fields (``name``, ``description``, ``directive_schema``,
``summary``, ``prompt``) but its ``run`` returns a streaming +
awaitable :class:`RunHandle` rather than a flat ``Message``, and
takes a JSON directive rather than a tool-call ``Message``. The
real ``Tool`` that wraps an Agent for recursive composition is
:class:`tools.AgentSpawn`: it constructs a child Agent from the
parent's directive, drives it via ``child.run(...)``, forwards
events to the parent's inbox, and returns the final ``Message``.

In other words: agents compose recursively *via* ``AgentSpawn``,
not by sticking an ``Agent`` instance into a ``tools=[...]`` list.

Model contract
--------------
``buffer()`` and ``stream()`` send a ``ModelRequest`` and return a
``ModelResponse``. They raise on failure (``PromptTooLongError``,
``RateLimitError``, ``StreamInterruptedError``). The Agent's retry
and compaction machinery handles recovery.

``is_context_overflow(error)`` returns ``True`` if the error is a
context-window overflow.

Compactor contract
------------------
- ``should_compact()`` -- whether to compact this request.
- ``compact()`` -- summarize messages into a compact list. Returns
  a failure Message if all retry attempts are exhausted (not raise --
  see error policy below).
- ``maintain()`` -- between-request context maintenance (e.g. clear
  stale tool results).

CompactRestorable
-----------------
Optional protocol for tools that need to restore state after
compaction (e.g. re-inject invoked skill bodies). The Agent calls
``post_compact_restore(messages, tool_state, budget_chars=N)`` on
tools that implement it.

Error policy
------------
**If your output enters the conversation, return a Message.
If your output is infrastructure plumbing, raise.**

This yields three patterns:

1. **Tools return error Messages.** A tool's ``run()`` returns
   ``Message`` by contract, so errors are ``text/x-error`` parts.
   The LLM sees the error as a tool result and can self-correct.
   ``_invoke_tool_safe`` is the safety net for unexpected
   exceptions -- it wraps them in ``<tool_use_error>`` so the
   LLM still sees something actionable.

2. **Infrastructure raises.** Model and Provider methods don't
   return Messages -- they return ``ModelResponse`` or build
   objects. Errors are exceptions. The Agent catches and handles:
   retry, compact, disable, or surface to the user. The LLM
   never sees these exceptions directly.

3. **Optional enrichment is individually isolated.** Post-compact
   steps (file re-attachment, tool state restoration, background
   status) are best-effort. Each is wrapped in its own try/except
   that logs and continues. One failure does not cascade -- it
   must not disable compaction or crash the conversation.

Compactor is illustrative: ``compact()`` returns ``list[Message]``
(a Message-returning interface), so on exhausted retries it
returns a failure Message rather than raising. The LLM needs to
know context was lost so it can recover.
"""

from sagent import providers, tools
from sagent.agent import Agent


__all__ = [
    "Agent",
    "providers",
    "tools",
]
