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

Inbox zero
----------
The agent loop pursues inbox zero: process everything, then go idle.

::

    while True:
        drain inbox -> inject as user messages
        call LLM
        if tool calls: dispatch, loop back to drain
        if inbox empty and LLM done: go idle

The inbox is the ONLY entry point for user-message content. User
prompts, background results, agent-to-agent messages -- all go
through ``inbox.put()`` / ``inbox.put_left()``. There is ONE
drain point (``_drain_inbox``) at the top of each loop iteration.
No other code reads from the inbox or appends user messages to the
message history.

Context-affecting slash commands are inbox messages too. For example,
``/clear`` must be enqueued with ``put_left`` and interpreted by
``_drain_inbox``; REPL/keybinding code must not clear messages or
mutate context directly. REPL-local commands that do not touch model
context, such as ``/model`` display/swap and ``/login``, may be handled
by the surface before they enter the inbox.

Priority is insertion order: user messages go to the front
(``put_left``), everything else appends at the back.

Two queues connect the Agent to its surface (REPL, Slackbot,
parent agent):

- ``inbox`` (Deque[str]) -- inbound.
- ``events`` (Queue[Message | None]) -- outbound. Every observable
  side effect flows here as a typed Message. ``None`` = request
  boundary.

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

An Agent is itself a Tool (``run`` takes a prompt, returns the
final response), so agents compose recursively.

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
