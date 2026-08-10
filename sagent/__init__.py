"""sagent -- inbox-driven LLM agent library.

::

    from sagent import tools
    from sagent.agent.agent import Agent
    from sagent.providers import Google

    agent = Agent(
        model=Google.from_env().model("gemini-3.1-pro-preview"),
        system="You are a scientist.",
        tools=[tools.Bash(), tools.Read(), tools.Grep()],
    )
    async for event in agent.run(UserMessage(text="analyze ./data/")):
        print(event)

See ``docs/private/agent_v4_contract.md`` for the binding spec.

Architecture
------------
An ``Agent`` composes :class:`AgentRuntime` -- the runtime owns the
single dispatch loop -- with rich-protocol wrappers
(:class:`_AgentModel`, :class:`_AgentTool`, :class:`_AgentCompactor`)
that bridge full provider / tool / compactor surfaces to the runtime's
minimal protocols. The runtime sees only its own protocols; everything
else is observers + adapters in the agent layer.

Runtime
-------
:class:`AgentRuntime.run_forever` drains a
``GatedDeque[RuntimeEvent]`` and dispatches each event in a ``match``
block. The match mutates instance state (``running_tools``, ``cohort``,
``model_call``, ``compact_task``, ``history``). After the batch, a
gate check fires the model when ``cohort`` is empty,
``model_call`` is None, ``compact_task`` is None, and history ends
with content the model should answer (``UserMessage`` or
``ToolResult``).

History
~~~~~~~
``list[UserMessage | AssistantMessage | ToolResult]``. Three frozen
dataclasses. Type-match, not string-tag.

Cohort
~~~~~~
``set[str]`` of pending tool ``call_id``s, populated from
``ModelResponseComplete.message.tool_calls``. The model fires only
when the cohort is empty.

Detach / splice
~~~~~~~~~~~~~~~
When the user preempts mid-cohort, the runtime stubs unfinished
tools with ``ToolResult(content="[detached]")`` placeholders and
moves their tasks to ``self.detached``. When the late
``DetachedResult`` arrives, the runtime splices the real content
into the placeholder's slot so history stays linear and the model
sees the real result in the slot it already expects.

Tools
-----
Three protocols stacked:

- ``runtime.Tool`` (in ``agent/runtime.py``) -- ``name`` + ``run`` only;
  what the runtime dispatches.
- ``types.tools.Tool`` -- rich surface (``description``,
  ``directive_schema``, ``summary``, ``prompt``);
  what providers and the system prompt consume.
- :class:`_AgentTool` (in ``agent/agent.py``) -- the runtime-side
  wrapper. Pre-validates args against ``directive_schema``,
  consumes ``BackgroundAwareTool``-injected ``background`` / ``delay``,
  publishes ``ToolLabel``, post-processes (oversized persist, empty
  marker, AGENTS.md ``paths:`` rule reminder).

Providers
---------
Each provider class exposes a richer model surface
(``types.model.Model``) with ``buffer(request) -> ModelResponse`` /
``stream(request, on_text=, on_thinking=) -> ModelResponse``. The
``_AgentModel`` bridge wraps this into the runtime's lean
``stream(history, system, tools, on_text, on_thinking) ->
AssistantMessage`` signature. The bridge owns retry, persistent-mode
backoff for 429/529, context-overflow recovery (compact-and-retry),
and cost recording into ``agent.cost_tracker`` (subagents inherit
the parent's tracker so the root sees the full spawn-tree spend).

System prompt
-------------
``system`` may be a literal string, a no-arg ``Callable[[], str]``,
or a ``dict[str, str | Callable[[], str]]`` of named sections.
``_AgentModel.stream`` rebuilds the prompt per request so cwd-aware
sections stay live after ``cd``.

Errors
------
Tools never raise out of ``run`` -- they return
``ToolResult(is_error=True, content=...)`` so the model sees the
error and can self-correct. ``_AgentTool`` wraps unexpected raises in
``ToolResult(is_error=True)`` at the dispatch boundary. Infrastructure
errors (provider timeouts, rate limits, context overflow) propagate as
exceptions through the runtime's ``_stream_and_post`` which converts
them to ``ModelResponseError`` events.
"""

from sagent import providers, tools
from sagent.agent import Agent


__all__ = [
    "Agent",
    "providers",
    "tools",
]
