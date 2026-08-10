"""Agent: actor-model with one foreground slot, observer fan-out, mailbox.

Three primitives, one role each:

- ``self.runtime.inbox`` -- external work queue. A ``GatedDeque`` of
  ``RuntimeEvent`` items (user/assistant/tool history, control verbs:
  ``Quit``, ``Halt``, ``Clear``, ``Compact``, ``Recompact``, ``Kill``,
  ``Detach``, ``ModelSwitch``). The single dispatch loop drains this
  deque in batches; one ``match`` statement routes each event.
- ``self.work`` -- the single foreground task (``asyncio.Task | None``).
  Holds the model call and the tool cohort currently in flight. Halt
  cancels it; Quit drains the deque and exits.
- ``self.observers`` -- synchronous fan-out callables for every
  published ``RuntimeEvent``. Used by the REPL renderer, cost tracker,
  activity tracker, persistence (``SaveSession``), and round-cap
  enforcement.

Composition (no inheritance):

- ``_AgentModel`` bridges the rich provider ``Model`` interface
  (``buffer`` / ``stream`` returning ``ModelResponse``) to the runtime's
  lean ``stream(history, system, tools, on_text, on_thinking) ->
  AssistantMessage`` protocol. Owns the retry loop and overflow
  recovery; records cost out-of-band on ``Agent.cost_tracker``.
- ``_AgentTool`` bridges a rich ``Tool`` (metadata + ``summary`` /
  ``prompt``) to the runtime's minimal
  ``run(args) -> ToolResult`` protocol. Emits ``ToolLabel`` before
  execution; post-processes the result for empty-marker and
  oversized-content handling.
- ``_AgentCompactor`` bridges the rich ``SummaryCompactor`` interface
  to the runtime's lean compactor protocol and runs the post-compaction
  enrich pipeline (file reattach, status injection, tool restore).
"""

from sagent.agent.agent import (
    ERROR_MAX_TOOL_CALL_ROUNDS,
    ActivityTracker,
    Agent,
    SystemPromptArg,
)
from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.compaction import CompactionState
from sagent.types.model import ContextBudget


__all__ = [
    "ERROR_MAX_TOOL_CALL_ROUNDS",
    "ActivityTracker",
    "Agent",
    "BackgroundTaskEntry",
    "CompactionState",
    "ContextBudget",
    "SystemPromptArg",
]
