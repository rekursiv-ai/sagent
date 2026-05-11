"""Agent: actor-model with one foreground slot, observer fan-out, mailbox.

Three primitives, one role each:

- ``self.inbox``: external work queue (``Inbox`` of source-tagged items)
- ``self.work``: the one foreground task (``asyncio.Task | None``)
- ``self.observers``: synchronous fan-out callables for ``Event`` payloads

See ``docs/private/agent_refactor.md`` and ``docs/private/execution_model.md``.
"""

from sagent.agent.agent import (
    ERROR_MAX_TOOL_CALL_ROUNDS,
    ActivityTracker,
    Agent,
    PendingOp,
    SystemPrompt,
)
from sagent.compactor import MICROCOMPACT_KEEP_RECENT
from sagent.custom_types import ContextBudget
from sagent.tools.background_task import BackgroundTaskEntry


__all__ = [
    "ERROR_MAX_TOOL_CALL_ROUNDS",
    "MICROCOMPACT_KEEP_RECENT",
    "ActivityTracker",
    "Agent",
    "BackgroundTaskEntry",
    "ContextBudget",
    "PendingOp",
    "SystemPrompt",
]
