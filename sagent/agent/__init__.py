"""Agent: deque + handler registry + dispatch loop.

The agent IS its deque. Every signal -- user input, model response,
tool batch, abort, quit -- is a message routed through the dispatch
loop to descriptor-keyed handlers. See ``agent.py`` for the runtime
and ``handlers/`` for the standard handler set.

Session persistence
-------------------
``Agent.save_session`` serializes ``history`` and metadata to
``session.jsonl`` after every model response (via
:class:`SessionSaveHandler`). Manual triggers (``set_status``,
``ClearHandler``) save immediately so the on-disk state never
trails the in-memory state by more than one model call.
"""

from sagent.agent.agent import (
    ERROR_MAX_TOOL_CALL_ROUNDS,
    Agent,
    RunHandle,
    SystemPrompt,
)
from sagent.compactor import MICROCOMPACT_KEEP_RECENT
from sagent.custom_types import ContextBudget
from sagent.tools.background_task import BackgroundTaskEntry


__all__ = [
    "ERROR_MAX_TOOL_CALL_ROUNDS",
    "MICROCOMPACT_KEEP_RECENT",
    "Agent",
    "BackgroundTaskEntry",
    "ContextBudget",
    "RunHandle",
    "SystemPrompt",
]
