"""Agent: Model + Tools + Prompt + Compactor behind two queues.

See ``sagent/__init__.py`` for the Agent architecture, inbox zero
loop, error policy, and contract definitions.

Session persistence
-------------------
``_save_session`` serializes ``_messages`` and metadata to
``session.jsonl``. The request loop wraps in ``try/finally`` so
every exit path persists. Internal mutations (status change,
compaction, clear) save immediately.
"""

from sagent.agent.agent import (
    ERROR_MAX_TOOL_CALL_ROUNDS,
    ERROR_NO_PROMPT,
    QUIT_SENTINEL,
    Agent,
    RunHandle,
    SystemPrompt,
)
from sagent.compactor import MICROCOMPACT_KEEP_RECENT
from sagent.custom_types import ContextBudget
from sagent.tools.background_task import BackgroundTaskEntry


__all__ = [
    "ERROR_MAX_TOOL_CALL_ROUNDS",
    "ERROR_NO_PROMPT",
    "MICROCOMPACT_KEEP_RECENT",
    "QUIT_SENTINEL",
    "Agent",
    "BackgroundTaskEntry",
    "ContextBudget",
    "RunHandle",
    "SystemPrompt",
]
