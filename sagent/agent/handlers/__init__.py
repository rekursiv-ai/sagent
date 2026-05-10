"""Handlers for ``Agent``: descriptor-routed message handlers.

The agent dispatch loop pops messages from a deque and routes them to
handlers by descriptor. Each handler subclasses :class:`InlineHandler`
(runs on the loop) or :class:`SpawnedHandler` (runs as a task).

Composition is constructor-injected: the caller passes the full
handler list to ``Agent``. :func:`core_handlers` returns the standard
set; CLI orchestrators add surface (REPL, Slack) and app-specific
handlers.

Layout:
  - :mod:`handlers.base` -- ``Handler`` protocol + ``InlineHandler`` /
    ``SpawnedHandler`` base classes.
  - :mod:`handlers.core` -- the small single-purpose inline handlers
    (User, History, Stats, Clear, Abort, Session) plus the
    :func:`core_handlers` factory.
  - :mod:`handlers.activity` -- ``ActivityTracker`` + ``ActivityHandler``.
  - :mod:`handlers.compact` -- ``BudgetWatcher`` + ``CompactHandler`` +
    ``RecompactHandler`` + shared ``run_compaction`` helper.
  - :mod:`handlers.model` -- ``ModelCallHandler`` + ``ModelResponseHandler``.
  - :mod:`handlers.tools` -- ``ToolBatchHandler`` + ``ToolBatchResultHandler``.
  - :mod:`handlers.model_switch` -- ``/model`` slash-command handler.
"""

from sagent.agent.handlers.activity import (
    ActivityHandler,
    ActivityTracker,
)
from sagent.agent.handlers.base import (
    Handler,
    InlineHandler,
    SpawnedHandler,
)
from sagent.agent.handlers.compact import (
    BudgetWatcher,
    CompactHandler,
    RecompactHandler,
)
from sagent.agent.handlers.core import (
    AbortHandler,
    BreakHandler,
    ClearHandler,
    HistoryHandler,
    SessionSaveHandler,
    StatsHandler,
    UserMessageHandler,
    core_handlers,
)
from sagent.agent.handlers.model import (
    ModelCallHandler,
    ModelResponseHandler,
)
from sagent.agent.handlers.model_switch import ModelSwitchHandler
from sagent.agent.handlers.tools import (
    ToolBatchHandler,
    ToolBatchResultHandler,
)


__all__ = [
    "AbortHandler",
    "ActivityHandler",
    "ActivityTracker",
    "BreakHandler",
    "BudgetWatcher",
    "ClearHandler",
    "CompactHandler",
    "Handler",
    "HistoryHandler",
    "InlineHandler",
    "ModelCallHandler",
    "ModelResponseHandler",
    "ModelSwitchHandler",
    "RecompactHandler",
    "SessionSaveHandler",
    "SpawnedHandler",
    "StatsHandler",
    "ToolBatchHandler",
    "ToolBatchResultHandler",
    "UserMessageHandler",
    "core_handlers",
]
