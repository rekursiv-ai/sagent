"""Cross-cutting type definitions for sagent.

This package is the root of the project's type system: every other
module depends *on* these types, and these types depend *only* on the
standard library plus ``sagent.lib`` leaves. Nothing in this tree imports
from ``agent``, ``providers``, ``tools``, or ``repl``.

This ``__init__`` does **not** flatten symbols into the package
namespace -- the type system is large enough that a flat namespace
hides where things live. Callers must reach into a specific submodule:

  - ``sagent.types.history`` -- conversation history
    records (``UserMessage``, ``AssistantMessage``, ``ToolResult``,
    ``ToolCall``, ``BytesMessage``, ``HistoryEntry``,
    ``SessionMessage``, ``reset_id_counter``).
  - ``sagent.types.runtime`` -- dispatch event
    vocabulary consumed by ``agent/runtime.py``'s engine.
  - ``sagent.types.model`` -- ``Model`` Protocol plus
    ``ModelRequest``, ``ModelResponse``, ``ModelSpec``,
    ``ContextBudget``, ``Pricing``, ``TokenCount``.
  - ``sagent.types.providers`` -- ``Provider``
    factory Protocol plus ``AuthReloadable``.
  - ``sagent.types.tools`` -- ``Tool`` Protocol.
  - ``sagent.types.compactor`` -- ``Compactor`` and
    ``CompactRestorable`` Protocols.
  - ``sagent.types.exceptions`` -- ``UserFacingError``
    hierarchy plus ``log_exception_or_warning``.
"""

from sagent.types import (
    compactor,
    exceptions,
    history,
    model,
    providers,
    runtime,
    tools,
)


__all__ = [
    "compactor",
    "exceptions",
    "history",
    "model",
    "providers",
    "runtime",
    "tools",
]
