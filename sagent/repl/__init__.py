"""REPL handler bundle for ``Agent``.

The REPL is a *handler bundle* registered on the same agent that
runs the dispatch loop. Render handlers consume descriptors the
model + tool dispatch produce; the input pump runs as a hidden
background task on ``agent.background_tasks`` and posts user
messages back into the inbox.

CLI orchestrators build the agent like this::

    agent = Agent(model=..., tools=[...], compactor=...)
    asyncio.run(run_repl(agent))

The pieces live in :mod:`repl.run_repl` (orchestrator), :mod:`repl.console`,
:mod:`repl.input`, :mod:`repl.keybindings`, :mod:`repl.prompt`,
:mod:`repl.render`, :mod:`repl.replay`, :mod:`repl.toolbar`,
:mod:`repl.format`, :mod:`repl.render_diff`, :mod:`repl.tight_markdown`.
"""

from sagent.repl.console import ConsolePrinter
from sagent.repl.input import (
    InputSource,
    StubInputSource,
    spawn_repl_pump,
)
from sagent.repl.prompt import PromptToolkitInputSource
from sagent.repl.render import (
    Printer,
    RecordingPrinter,
    RenderChildEvent,
    RenderError,
    RenderInterrupted,
    RenderStream,
    RenderToolLabel,
    RenderToolResult,
    RenderUserBar,
    repl_handler_set,
)
from sagent.repl.replay import replay_messages
from sagent.repl.run_repl import run_repl


__all__ = [
    "ConsolePrinter",
    "InputSource",
    "Printer",
    "PromptToolkitInputSource",
    "RecordingPrinter",
    "RenderChildEvent",
    "RenderError",
    "RenderInterrupted",
    "RenderStream",
    "RenderToolLabel",
    "RenderToolResult",
    "RenderUserBar",
    "StubInputSource",
    "repl_handler_set",
    "replay_messages",
    "run_repl",
    "spawn_repl_pump",
]
