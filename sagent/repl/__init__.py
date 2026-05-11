"""REPL surface for ``Agent``.

The REPL attaches one render observer to ``agent.observers`` and runs an
input pump as a hidden background task. ``run_repl`` is the entry point;
the rest of this package provides the pieces it composes.

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
    HELP_TEXT,
    Printer,
    RecordingPrinter,
    RenderObserver,
    make_render_observer,
    render_tool_result,
)
from sagent.repl.replay import replay_messages
from sagent.repl.run_repl import run_repl


__all__ = [
    "HELP_TEXT",
    "ConsolePrinter",
    "InputSource",
    "Printer",
    "PromptToolkitInputSource",
    "RecordingPrinter",
    "RenderObserver",
    "StubInputSource",
    "make_render_observer",
    "render_tool_result",
    "replay_messages",
    "run_repl",
    "spawn_repl_pump",
]
