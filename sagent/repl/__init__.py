"""REPL surface for ``Agent``.

The REPL attaches one render observer to ``agent.observers`` and runs
an input pump as a hidden background task. ``run_repl`` is the entry
point; the rest of this package provides the pieces it composes.

CLI orchestrators build the agent like this::

    agent = Agent(model=..., tools=[...], compactor=...)
    asyncio.run(run_repl(agent))

The visible REPL has four zones, top to bottom:

- ``console_pane`` -- scrollback (rich-rendered output):
  :mod:`repl.console_pane`.
- ``queued_input_pane`` -- optional dim preview line of texts waiting
  to be sent (rendered by :func:`repl.input_pane.render_input_pane`
  when busy).
- ``input_pane`` -- the ``> `` prompt where the user types:
  :mod:`repl.input_pane`.
- ``status_pane`` -- bracketed totals + spinner line below the prompt:
  :mod:`repl.status_pane`.

Other supporting modules: :mod:`repl.run_repl` (orchestrator),
:mod:`repl.keybindings`, :mod:`repl.render`, :mod:`repl.replay`,
:mod:`repl.format`, :mod:`repl.render_diff`, :mod:`repl.tight_markdown`.
"""

from sagent.repl.console_pane import ConsolePrinter
from sagent.repl.input_pane import (
    InputSource,
    PromptToolkitInputSource,
    StubInputSource,
    spawn_repl_pump,
)
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
