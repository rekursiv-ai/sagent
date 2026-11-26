"""Inspect recorded commands and diffs without changing the agent or its input."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.application import Application
from prompt_toolkit.document import Document
from prompt_toolkit.filters import is_searching
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.search import start_search
from prompt_toolkit.widgets import Label, SearchToolbar, TextArea

from sagent.journal.types import EditResult, Splice, ToolCall, ToolMessage, WriteResult
from sagent.tui.diff import splice_diff, write_diff


if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent

    from sagent.agent.types import Agent


def entries(agent: Agent) -> list[tuple[str, str]]:
    """Collect the focused agent's visible Bash inputs and completed file diffs.

    Args:
      agent: The agent shown in the prompt.

    Returns:
      items: Labels and original payloads in conversation order.

    """
    items: list[tuple[str, str]] = []
    for record in agent.session:
        if isinstance(record, ToolCall) and record.name == "Bash":
            tool = agent.tools.get("Bash")
            command = record.arguments.get("command")
            if (
                tool
                and tool.display.command.lines != (0, 0)
                and isinstance(command, str)
            ):
                items.append(("Bash", command))
        elif isinstance(record, ToolMessage) and not record.is_error:
            result = record.result
            if isinstance(result, EditResult):
                tool = agent.tools.get("Edit")
                edits = [edit for edit in record.stdout if isinstance(edit, Splice)]
                if tool and tool.display.output.lines != (0, 0) and edits:
                    items.append((f"Edit {result.path or ''}", splice_diff(edits)))
            elif (
                isinstance(result, WriteResult)
                and result.content is not None
                and result.content != ""
            ):
                tool = agent.tools.get("Write")
                if tool and tool.display.output.lines != (0, 0):
                    items.append(
                        (f"Write {result.path or ''}", write_diff(result.content))
                    )
    return items


async def inspect(items: list[tuple[str, str]], number: int) -> None:
    """Open the selected entry in a read-only, searchable terminal view.

    Args:
      items: Labels and payloads from the current conversation.
      number: One-based index of the first item to show.

    """
    index = number - 1
    search = SearchToolbar()
    body = TextArea(read_only=True, scrollbar=True, search_field=search)
    keys = KeyBindings()

    @keys.add("c-p", filter=~is_searching)
    def previous(event: KeyPressEvent) -> None:
        del event
        nonlocal index
        index = _display(body, items, index - 1)

    @keys.add("c-n", filter=~is_searching)
    def following(event: KeyPressEvent) -> None:
        del event
        nonlocal index
        index = _display(body, items, index + 1)

    @keys.add("c-f", filter=~is_searching)
    def find(event: KeyPressEvent) -> None:
        del event
        start_search(body.control)

    @keys.add("q", filter=~is_searching)
    @keys.add("escape", filter=~is_searching)
    @keys.add("c-c")
    def close(event: KeyPressEvent) -> None:
        event.app.exit()

    _display(body, items, index)
    layout = HSplit(
        [
            Label(lambda: f"{index + 1}/{len(items)}  {items[index][0]}"),
            body,
            search,
            Label("Ctrl+P/N previous/next   Ctrl+F search   Esc close"),
        ]
    )
    app: Application[None] = Application(
        layout=Layout(layout, focused_element=body), key_bindings=keys, full_screen=True
    )
    await app.run_async()


def _display(body: TextArea, items: list[tuple[str, str]], position: int) -> int:
    """Replace the read-only document and return the cursor to its start."""
    index = max(0, min(position, len(items) - 1))
    body.buffer.set_document(
        Document(items[index][1], cursor_position=0), bypass_readonly=True
    )
    return index
