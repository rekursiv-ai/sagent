"""Mouse and keyboard inspection on an optional alternate terminal screen."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import base64
import platform

from prompt_toolkit.application import Application, in_terminal
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, is_searching
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.search import start_search
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, SearchToolbar, TextArea

from sagent.repl.tool_details import display_text
from sagent.tools.display import OutputSpec, format_preview


if TYPE_CHECKING:
    from collections.abc import Callable

    from prompt_toolkit.key_binding import KeyPressEvent

    from sagent.repl.tool_details import ToolDetails


class DetailsView:
    """A read-only view whose controls never dispatch to an agent."""

    def __init__(self, store: ToolDetails, number: int = 0) -> None:
        self.store = store
        entry = store.get(number)
        if entry is None:
            raise ValueError("No matching tool details")
        self.selected = entry.number - 1
        self.expanded = number != 0
        self.mouse = True
        self.opened_count = len(store.entries)
        self.notice = ""
        self.search = SearchToolbar()
        self.body = TextArea(
            read_only=True,
            scrollbar=True,
            wrap_lines=True,
            search_field=self.search,
        )
        self.list_control = FormattedTextControl(
            self._list_text,
            focusable=True,
            get_cursor_position=self._cursor_position,
        )
        kb = KeyBindings()
        kb.add("q", filter=~is_searching)(
            self._close,
        )
        kb.add("escape", filter=~is_searching)(self._close)
        kb.add("c-c", filter=~is_searching)(self._close)
        kb.add("f2")(self._toggle_mouse)
        kb.add("c-y")(self._copy)
        kb.add("c-f")(self._search)
        kb.add("tab")(self._focus)
        list_focused = Condition(lambda: self.app.layout.has_focus(self.list_control))
        kb.add("up", filter=list_focused)(self._previous)
        kb.add("down", filter=list_focused)(self._next)
        kb.add("enter", filter=list_focused)(self._toggle)
        kb.add(" ", filter=list_focused)(self._toggle)
        kb.add("c-o")(self._toggle)
        self.app: Application[None] = Application(
            layout=Layout(
                HSplit(
                    [
                        Window(FormattedTextControl(self._heading), height=1),
                        Frame(Window(self.list_control, height=7, wrap_lines=False)),
                        Frame(self.body, title="Command / diff"),
                        self.search,
                        Window(
                            FormattedTextControl(self._footer),
                            height=self._footer_height,
                            wrap_lines=True,
                        ),
                    ],
                ),
                focused_element=self.list_control,
            ),
            key_bindings=kb,
            full_screen=True,
            mouse_support=Condition(lambda: self.mouse),
            refresh_interval=0.25,
            style=Style.from_dict(
                {"selected": "bold ansicyan", "hint": "ansibrightblack"},
            ),
        )
        self._update_body()

    @property
    def _first(self) -> int:
        return max(0, self.selected - 3)

    def _cursor_position(self) -> Point:
        return Point(x=0, y=self.selected - self._first)

    def _heading(self) -> str:
        return f"Tool details · {len(self.store.entries)} items · read only"

    def _list_text(self) -> FormattedText:
        parts: list[tuple[str, str, Callable[[MouseEvent], None]]] = []
        for index in range(self._first, min(len(self.store.entries), self._first + 7)):
            entry = self.store.entries[index]
            arrow = "▾" if index == self.selected and self.expanded else "▸"
            style = "class:selected" if index == self.selected else ""

            def click(event: MouseEvent, index: int = index) -> None:
                if event.event_type == MouseEventType.MOUSE_UP:
                    if self.selected == index:
                        self.expanded = not self.expanded
                    else:
                        self.selected = index
                        self.expanded = True
                    self._update_body()
                    self.app.layout.focus(self.list_control)
                    self.app.invalidate()
                elif event.event_type == MouseEventType.SCROLL_UP:
                    self.select(self.selected - 1)
                elif event.event_type == MouseEventType.SCROLL_DOWN:
                    self.select(self.selected + 1)

            parts.append(
                (
                    style,
                    f"{arrow} #{entry.number} {entry.kind} · {display_text(entry.title)}\n",
                    click,
                ),
            )
        return FormattedText(parts)

    def _footer(self) -> str:
        new = len(self.store.entries) - self.opened_count
        activity = f"{new} new items" if new else ""
        return (
            "Click / Enter expand · Tab focus · Ctrl+F search · Ctrl+Y copy\n"
            f"Esc close · F2 mouse {'on' if self.mouse else 'off (native selection)'} · {activity} {self.notice}\n"
            f"Raw text: {self.store.entries[self.selected].path}"
        )

    def _footer_height(self) -> int:
        width = max(1, self.app.output.get_size().columns)
        return sum(
            max(1, (len(line) + width - 1) // width)
            for line in self._footer().splitlines()
        )

    def select(self, index: int) -> None:
        """Move within the local list without following newly arriving items."""
        self.selected = max(0, min(index, len(self.store.entries) - 1))
        self._update_body()
        self.app.invalidate()

    def _update_body(self) -> None:
        text = display_text(self.store.entries[self.selected].read())
        if not self.expanded:
            width = max(1, self.app.output.get_size().columns - 4)
            text = "\n".join(
                format_preview(
                    text,
                    OutputSpec(show=True, head_rows=8, tail_rows=4),
                    width=width,
                ),
            )
        self.body.buffer.set_document(
            Document(text, cursor_position=0),
            bypass_readonly=True,
        )
        self.body.window.vertical_scroll = 0

    def _previous(self, _event: KeyPressEvent) -> None:
        self.select(self.selected - 1)

    def _next(self, _event: KeyPressEvent) -> None:
        self.select(self.selected + 1)

    def _toggle(self, _event: KeyPressEvent) -> None:
        self.expanded = not self.expanded
        self._update_body()

    def _close(self, _event: KeyPressEvent) -> None:
        self.app.exit()

    def _toggle_mouse(self, _event: KeyPressEvent) -> None:
        self.mouse = not self.mouse

    def _focus(self, _event: KeyPressEvent) -> None:
        target = (
            self.body
            if self.app.layout.has_focus(self.list_control)
            else self.list_control
        )
        self.app.layout.focus(target)

    def _search(self, _event: KeyPressEvent) -> None:
        if not self.expanded:
            self.expanded = True
            self._update_body()
        self.app.layout.focus(self.body)
        start_search(self.body.control)

    def _copy(self, _event: KeyPressEvent) -> None:
        self.app.create_background_task(self.copy())

    async def copy(self) -> None:
        """Copy the selected text, or the complete original entry on request."""
        buffer = self.body.buffer
        text = (
            buffer.copy_selection().text
            if buffer.selection_state is not None
            else self.store.entries[self.selected].read()
        )
        if platform.system() == "Darwin":
            try:
                process = await asyncio.create_subprocess_exec(
                    "pbcopy",
                    stdin=asyncio.subprocess.PIPE,
                )
                await process.communicate(text.encode("utf-8"))
                copied = process.returncode == 0
            except OSError:
                copied = False
            self.notice = (
                "Copied" if copied else "Copy failed; use F2 for native selection"
            )
        else:
            encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
            self.app.output.write_raw(f"\x1b]52;c;{encoded}\x07")
            self.app.output.flush()
            self.notice = "Copy sent to terminal; use F2 if unsupported"
        self.app.invalidate()

    async def run(self) -> None:
        """Suspend only the prompt UI while the agent continues running."""
        async with in_terminal():
            await self.app.run_async(set_exception_handler=False)


async def show_details(store: ToolDetails, number: int = 0) -> None:
    """Open a viewer while preserving the caller's prompt application."""
    view = DetailsView(store, number)
    await view.run()
