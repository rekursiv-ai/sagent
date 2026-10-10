"""Lossless storage, bounded rendering, and read-only inspector controls."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import asyncio
import io
import stat

from prompt_toolkit import PromptSession
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Point
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput
from rich.console import Console

import pytest

from sagent.agent.agent import Agent
from sagent.repl.console_pane import ConsolePrinter
from sagent.repl.input_pane import _dispatch
from sagent.repl.input_queues import InputQueues
from sagent.repl.keybindings import NavState, build_key_bindings
from sagent.repl.render import (
    Printer as PrinterProtocol,
    render_tool_result,
)
from sagent.repl.slash import Details, Unknown, parse_slash
from sagent.repl.tool_details import ToolDetails, display_text
from sagent.repl.tool_details_view import DetailsView
from sagent.tools.display import OutputSpec, ToolDisplay, format_preview
from sagent.types.runtime import ToolLabel, ToolResult


if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def test_original_payload_permissions_and_cleanup() -> None:
    text = "हिन्दी 👩🏽‍💻\r\n\tsecond\n\n\x1b[31m"
    with ToolDetails() as store:
        detail = store.add("Bash", "command", text)
        assert detail.read() == text
        assert detail.path.read_bytes() == text.encode()
        assert stat.S_IMODE(detail.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(detail.path.parent.stat().st_mode) == 0o700
        directory = detail.path.parent
    assert not directory.exists()


def test_cleanup_after_failure() -> None:
    with ToolDetails() as store:
        path = store.add("file.py", "diff", "original").path
        with pytest.raises(RuntimeError):
            raise RuntimeError("viewer failed")
    assert not path.exists()


@pytest.mark.parametrize("width", [1, 10, 40, 80, 120])
def test_long_single_line_has_a_physical_row_budget(width: int) -> None:
    text = "python -c '" + "print(123);" * 400 + "'"
    rows = format_preview(
        text,
        OutputSpec(show=True, head_rows=3, tail_rows=1),
        width=width,
    )
    assert len(rows) == 5
    assert "rows omitted" in rows[3]
    assert rows[0].startswith("python"[:width])
    assert rows[-1].endswith("'")


@pytest.mark.parametrize("width", [40, 80, 120])
def test_command_preview_and_original_are_separate(width: int) -> None:
    command = "python -c '" + "print(123);" * 400 + "'\n"
    buffer = io.StringIO()
    with ToolDetails() as store:
        printer = ConsolePrinter(Console(file=buffer, width=width), details=store)
        printer.write_tool_label(
            "Bash\n" + command,
            command=OutputSpec(show=True, head_rows=3, tail_rows=1),
        )
        output = buffer.getvalue()
        assert "rows omitted" in output
        assert "Full command: /details 1" in output
        assert len(output.splitlines()) <= 9
        assert store.entries[0].read() == command


@pytest.mark.parametrize("width", [40, 80, 120])
def test_diff_preview_preserves_totals_tail_and_original(width: int) -> None:
    diff = (
        "@@ -1,80 +1,80 @@\n"
        + "\n".join(line for i in range(80) for line in (f"-before {i}", f"+after {i}"))
        + "\n"
    )
    buffer = io.StringIO()
    with ToolDetails() as store:
        printer = ConsolePrinter(Console(file=buffer, width=width), details=store)
        printer.write_diff(diff, "example.py")
        output = buffer.getvalue()
        assert "Added 80 lines, removed 80 lines" in output
        assert "148 rows omitted" in output
        assert "/details 1" in output
        assert "after 79" in output
        assert len(output.splitlines()) == 14
        assert store.entries[0].read() == diff


def test_short_diff_is_complete_without_an_omission_hint() -> None:
    buffer = io.StringIO()
    with ToolDetails() as store:
        printer = ConsolePrinter(Console(file=buffer, width=80), details=store)
        printer.write_diff("@@ -2 +2 @@\n-old\n+new\n", "test.py")
        assert "old" in buffer.getvalue()
        assert "new" in buffer.getvalue()
        assert "omitted" not in buffer.getvalue()


def test_short_diff_retains_word_highlighting_across_preview_regions() -> None:
    diff = (
        "@@ -1,6 +1,6 @@\n"
        + "".join(f"-value_{i} = old\n" for i in range(6))
        + "".join(f"+value_{i} = new\n" for i in range(6))
    )
    full = io.StringIO()
    preview = io.StringIO()
    ConsolePrinter(Console(file=full, width=80, force_terminal=True)).write_diff(
        diff,
        "test.py",
    )
    with ToolDetails() as store:
        ConsolePrinter(
            Console(file=preview, width=80, force_terminal=True),
            details=store,
        ).write_diff(diff, "test.py")
    assert full.getvalue() == preview.getvalue()


def test_worker_uses_shared_ids_and_its_available_width() -> None:
    command = "echo " + "界" * 300
    buffer = io.StringIO()
    with ToolDetails() as store:
        printer = ConsolePrinter(Console(file=buffer, width=40), details=store)
        printer.write_tool_label(
            "Bash\necho parent",
            command=OutputSpec(show=True, unbounded=True),
        )
        printer.write_child_block(
            "worker",
            [ToolLabel(call_id="child-1", text="Bash\n" + command)],
            output_policy=lambda _: ToolDisplay(
                command=OutputSpec(show=True, head_rows=3, tail_rows=1),
            ),
        )
        assert [entry.number for entry in store.entries] == [1, 2]
        assert store.entries[1].title == "worker: Bash"
        assert store.entries[1].read() == command
        assert "/details 2" in buffer.getvalue()
        assert "rows omitted" in buffer.getvalue()


def test_error_and_hint_do_not_get_hidden() -> None:
    buffer = io.StringIO()
    with ToolDetails() as store:
        printer = ConsolePrinter(Console(file=buffer, width=80), details=store)
        render_tool_result(
            printer,
            ToolResult(
                call_id="bad",
                content="important error",
                is_error=True,
                diff="should not render",
            ),
        )
        render_tool_result(
            printer,
            ToolResult(call_id="good", content="hidden body", hint="important hint"),
        )
        assert "important error" in buffer.getvalue()
        assert "important hint" in buffer.getvalue()
        assert "hidden body" not in buffer.getvalue()
        assert store.entries == []


def test_store_failure_shows_complete_command(monkeypatch: MonkeyPatch) -> None:
    command = "line\n" * 50
    buffer = io.StringIO()

    def fail(*_args: object) -> None:
        raise OSError("disk full")

    with ToolDetails() as store:
        monkeypatch.setattr(store, "add", fail)
        printer = ConsolePrinter(Console(file=buffer, width=80), details=store)
        printer.write_tool_label(
            "Bash\n" + command,
            command=OutputSpec(show=True, head_rows=3, tail_rows=1),
        )
        assert "disk full" in buffer.getvalue()
        assert buffer.getvalue().count("line\n") == 50


def test_display_controls_are_visible_but_original_is_unchanged() -> None:
    text = "abc\x1b[2J\r\b\x00def\nnext\tcolumn"
    assert "\x1b" not in display_text(text)
    assert r"\x1b" in display_text(text)
    assert "\nnext\tcolumn" in display_text(text)


@pytest.mark.parametrize("value", ["-1", "0", "foo", "1 2", "١", "9" * 5000])
def test_invalid_detail_ids_are_local_errors(value: str) -> None:
    assert isinstance(parse_slash("/details " + value), Unknown)


def test_detail_command_parses_without_becoming_user_text() -> None:
    assert parse_slash("/details") == Details()
    assert parse_slash("/details 12") == Details(number=12)


@pytest.mark.asyncio
async def test_details_dispatch_never_touches_agent() -> None:
    class Printer:
        def __init__(self) -> None:
            self.seen: list[int] = []

        async def inspect_details(self, number: int = 0) -> None:
            self.seen.append(number)

    printer = Printer()
    # A sentinel with no Agent methods fails if inspection reaches execution.
    assert not await _dispatch(
        cast(Agent, object()),
        Details(number=3),
        cast(PrinterProtocol, printer),
    )
    assert printer.seen == [3]


def test_mouse_expand_collapse_and_new_items_preserve_selection() -> None:
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
        ToolDetails() as store,
    ):
        text = "\n".join(f"line {i}" for i in range(80))
        store.add("test.py", "diff", text)
        view = DetailsView(store)
        assert "rows omitted" in view.body.text
        handler = cast(
            tuple[str, str, Callable[[MouseEvent], None]],
            view._list_text()[0],
        )[2]
        click = MouseEvent(
            Point(0, 0),
            MouseEventType.MOUSE_UP,
            MouseButton.LEFT,
            frozenset(),
        )
        handler(click)
        assert view.expanded
        assert view.body.text == text
        assert view.body.buffer.cursor_position == 0
        store.add("worker", "command", "echo new")
        assert view.selected == 0
        assert "1 new items" in view._footer()
        handler(click)
        assert not view.expanded
        assert "rows omitted" in view.body.text


async def _until(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(2):
        while not predicate():  # noqa: ASYNC110 -- Observe actual UI state; the renderer exposes no readiness event.
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_real_application_keyboard_search_mouse_toggle_and_close() -> None:
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
        ToolDetails() as store,
    ):
        text = "\n".join(f"searchable line {i}" for i in range(80))
        store.add("test.py", "diff", text)
        view = DetailsView(store)
        task = asyncio.create_task(view.run())
        await _until(lambda: view.app.is_running)
        pipe.send_text("\r")
        await _until(lambda: view.expanded)
        assert view.body.text == text
        pipe.send_text("\x1bOQ")  # F2 in the terminal protocol.
        await _until(lambda: not view.mouse)
        pipe.send_text("\x06")  # Ctrl+F.
        await _until(lambda: view.app.layout.is_searching)
        pipe.send_text("line 50\r")
        await _until(lambda: not view.app.layout.is_searching)
        assert view.body.buffer.cursor_position > 0
        pipe.send_text("q")
        await asyncio.wait_for(task, 2)


@pytest.mark.asyncio
async def test_pending_output_and_console_restored_after_viewer_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    buffer = io.StringIO()
    with ToolDetails() as store:
        store.add("Bash", "command", "echo original")
        printer = ConsolePrinter(
            Console(file=buffer, force_terminal=True),
            details=store,
        )

        async def fail(_store: ToolDetails, _number: int) -> None:
            printer.write_tool_error("error while inspecting")
            printer.write_child_block(
                "worker",
                [ToolLabel(call_id="t", text="child finished")],
            )
            raise RuntimeError("viewer failed")

        monkeypatch.setattr("sagent.repl.console_pane.show_details", fail)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        with pytest.raises(RuntimeError, match="viewer failed"):
            await printer.inspect_details()
        assert printer.console.file is buffer
        assert not printer._inspecting
        assert "error while inspecting" in buffer.getvalue()
        assert "child finished" in buffer.getvalue()


@pytest.mark.asyncio
async def test_ctrl_o_preserves_prompt_cursor_and_queues(
    monkeypatch: MonkeyPatch,
) -> None:
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
        ToolDetails() as store,
    ):
        store.add("Bash", "command", "\n".join(f"row {i}" for i in range(80)))
        output = io.StringIO()
        printer = ConsolePrinter(
            Console(file=output, force_terminal=True),
            details=store,
        )
        queues = InputQueues()
        queues.stage_queue("urgent waiting")
        queues.stage_deferred("deferred waiting")
        view = DetailsView(store)

        async def show(_store: ToolDetails, _number: int) -> None:
            await view.run()

        monkeypatch.setattr("sagent.repl.console_pane.show_details", show)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        session: PromptSession[str] = PromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=build_key_bindings(
                cast(Agent, object()),
                queues,
                NavState(),
                inspect_details=printer.inspect_details,
            ),
        )
        task = asyncio.create_task(session.prompt_async())
        await _until(lambda: session.app.is_running or task.done())
        if task.done():
            _ = await task
        pipe.send_text("draft intact\x02\x02")
        await _until(lambda: session.default_buffer.cursor_position == 10)
        pipe.send_text("\x0f")
        await _until(lambda: view.app.is_running)
        printer.write_line("background finished")
        store.add("worker", "command", "echo arrived")
        assert "background finished" not in output.getvalue()
        pipe.send_text("\rq")
        await _until(lambda: not printer._inspecting)
        assert session.default_buffer.text == "draft intact"
        assert session.default_buffer.cursor_position == 10
        assert queues.queue is not None
        assert queues.queue.text == "urgent waiting"
        assert queues.deferred is not None
        assert queues.deferred.text == "deferred waiting"
        assert view.selected == 0
        assert "background finished" in output.getvalue()
        session.app.exit(result=session.default_buffer.text)
        assert await task == "draft intact"


@pytest.mark.asyncio
async def test_cancellation_flushes_output_and_restores_console(
    monkeypatch: MonkeyPatch,
) -> None:
    with ToolDetails() as store:
        store.add("Bash", "command", "echo original")
        output = io.StringIO()
        printer = ConsolePrinter(
            Console(file=output, force_terminal=True),
            details=store,
        )
        opened = asyncio.Event()

        async def wait(_store: ToolDetails, _number: int) -> None:
            printer.write_line("completed before cancellation")
            opened.set()
            await asyncio.Event().wait()

        monkeypatch.setattr("sagent.repl.console_pane.show_details", wait)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        task = asyncio.create_task(printer.inspect_details())
        await opened.wait()
        _ = task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert printer.console.file is output
        assert not printer._inspecting
        assert "completed before cancellation" in output.getvalue()


@pytest.mark.asyncio
async def test_copy_failure_is_a_local_notice(monkeypatch: MonkeyPatch) -> None:
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
        ToolDetails() as store,
    ):
        store.add("Bash", "command", "exact\r\noriginal")
        view = DetailsView(store)

        async def fail(*_args: object, **_kwargs: object) -> None:
            raise OSError("clipboard unavailable")

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("asyncio.create_subprocess_exec", fail)
        await view.copy()
        assert "Copy failed" in view.notice


@pytest.mark.asyncio
async def test_copy_uses_complete_original_while_collapsed(
    monkeypatch: MonkeyPatch,
) -> None:
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
        ToolDetails() as store,
    ):
        text = "\r\n".join(f"original {i}" for i in range(80))
        store.add("Bash", "command", text)
        view = DetailsView(store)
        seen: list[bytes] = []

        class Process:
            returncode: int = 0

            async def communicate(self, value: bytes) -> None:
                seen.append(value)

        async def start(*_args: object, **_kwargs: object) -> Process:
            return Process()

        monkeypatch.setattr("platform.system", lambda: "Darwin")
        monkeypatch.setattr("asyncio.create_subprocess_exec", start)
        assert "rows omitted" in view.body.text
        await view.copy()
        assert seen == [text.encode()]
        assert view.notice == "Copied"


@pytest.mark.asyncio
async def test_nonterminal_details_prints_complete_safe_text() -> None:
    with ToolDetails() as store:
        text = "line\n" * 80 + "\x1b[2J"
        store.add("Bash", "command", text)
        output = io.StringIO()
        printer = ConsolePrinter(Console(file=output, width=80), details=store)
        await printer.inspect_details(1)
        assert output.getvalue().count("line\n") == 80
        assert r"\x1b[2J" in output.getvalue()
        assert "\x1b" not in output.getvalue()
        assert store.entries[0].read() == text


@pytest.mark.parametrize(("head", "tail"), [(0, 4), (4, 0), (0, 0), (2, 2)])
def test_preview_budgets_do_not_duplicate_rows(head: int, tail: int) -> None:
    rows = format_preview(
        "one\ntwo\nthree",
        OutputSpec(show=True, head_rows=head, tail_rows=tail),
        width=80,
    )
    assert rows == (["one", "two", "three"] if head + tail else [])
