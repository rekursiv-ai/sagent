"""Tests for what a tool is, its argument schema and parsing, and per-agent state."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Final, Generic, Literal, cast
from typing_extensions import TypeVar
from unittest.mock import Mock

import asyncio
import os
import re
import threading

from treekle import PlainTree, immutable
from wesearch.fetch import FetchSession
from wesearch.fetch.session import default_session

import pytest

from sagent.session.types import (
    Attachment,
    Splice,
    ToolMessage,
    UncategorizedToolResult,
)
from sagent.tools.display import ToolDisplay
from sagent.tools.types import (
    Arg,
    FileTracker,
    Stream,
    ThreadWriter,
    Tool,
    ToolContext,
    ToolOutput,
    ToolStreamingResult,
    _ends,
    checked,
    json_schema,
    parse,
    settled,
    whole_output,
    write_fitting,
)


type _Color = Literal["red", "green"]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Arguments:
    text: Annotated[str, "What to say."]
    times: Annotated[int, "How often.", Arg(minimum=1, maximum=3)] = 1
    ratio: Annotated[float, "A share."] = 0.5
    loud: Annotated[bool, "Shout it."] = False
    color: Annotated[_Color, "A color."] = "red"
    level: Annotated[Literal[1, 2], "A level."] = 1
    limit: Annotated[int | None, "A cap.", Arg(minimum=1)] = None
    names: Annotated[tuple[str, ...], "Some names."] = ()
    either: Annotated[tuple[str, ...] | str, "One or many."] = ()
    headers: Annotated[Mapping[str, str] | None, "Headers."] = None
    body: Annotated[PlainTree, "Anything."] = None
    before: Annotated[int, "Lines before.", Arg(key="-B")] = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class _Whole[O, E]:
    """A tool that does not stream: it returns ``output``, or raises ``error``."""

    output: ToolOutput[UncategorizedToolResult, O, E]
    error: Exception | None = None

    name = "Whole"
    description = "Say what it was built with."
    schema = json_schema(_Arguments)
    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return "Whole"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[UncategorizedToolResult, O, E]:
        del context, arguments
        return whole_output(self._run())

    async def _run(self) -> ToolOutput[UncategorizedToolResult, O, E]:
        if self.error is not None:
            raise self.error
        return self.output


_CONTEXT: Final = ToolContext(cwd=Path("/"))


@pytest.mark.asyncio
async def test_a_text_tool_writes_text_and_returns_its_exit_code() -> None:
    """A bare ``Tool`` writes text on both streams and returns an exit code."""
    bash: Tool = _Whole[str, str](
        output=ToolOutput(
            stdout=("out\n",),
            stderr=("warn\n",),
            result=UncategorizedToolResult(exit_code=2),
        ),
    )

    streaming_result = bash.call(_CONTEXT, {})

    assert [line async for line in streaming_result.stdout] == ["out\n"]
    assert [line async for line in streaming_result.stderr] == ["warn\n"]
    assert await streaming_result.result == UncategorizedToolResult(exit_code=2)


@pytest.mark.asyncio
async def test_a_structured_tool_writes_typed_items_beside_its_text() -> None:
    """As Edit does: a ``Splice`` on stdout after the line the model reads."""
    edit = Splice(before="a", after="b", start=3, count=1)
    tool: Tool[UncategorizedToolResult, Splice | str, str] = _Whole[Splice | str, str](
        output=ToolOutput(
            stdout=("Edited a.py.", edit),
            result=UncategorizedToolResult(),
        ),
    )

    result = await settled(tool.call(_CONTEXT, {}), call_id="c1")

    assert result == ToolMessage(
        call_id="c1",
        state="done",
        stdout=("Edited a.py.", edit),
        result=UncategorizedToolResult(),
    )
    assert result.text() == "Edited a.py."


@pytest.mark.asyncio
async def test_tools_with_different_item_types_fit_one_any_tool_list() -> None:
    """A runtime keeps its tools in one list and settles each the same way."""
    image = Attachment(mime_descriptor="image/png", data=b"png")
    tools: list[Tool] = [
        _Whole[str, str](
            output=ToolOutput(stdout=("text",), result=UncategorizedToolResult()),
        ),
        _Whole[Splice | str, str](
            output=ToolOutput(
                stdout=(Splice(after="x"),),
                result=UncategorizedToolResult(),
            ),
        ),
        _Whole[Attachment | str, str](
            output=ToolOutput(
                stdout=("[image]", image),
                result=UncategorizedToolResult(),
            ),
        ),
    ]

    done = [await settled(tool.call(_CONTEXT, {})) for tool in tools]

    assert [result.text() for result in done] == ["text", "", "[image]"]
    assert [result.attachments for result in done] == [(), (), (image,)]


@pytest.mark.asyncio
async def test_a_call_starts_at_once_and_its_streams_need_no_reader() -> None:
    """The tool finishes with both streams unread; awaiting the result is enough."""
    ran: list[str] = []

    async def tool_output() -> ToolOutput[UncategorizedToolResult, str, str]:
        ran.append("ran")
        return ToolOutput(stdout=("out",), result=UncategorizedToolResult(exit_code=3))

    streaming_result = whole_output(tool_output())
    await asyncio.sleep(0)

    assert ran == ["ran"]
    assert await streaming_result.result == UncategorizedToolResult(exit_code=3)


def test_an_output_requires_a_result_and_defaults_to_no_output() -> None:
    with pytest.raises(TypeError, match="result"):
        _ = cast("Callable[..., object]", ToolOutput)(stdout=("out",))
    assert ToolOutput(result=UncategorizedToolResult()) == ToolOutput(
        stdout=(),
        stderr=(),
        result=UncategorizedToolResult(),
    )


@pytest.mark.asyncio
async def test_a_tool_that_raised_writes_nothing_and_its_result_raises() -> None:
    tool = _Whole[str, str](
        output=ToolOutput(stdout=("never",), result=UncategorizedToolResult()),
        error=ValueError("no"),
    )

    streaming_result = tool.call(_CONTEXT, {})

    assert [line async for line in streaming_result.stdout] == []
    assert [line async for line in streaming_result.stderr] == []
    with pytest.raises(ValueError, match="no"):
        _ = await streaming_result.result


@pytest.mark.asyncio
async def test_a_failed_call_s_record_ends_stderr_with_the_error() -> None:
    tool = _Whole[str, str](
        output=ToolOutput(result=UncategorizedToolResult()),
        error=ValueError("no"),
    )

    result = await settled(tool.call(_CONTEXT, {}), call_id="c1")

    assert result == ToolMessage(
        call_id="c1",
        state="failed",
        stderr=("ValueError: no",),
    )
    assert result.is_error
    assert result.text() == "ValueError: no"


@pytest.mark.asyncio
async def test_a_stream_gives_what_was_written_in_order_until_closed() -> None:
    stream = Stream[str]()
    stream.write("a")
    stream.write("b")
    stream.close()

    assert [item async for item in stream] == ["a", "b"]
    assert [item async for item in stream] == []


@pytest.mark.asyncio
async def test_a_stream_gives_an_item_written_while_it_is_read() -> None:
    stream = Stream[int]()
    reading = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)
    assert not reading.done()

    stream.write(7)

    assert await reading == 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cap", "written", "whole"),
    [
        (100, ["AA", "\nBBB", "\nC"], True),
        (7, ["AA", "\nBBB"], False),
        (1, ["A\n... (1 chars omitted)"], False),
    ],
    ids=["all-fit", "cut-between", "first-cut-midway"],
)
async def test_lines_are_written_while_they_fit_and_every_item_is_kept(
    cap: int,
    written: list[str],
    *,
    whole: bool,
) -> None:
    stream = Stream[str]()
    writer = ThreadWriter(stream, asyncio.get_running_loop())
    items = iter(["aa", "bbb", "c"])
    kept, all_written = await asyncio.to_thread(
        write_fitting,
        ToolContext(cwd=Path(), max_result_chars=cap),
        items,
        str.upper,
        writer,
    )
    stream.close()
    assert [unit async for unit in stream] == written
    assert (kept, all_written) == (["aa", "bbb", "c"], whole)


@pytest.mark.asyncio
async def test_a_thread_s_writes_have_landed_once_it_is_told_so() -> None:
    stream = Stream[str]()
    writer = ThreadWriter(stream, asyncio.get_running_loop())

    def work() -> None:
        writer.write("a")
        writer.write("b")
        writer.landed()

    thread = threading.Thread(target=work)
    thread.start()
    # The loop is held here, so it has written nothing and the thread still waits.
    thread.join(timeout=0.02)
    assert thread.is_alive()

    await asyncio.to_thread(thread.join)
    stream.close()

    assert [item async for item in stream] == ["a", "b"]


async def _lines(
    out: Stream[str],
    err: Stream[str],
    stop: asyncio.Event,
) -> UncategorizedToolResult:
    """Write a line on each stream, then run until ``stop`` or a cancel."""
    out.write("out\n")
    err.write("err\n")
    try:
        _ = await stop.wait()
    finally:
        out.close()
        err.close()
    return UncategorizedToolResult()


@pytest.mark.asyncio
async def test_a_cancelled_call_stops_its_tool_and_keeps_its_output() -> None:
    """A cancelled ``settled`` returns the record; it does not raise."""
    out, err, stop = Stream[str](), Stream[str](), asyncio.Event()
    result_future = asyncio.ensure_future(_lines(out, err, stop))
    waiting = asyncio.ensure_future(
        settled(
            ToolStreamingResult(result=result_future, stdout=out, stderr=err),
            call_id="c1",
        ),
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    _ = waiting.cancel()
    result = await waiting

    assert result_future.cancelled()
    assert result == ToolMessage(
        call_id="c1",
        state="cancelled",
        stdout=("out\n",),
        stderr=("err\n",),
    )


async def _broken() -> AsyncIterator[str]:
    """Yield a line, then raise as reading a lost pipe does."""
    yield "out\n"
    raise OSError("pipe lost")


@pytest.mark.asyncio
async def test_a_stream_that_breaks_fails_the_call_once_its_tool_has_stopped() -> None:
    err, stop = Stream[str](), asyncio.Event()
    result_future = asyncio.ensure_future(stop.wait())

    result = await settled(
        ToolStreamingResult(result=result_future, stdout=_broken(), stderr=err),
        call_id="c1",
    )

    assert result == ToolMessage(
        call_id="c1",
        state="failed",
        stdout=("out\n",),
        stderr=("OSError: pipe lost",),
    )
    assert result_future.cancelled()


@pytest.mark.asyncio
async def test_a_call_cancelled_twice_still_waits_for_its_tool_to_stop() -> None:
    """A second cancel must not return while the tool is still stopping."""
    out, err, stopped = Stream[str](), Stream[str](), asyncio.Event()

    async def slow_to_stop() -> int:
        try:
            _ = await asyncio.Event().wait()
        finally:
            _ = await stopped.wait()
        return 0

    result_future = asyncio.ensure_future(slow_to_stop())
    waiting = asyncio.ensure_future(
        settled(ToolStreamingResult(result=result_future, stdout=out, stderr=err)),
    )
    await asyncio.sleep(0)
    _ = waiting.cancel()
    await asyncio.sleep(0)
    _ = waiting.cancel()
    await asyncio.sleep(0)
    assert not waiting.done()

    stopped.set()

    assert (await waiting).state == "cancelled"
    assert result_future.cancelled()


@pytest.mark.asyncio
async def test_a_tool_that_relays_a_record_has_it_returned_for_this_call() -> None:
    """As RemoteTool: the call ran elsewhere, and the tool's result is its record."""
    far = ToolMessage(call_id="far", state="failed", stderr=("no such file",))
    out, err = Stream[object](), Stream[object]()
    out.close()
    err.close()
    relayed = asyncio.Future[ToolMessage]()
    relayed.set_result(far)

    result = await settled(
        ToolStreamingResult(result=relayed, stdout=out, stderr=err),
        call_id="here",
    )

    assert result == ToolMessage(
        call_id="here",
        state="failed",
        stderr=("no such file",),
    )


@pytest.mark.asyncio
async def test_a_live_call_is_recorded_once_its_streams_end() -> None:
    out, err, stop = Stream[str](), Stream[str](), asyncio.Event()
    result_future = asyncio.ensure_future(_lines(out, err, stop))
    waiting = asyncio.ensure_future(
        settled(ToolStreamingResult(result=result_future, stdout=out, stderr=err)),
    )
    await asyncio.sleep(0)
    assert not waiting.done()

    stop.set()

    assert await waiting == ToolMessage(
        call_id="",
        state="done",
        stdout=("out\n",),
        stderr=("err\n",),
        result=UncategorizedToolResult(),
    )


@pytest.mark.parametrize("fails", [False, True], ids=["returns", "raises"])
@pytest.mark.asyncio
async def test_a_tool_that_leaves_its_streams_open_is_recorded_when_it_ends(
    *,
    fails: bool,
) -> None:
    """As a process's pipes end when it exits: no reader waits for a tool that stopped."""
    out, err = Stream[str](), Stream[str]()

    async def tool() -> UncategorizedToolResult:
        out.write("out\n")
        err.write("err\n")
        if fails:
            raise ValueError("no")
        return UncategorizedToolResult()

    streaming_result = ToolStreamingResult(
        result=asyncio.ensure_future(tool()),
        stdout=out,
        stderr=err,
    )
    async with asyncio.timeout(0.2):
        result = await settled(streaming_result)

    assert result == (
        ToolMessage(
            call_id="",
            state="failed",
            stdout=("out\n",),
            stderr=("err\n", "ValueError: no"),
        )
        if fails
        else ToolMessage(
            call_id="",
            state="done",
            stdout=("out\n",),
            stderr=("err\n",),
            result=UncategorizedToolResult(),
        )
    )
    assert [item async for item in out] == []
    assert [item async for item in err] == []


@pytest.mark.asyncio
async def test_each_item_is_seen_as_it_is_read_while_the_tool_runs() -> None:
    out, err = Stream[str](), Stream[str]()
    stop, seen = asyncio.Event(), list[tuple[str, object]]()

    async def tool() -> UncategorizedToolResult:
        out.write("epoch 1\n")
        err.write("warn\n")
        _ = await stop.wait()
        out.write("done\n")
        return UncategorizedToolResult()

    waiting = asyncio.ensure_future(
        settled(
            ToolStreamingResult(
                result=asyncio.ensure_future(tool()),
                stdout=out,
                stderr=err,
            ),
            on_item=lambda stream, item: seen.append((stream, item)),
        ),
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert sorted(seen) == [("stderr", "warn\n"), ("stdout", "epoch 1\n")]
    assert not waiting.done()

    stop.set()

    assert await waiting == ToolMessage(
        call_id="",
        state="done",
        stdout=("epoch 1\n", "done\n"),
        stderr=("warn\n",),
        result=UncategorizedToolResult(),
    )
    assert seen[-1] == ("stdout", "done\n")


@pytest.mark.parametrize("shown", [None, "a\n... (1 lines omitted)\nc\n"])
@pytest.mark.asyncio
async def test_a_view_goes_to_the_record_beside_every_item(
    shown: str | None,
) -> None:
    out, err = Stream[str](), Stream[str]()
    for line in ("a\n", "b\n", "c\n"):
        out.write(line)
    done = asyncio.Future[UncategorizedToolResult]()
    done.set_result(UncategorizedToolResult())
    view = asyncio.Future[str | None]()
    view.set_result(shown)

    result = await settled(
        ToolStreamingResult(result=done, stdout=out, stderr=err, shown=view),
    )

    assert result == ToolMessage(
        call_id="",
        state="done",
        stdout=("a\n", "b\n", "c\n"),
        result=UncategorizedToolResult(),
        shown=shown,
    )
    assert result.text() == (shown if shown is not None else "a\nb\nc\n")


@pytest.mark.asyncio
async def test_whole_output_passes_its_view_to_the_record() -> None:
    output = ToolOutput(
        stdout=("a\n", "b\n"),
        result=UncategorizedToolResult(),
        shown="b\n",
    )
    result = await settled(whole_output(output))
    assert (result.stdout, result.shown) == (("a\n", "b\n"), "b\n")


@pytest.mark.asyncio
async def test_work_ends_of_itself_unless_it_is_said_to_run_until_stopped() -> None:
    output = ToolOutput(result=UncategorizedToolResult())

    assert [
        whole_output(output).runs_until_stopped,
        whole_output(output, runs_until_stopped=True).runs_until_stopped,
    ] == [False, True]


@pytest.mark.parametrize("fails", [False, True], ids=["returns", "raises"])
@pytest.mark.asyncio
async def test_work_writes_to_the_stdout_it_is_given_as_it_goes(*, fails: bool) -> None:
    stdout, stop = Stream[str](), asyncio.Event()

    async def work() -> ToolOutput[UncategorizedToolResult, str, str]:
        stdout.write("half\n")
        _ = await stop.wait()
        if fails:
            raise ValueError("no")
        return ToolOutput(stdout=("all\n",), result=UncategorizedToolResult())

    streaming_result = whole_output(work(), stdout=stdout)

    assert await asyncio.wait_for(anext(streaming_result.stdout), timeout=1) == "half\n"
    stop.set()
    async with asyncio.timeout(1):
        assert [item async for item in streaming_result.stdout] == (
            [] if fails else ["all\n"]
        )
    assert streaming_result.stdout is stdout


@pytest.mark.asyncio
async def test_work_writes_to_the_stderr_it_is_given_too() -> None:
    stderr = Stream[str]()
    stderr.write("half\n")
    output = ToolOutput(stderr=("all\n",), result=UncategorizedToolResult())

    streaming_result = whole_output(output, stderr=stderr)

    assert streaming_result.stderr is stderr
    assert (await settled(streaming_result)).stderr == ("half\n", "all\n")


@pytest.mark.asyncio
async def test_an_output_that_is_already_whole_is_written_as_it_is() -> None:
    """A tool with its answer at hand, such as a refusal, needs no coroutine."""
    output = ToolOutput(
        stdout=("out",),
        stderr=("err",),
        result=UncategorizedToolResult(exit_code=2),
    )

    assert await settled(whole_output(output), call_id="c1") == ToolMessage(
        call_id="c1",
        state="done",
        stdout=("out",),
        stderr=("err",),
        result=UncategorizedToolResult(exit_code=2),
    )


def test_schema_names_each_field_and_requires_those_without_defaults() -> None:
    schema = json_schema(_Arguments)
    assert schema["type"] == "object"
    assert schema["required"] == ["text"]
    assert schema["additionalProperties"] is False
    properties = schema["properties"]
    assert isinstance(properties, Mapping)
    assert properties["text"] == {"type": "string", "description": "What to say."}
    assert properties["times"] == {
        "type": "integer",
        "description": "How often.",
        "minimum": 1,
        "maximum": 3,
    }


def test_schema_spells_each_type_as_json_schema() -> None:
    properties = json_schema(_Arguments)["properties"]
    assert isinstance(properties, Mapping)
    got = {
        k: {**v, "description": ""}
        for k, v in properties.items()
        if isinstance(v, Mapping)
    }
    assert got["ratio"] == {"type": "number", "description": ""}
    assert got["loud"] == {"type": "boolean", "description": ""}
    assert got["color"] == {
        "type": "string",
        "enum": ["red", "green"],
        "description": "",
    }
    assert got["level"] == {"type": "integer", "enum": [1, 2], "description": ""}
    assert got["limit"] == {"type": "integer", "minimum": 1, "description": ""}
    assert got["names"] == {
        "type": "array",
        "items": {"type": "string"},
        "description": "",
    }
    assert got["either"] == {
        "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}],
        "description": "",
    }
    assert got["headers"] == {
        "type": "object",
        "additionalProperties": {"type": "string"},
        "description": "",
    }
    assert got["body"] == {"description": ""}


def test_schema_uses_the_models_name_for_a_renamed_argument() -> None:
    properties = json_schema(_Arguments)["properties"]
    assert isinstance(properties, Mapping)
    assert "-B" in properties
    assert "before" not in properties


@dataclass(frozen=True, slots=True, kw_only=True)
class _Counts:
    counts: Annotated[tuple[int, ...], "Counts.", Arg(minimum=1, maximum=9)] = ()
    maybe: Annotated[tuple[int, ...] | None, "Counts.", Arg(minimum=1)] = None


def test_bounds_on_a_sequence_bound_each_item() -> None:
    properties = json_schema(_Counts)["properties"]
    assert isinstance(properties, Mapping)
    assert properties["counts"] == {
        "type": "array",
        "items": {"type": "integer", "minimum": 1, "maximum": 9},
        "description": "Counts.",
    }
    assert properties["maybe"] == {
        "type": "array",
        "items": {"type": "integer", "minimum": 1},
        "description": "Counts.",
    }
    assert parse({"counts": [1, 9], "maybe": [3]}, _Counts) == _Counts(
        counts=(1, 9),
        maybe=(3,),
    )
    with pytest.raises(ValueError, match="'counts' must be at least 1, got 0"):
        parse({"counts": [1, 0]}, _Counts)
    with pytest.raises(ValueError, match="'counts' must be at most 9, got 10"):
        parse({"counts": [10]}, _Counts)
    with pytest.raises(ValueError, match="'maybe' must be at least 1, got 0"):
        parse({"maybe": [0]}, _Counts)


def test_schema_refuses_a_field_without_a_description() -> None:
    @dataclass(frozen=True, slots=True, kw_only=True)
    class Bare:
        x: int

    with pytest.raises(TypeError, match=r"Bare\.x needs Annotated"):
        json_schema(Bare)


def test_a_field_built_without_annotated_may_carry_its_description() -> None:
    @dataclass(frozen=True, slots=True, kw_only=True)
    class Built:
        x: int = field(metadata={"description": "An x."})

    assert json_schema(Built)["properties"] == {
        "x": {"type": "integer", "description": "An x."},
    }
    assert parse({"x": 2}, Built) == Built(x=2)


def test_schema_refuses_a_type_json_cannot_spell() -> None:
    @dataclass(frozen=True, slots=True, kw_only=True)
    class Odd:
        x: Annotated[bytes, "Raw."]

    with pytest.raises(TypeError, match="No JSON schema"):
        json_schema(Odd)


def test_schema_refuses_a_literal_member_json_cannot_hold() -> None:
    @dataclass(frozen=True, slots=True, kw_only=True)
    class Odd:
        x: Annotated[Literal["a", b"b"], "Raw."]

    with pytest.raises(TypeError, match="not a JSON value"):
        json_schema(Odd)


def test_parse_fills_defaults_for_what_was_omitted() -> None:
    assert parse({"text": "hi"}, _Arguments) == _Arguments(text="hi")


def test_a_field_with_a_default_factory_is_optional() -> None:
    @dataclass(frozen=True, slots=True, kw_only=True)
    class Made:
        tags: Annotated[tuple[str, ...], "Tags."] = field(default_factory=tuple)

    assert json_schema(Made)["required"] == []
    assert parse({}, Made) == Made()


def test_parse_maps_a_renamed_argument_to_its_field() -> None:
    assert parse({"text": "hi", "-B": 2}, _Arguments).before == 2


def test_parse_reads_sequences_as_tuples_and_keeps_mappings() -> None:
    a = parse(
        {"text": "hi", "names": ["a", "b"], "headers": {"k": "v"}, "body": [1]},
        _Arguments,
    )
    assert a.names == ("a", "b")
    assert a.headers == {"k": "v"}
    assert a.body == [1]


def test_parse_tries_each_union_member_in_turn() -> None:
    assert parse({"text": "x", "either": "one"}, _Arguments).either == "one"
    assert parse({"text": "x", "either": ["a"]}, _Arguments).either == ("a",)
    assert parse({"text": "x", "limit": None}, _Arguments).limit is None


def test_parse_accepts_each_member_of_a_literal() -> None:
    a = parse({"text": "x", "color": "green", "level": 2}, _Arguments)
    assert (a.color, a.level) == ("green", 2)


def test_parse_widens_an_integer_to_a_float() -> None:
    ratio = parse({"text": "x", "ratio": 2}, _Arguments).ratio
    assert ratio == 2.0
    assert type(ratio) is float


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "Missing argument 'text'"),
        (
            {"text": "x", "f": 1, "e": 1, "d": 1, "c": 1, "b": 1, "a": 1},
            "Unknown arguments: a, b, c, d, e, f.",
        ),
        ({"text": 1}, "'text' must be string, got 1"),
        ({"text": "x", "times": True}, "'times' must be integer"),
        ({"text": "x", "times": 1.5}, "'times' must be integer"),
        ({"text": "x", "loud": 1}, "'loud' must be boolean"),
        ({"text": "x", "ratio": True}, "'ratio' must be number"),
        ({"text": "x", "ratio": "1"}, "'ratio' must be number"),
        (
            {"text": "x", "color": "blue"},
            r"'color' must be one of \['red', 'green'\], got 'blue'",
        ),
        ({"text": "x", "level": True}, r"'level' must be one of \[1, 2\]"),
        ({"text": "x", "names": "ab"}, "'names' must be array"),
        ({"text": "x", "either": None}, "'either' must be array"),
        ({"text": "x", "headers": ["k"]}, r"'headers' must be object, got \['k'\]"),
        ({"text": "x", "headers": {"k": 1}}, "'headers' must be string"),
        ({"text": "x", "times": 0}, "'times' must be at least 1, got 0"),
        ({"text": "x", "times": 4}, "'times' must be at most 3, got 4"),
        ({"text": "x", "limit": 0}, "'limit' must be at least 1, got 0"),
    ],
)
def test_parse_rejects_arguments_that_do_not_fit(
    arguments: dict[str, PlainTree],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse(arguments, _Arguments)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Inner:
    name: str
    data: bytes = b""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Deeper(_Inner):
    count: Annotated[int, "A count."] = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class _Other:
    name: str


def _holding[R](record: R, **fields: object) -> R:
    """Return ``record`` with ``fields`` set past their types, as a decoded peer's may."""
    for name, value in fields.items():
        object.__setattr__(record, name, value)
    return record


@dataclass(frozen=True, slots=True, kw_only=True)
class _Boxed[T]:
    """A generic record, as ``ToolMessage`` is."""

    item: T
    items: tuple[T, ...] = ()
    name: str = ""


_Item_co = TypeVar("_Item_co", covariant=True, default=str)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Defaulted(Generic[_Item_co]):
    """A record whose type parameter has a default, as ``ToolOutput``'s have."""

    item: _Item_co


@pytest.mark.parametrize(
    ("hint", "value"),
    [
        (object, _Other(name="a")),
        (_Boxed[int], _Boxed(item=1, items=(2, 3))),
        (_Boxed, _Boxed(item=_Other(name="a"), items=("b", 1))),
        (_Defaulted, _Defaulted(item="a")),
        (
            ToolMessage[UncategorizedToolResult, Splice | str, str],
            ToolMessage(call_id="1", state="done", stdout=("a", Splice(after="b"))),
        ),
    ],
    ids=["object", "parameterized", "bare", "defaulted", "tool-result"],
)
def test_checked_takes_a_generic_record_whose_items_are_of_its_parameters(
    hint: object,
    value: object,
) -> None:
    assert checked(value, hint, key="x") is value


def test_checked_takes_anything_for_a_type_parameter_with_no_default() -> None:
    """On 3.12 a plain type parameter has no ``__default__``; this stands in for one."""
    parameter = Mock(spec=type(_Item_co))
    del parameter.__default__

    assert checked(3, parameter, key="x") == 3


@pytest.mark.parametrize(
    ("hint", "value", "where"),
    [
        (_Boxed[int], _Boxed(item="a"), "x.item"),
        (_Boxed[int], _Boxed(item=1, items=("a",)), "x.items"),
        (_Boxed[int], _holding(_Boxed(item=1), name=2), "x.name"),
        (_Defaulted, _Defaulted(item=1), "x.item"),
        (
            ToolMessage[UncategorizedToolResult, Splice | str, str],
            ToolMessage(call_id="1", state="done", stderr=(Splice(after="b"),)),
            "x.stderr",
        ),
    ],
    ids=["item", "items", "plain", "default", "structured"],
)
def test_checked_refuses_a_generic_record_holding_what_its_parameters_are_not(
    hint: object,
    value: object,
    where: str,
) -> None:
    with pytest.raises(ValueError, match=rf"^'{re.escape(where)}"):
        _ = checked(value, hint, key="x")


_INNER: Final = _Inner(name="a")


@pytest.mark.parametrize(
    ("hint", "value", "expected"),
    [
        (Annotated[int, "A count."], 1, 1),
        (bytes, b"\x00", b"\x00"),
        (datetime, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
        (Path, Path("/w"), Path("/w")),
        (tuple[int, str], [1, "a"], (1, "a")),
        (frozenset[str], {"a"}, frozenset({"a"})),
        (Mapping[str, int], {"a": 1}, {"a": 1}),
        (_Inner, _INNER, _INNER),
        (_Inner, _Deeper(name="a", count=2), _Deeper(name="a", count=2)),
        (_Inner | _Other | None, _INNER, _INNER),
        (type[_Inner | _Other] | None, _Other, _Other),
        (type[_Inner], _Inner, _Inner),
        (type[_Inner], _Deeper, _Deeper),
    ],
)
def test_checked_takes_what_wire_records_hold(
    hint: object,
    value: object,
    expected: object,
) -> None:
    assert checked(value, hint, key="x") == expected


@pytest.mark.parametrize(
    ("hint", "value", "message"),
    [
        (bytes, "a", "'x' must be bytes, got 'a'"),
        (datetime, "2026-01-01", "'x' must be datetime, got '2026-01-01'"),
        (Path, "/w", "'x' must be Path, got '/w'"),
        (tuple[str, ...], b"ab", "'x' must be array"),
        (tuple[int, str], (1,), "'x' must be 2 items, got"),
        (tuple[int, str], (1, 2), "'x' must be string, got 2"),
        (frozenset[str], ["a"], r"'x' must be set, got \['a'\]"),
        (frozenset[str], frozenset({1}), "'x' must be string, got 1"),
        (Mapping[str, int], {1: 1}, "'x' must be string, got 1"),
        (_Inner, _Other(name="a"), "'x' must be _Inner, got _Other"),
        # A wire message holds records; only a tool's arguments build one from JSON.
        (_Inner, {"name": "a"}, r"'x' must be _Inner, got \{'name': 'a'\}"),
        (_Inner, _holding(_Inner(name="a"), name=1), r"^'x\.name' must be string"),
        (_Inner, _holding(_Deeper(name="a"), count="2"), r"'x\.count' must be integer"),
        # Only the member a record is an instance of judges it.
        (_Inner | _Other, _holding(_Inner(name="a"), name=1), r"^'x\.name'[^_]*$"),
        (_Other | None, _INNER, "'x' must be a record of its type, got _Inner"),
        (int, "y" * 1000, r"^'x' must be integer, got .{0,60}$"),
        # A class named nowhere in ``type[...]``, or an instance of one, is neither.
        (
            type[_Inner | _Other],
            str,
            r"'x' must be one of \['_Inner', '_Other'\], got <class 'str'>",
        ),
        (type[_Inner], _INNER, r"'x' must be one of \['_Inner'\], got _Inner\("),
    ],
)
def test_checked_says_where_a_value_is_not_of_its_type(
    hint: object,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = checked(value, hint, key="x")


def test_checked_refuses_a_type_it_cannot_check() -> None:
    with pytest.raises(TypeError, match="No JSON schema"):
        _ = checked(1j, complex, key="x")


def test_parse_accepts_the_bounds_themselves() -> None:
    assert parse({"text": "x", "times": 1}, _Arguments).times == 1
    assert parse({"text": "x", "times": 3}, _Arguments).times == 3


@dataclass(frozen=True, slots=True, kw_only=True)
class _Step:
    step: Annotated[str, "What to do."]
    status: Annotated[Literal["pending", "done"], "Where it stands."] = "pending"


@dataclass(frozen=True, slots=True, kw_only=True)
class _Timed:
    before: Annotated[int, "Seconds before it.", Arg(minimum=0, key="-B")] = 0


type _Steps = tuple[_Step, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Plan:
    plan: Annotated[tuple[_Step, ...], "The steps."]
    lead: Annotated[_Step | None, "The step in hand."] = None
    timed: Annotated[tuple[_Timed, ...], "Waits."] = ()
    later: Annotated[_Steps, "Steps for later."] = ()
    notes: Annotated[Mapping[str, str], "Notes by step."] = field(
        default_factory=dict[str, str],
    )
    named: Annotated[Mapping[str, _Step], "Steps by name."] = field(
        default_factory=dict[str, _Step],
    )
    either: Annotated[_Timed | _Step | None, "A wait or a step."] = None
    maybe: Annotated[tuple[_Step, ...] | None, "Steps, if any."] = None


def test_schema_spells_a_nested_record_as_an_object_of_its_own() -> None:
    step = {
        "type": "object",
        "properties": {
            "step": {"type": "string", "description": "What to do."},
            "status": {
                "type": "string",
                "enum": ["pending", "done"],
                "description": "Where it stands.",
            },
        },
        "required": ["step"],
        "additionalProperties": False,
    }
    properties = json_schema(_Plan)["properties"]
    assert isinstance(properties, Mapping)
    assert properties["plan"] == {
        "type": "array",
        "items": step,
        "description": "The steps.",
    }
    assert properties["lead"] == {**step, "description": "The step in hand."}


_EMPTY: Final[dict[str, PlainTree]] = {}
"""A nested record's object with none of its fields."""


def test_parse_builds_each_nested_record_from_its_object() -> None:
    assert parse(
        {
            "plan": [{"step": "a"}, {"step": "b", "status": "done"}],
            "lead": {"step": "a"},
        },
        _Plan,
    ) == _Plan(
        plan=(_Step(step="a"), _Step(step="b", status="done")),
        lead=_Step(step="a"),
    )
    assert parse({"plan": [], "lead": None}, _Plan) == _Plan(plan=())
    # Through a type alias too; an object a record does not stand for stays as it came.
    assert parse(
        {"plan": [], "later": [{"step": "c"}], "notes": {"a": "b"}},
        _Plan,
    ) == _Plan(plan=(), later=(_Step(step="c"),), notes={"a": "b"})
    # A nested record's fields keep their own names and bounds.
    assert parse({"plan": [], "timed": [{"-B": 2}]}, _Plan).timed == (_Timed(before=2),)
    # As the schema offers them: an object's values, and any record of a union.
    assert parse({"plan": [], "named": {"x": {"step": "a"}}}, _Plan).named == {
        "x": _Step(step="a"),
    }
    assert parse({"plan": [], "either": {"step": "a"}}, _Plan).either == _Step(step="a")
    assert parse({"plan": [], "either": {"-B": 1}}, _Plan).either == _Timed(before=1)
    assert parse({"plan": [], "maybe": [{"step": "a"}]}, _Plan).maybe == (
        _Step(step="a"),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Free:
    tree: Annotated[PlainTree, "Any JSON."]


@dataclass(frozen=True, slots=True, kw_only=True)
class _StepOrNotes:
    either: Annotated[_Step | Mapping[str, str], "A step, or notes."]


def test_parse_takes_an_object_a_union_s_other_member_holds_when_its_record_refuses() -> (
    None
):
    assert parse({"either": {"a": "b"}}, _StepOrNotes).either == {"a": "b"}
    assert parse({"either": {"step": "a"}}, _StepOrNotes).either == _Step(step="a")
    # Held by neither: the record says why.
    with pytest.raises(ValueError, match=r"^In 'either': Unknown arguments: a\.$"):
        parse({"either": {"a": 1}}, _StepOrNotes)


def _nested(depth: int) -> PlainTree:
    """Return ``depth`` arrays, each holding the next."""
    return (_nested(depth - 1),) if depth else ()


def test_parse_hands_on_a_plain_tree_as_it_came() -> None:
    tree = immutable({"a": [{"b": 1}]})
    assert parse({"tree": tree}, _Free).tree is tree
    # A tree can hold no record, so it is never walked: depth costs nothing.
    deep = _nested(400)
    assert parse({"tree": deep}, _Free).tree is deep


@dataclass(frozen=True, slots=True, kw_only=True)
class _Ordered:
    notes: Annotated[Mapping[str, str] | _Step, "Notes, or a step."] = field(
        default_factory=dict[str, str],
    )
    steps: Annotated[tuple[str, ...] | tuple[_Step, ...], "Names, or steps."] = ()


def test_a_union_member_that_builds_nothing_gives_way_to_one_that_builds() -> None:
    parsed = parse({"notes": {"step": "a"}, "steps": [{"step": "b"}]}, _Ordered)
    assert parsed == _Ordered(notes=_Step(step="a"), steps=(_Step(step="b"),))


def test_parse_builds_nested_records_from_arguments_as_the_runtime_freezes_them() -> (
    None
):
    """A model's arguments reach a tool frozen: objects as mapping proxies, arrays as tuples."""
    assert parse(
        immutable({"plan": [{"step": "a"}], "lead": {"step": "b", "status": "done"}}),
        _Plan,
    ) == _Plan(plan=(_Step(step="a"),), lead=_Step(step="b", status="done"))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"plan": [_EMPTY]}, r"^In 'plan\[0\]': Missing argument 'step'\.$"),
        ({"plan": [{"step": "a"}, {"step": 1}]}, r"^In 'plan\[1\]': 'step' must be"),
        (
            {"plan": [{"step": "a", "why": "b"}]},
            r"^In 'plan\[0\]': Unknown arguments: why\.$",
        ),
        ({"plan": ["a"]}, r"^'plan' must be _Step, got 'a'\.$"),
        ({"plan": [], "lead": {"status": "done"}}, r"^In 'lead': Missing argument"),
        (
            {"plan": [], "later": [_EMPTY]},
            r"^In 'later\[0\]': Missing argument 'step'\.$",
        ),
        (
            {"plan": [], "timed": [{"-B": -1}]},
            r"^In 'timed\[0\]': '-B' must be at least 0, got -1\.$",
        ),
        (
            {"plan": [], "timed": [{"before": 1}]},
            r"^In 'timed\[0\]': Unknown arguments",
        ),
        ({"plan": [], "named": {"x": _EMPTY}}, r"^In 'named\[x\]': Missing argument"),
        # A union says why each of its records refused the object.
        (
            {"plan": [], "either": {"why": 1}},
            r"^In 'either': Unknown arguments: why\. In 'either': Unknown arguments: why\.$",
        ),
    ],
)
def test_parse_says_where_a_nested_record_does_not_fit(
    arguments: dict[str, PlainTree],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse(arguments, _Plan)


def test_a_file_never_seen_has_not_changed(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    files = FileTracker()
    assert f not in files
    assert not files.changed(f)


def test_a_seen_file_changes_when_its_bytes_do(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    files = FileTracker()
    files.saw(f)
    assert f in files
    assert not files.changed(f)
    f.write_text("y")
    _bump_mtime(f)
    assert files.changed(f)


def test_an_atomic_replacement_changes_even_with_the_same_mtime(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    before = f.stat()
    files = FileTracker()
    files.saw(f)
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("y")
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(f)

    assert f.stat().st_mtime_ns == before.st_mtime_ns
    assert f.stat().st_size == before.st_size
    assert files.changed(f)


def test_a_touch_that_keeps_every_byte_is_no_change(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("")
    files = FileTracker()
    files.saw(f)
    _bump_mtime(f)
    assert not files.changed(f)


def test_an_unchanged_stat_is_trusted_without_reading_the_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    files = FileTracker()
    files.saw(f)
    reading = Mock(side_effect=AssertionError("unchanged files must not be read"))
    monkeypatch.setattr(Path, "read_bytes", reading)

    assert not files.changed(f)
    reading.assert_not_called()


def test_a_removed_file_has_changed(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    files = FileTracker()
    files.saw(f)
    f.unlink()
    assert files.changed(f)


def test_lines_shown_hold_only_while_the_file_is_as_shown(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    files = FileTracker()
    assert not files.shown(f, lines=(1, None))
    files.saw(f)
    files.showed(f, lines=(1, None))
    assert files.shown(f, lines=(1, None))
    assert not files.shown(f, lines=(2, None))
    f.write_text("y")
    _bump_mtime(f)
    assert not files.shown(f, lines=(1, None))


def test_seeing_a_file_again_resets_what_was_shown(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    files = FileTracker()
    files.saw(f)
    files.showed(f, lines=(1, None))
    files.saw(f)
    assert not files.shown(f, lines=(1, None))


def test_a_file_is_one_file_however_its_path_is_spelled(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    real = tmp_path / "a.py"
    real.write_text("x")
    (tmp_path / "link.py").symlink_to(real)
    files = FileTracker()
    files.saw(tmp_path / "sub" / ".." / "a.py")
    assert real in files
    assert tmp_path / "link.py" in files
    assert "a.py" not in files
    files.showed(tmp_path / "link.py", lines=(1, None))
    assert files.shown(real, lines=(1, None))
    assert files.recent() == [real]
    real.write_text("y")
    _bump_mtime(real)
    assert files.changed(tmp_path / "sub" / ".." / "a.py")


def test_recent_orders_files_by_last_sight_and_forget_clears(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("a")
    b.write_text("b")
    files = FileTracker()
    files.saw(a)
    files.saw(b)
    files.saw(a)
    assert files.recent() == [b, a]
    files.forget()
    assert files.recent() == []
    assert a not in files


def test_forgetting_starts_a_window_no_other_tracker_shares() -> None:
    files, other = FileTracker(), FileTracker()
    assert files.window != other.window
    before = files.window
    files.forget()
    assert files.window not in (before, other.window)


def test_changes_yields_a_diff_and_sees_the_file_anew(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("alpha\n")
    files = FileTracker()
    files.saw(f)
    f.write_text("beta\n")
    _bump_mtime(f)
    ((path, diff),) = list(files.changes())
    assert path == f
    assert f"--- {f}" in diff
    assert "-alpha\n" in diff
    assert "+beta\n" in diff
    assert list(files.changes()) == []
    assert not files.changed(f)


def test_changes_skips_unchanged_and_removed_files(tmp_path: Path) -> None:
    kept, gone = tmp_path / "kept", tmp_path / "gone"
    kept.write_text("k")
    gone.write_text("g")
    files = FileTracker()
    files.saw(kept)
    files.saw(gone)
    gone.unlink()
    assert list(files.changes()) == []


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (b"\xff\xfe", b"now text\n"),
        (b"x" * ((1 << 20) + 1), b"now text\n"),
        (b"was text\n", b"\xff\xfe"),
    ],
    ids=["binary-before", "large-before", "binary-after"],
)
def test_changes_asks_for_a_reread_when_either_text_is_not_kept(
    tmp_path: Path,
    before: bytes,
    after: bytes,
) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(before)
    files = FileTracker()
    files.saw(f)
    f.write_bytes(after)
    _bump_mtime(f)
    assert list(files.changes()) == [(f, f"{f} changed on disk; read it again.")]


def test_a_file_at_the_size_cap_is_kept_to_diff(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_bytes(b"x" * (1 << 20))
    files = FileTracker()
    files.saw(f)
    f.write_text("y\n")
    _bump_mtime(f)
    ((_, diff),) = list(files.changes())
    assert "+y\n" in diff


def test_path_resolves_a_name_from_the_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    context = ToolContext(cwd=tmp_path / "work")
    assert context.path("a/../b.py") == tmp_path / "work" / "a" / ".." / "b.py"
    assert context.path("/etc/hosts") == Path("/etc/hosts")
    assert context.path("~/notes.md") == tmp_path / "home" / "notes.md"


def test_fit_keeps_whole_units_while_they_fit() -> None:
    context = ToolContext(cwd=Path("/"), max_result_chars=10)
    assert context.fit(["abcd", "efgh", "ijkl"]) == ("abcdefgh", 2)
    assert context.fit(["abcde", "fghij"]) == ("abcdefghij", 2)
    assert context.fit(["abcd", "efgh"], reserve=3) == ("abcd", 1)
    assert context.fit([]) == ("", 0)
    assert context.fit(["abcd", "efgh"], cap=6) == ("abcd", 1)
    assert context.fit(["abcd", "efgh"], cap=60) == ("abcdefgh", 2)


def test_fit_stops_before_a_later_unit_too_large_to_keep() -> None:
    context = ToolContext(cwd=Path("/"), max_result_chars=10)
    assert context.fit(["ab", "x" * 20]) == ("ab", 1)


def test_fit_cuts_a_first_unit_too_large_to_keep_whole() -> None:
    context = ToolContext(cwd=Path("/"), max_result_chars=10)
    assert context.fit(["x" * 1_015, "y"]) == (
        "x" * 10 + "\n... (1,005 chars omitted)",
        1,
    )


def test_fit_always_keeps_some_of_the_first_unit() -> None:
    context = ToolContext(cwd=Path("/"), max_result_chars=10)
    assert context.fit(["abc"], reserve=50) == ("a\n... (2 chars omitted)", 1)


# The line that counts the omitted lines takes 23 characters here, so a cap of 33
# leaves 10: 6 for the end, and what the end does not use for the start.
@pytest.mark.parametrize(
    ("lines", "cap", "expected"),
    [
        (
            ["a0\n", "a1\n", "a2\n", "a3\n", "a4\n"],
            33,
            "a0\n... (2 lines omitted)\na3\na4\n",
        ),
        (
            ["x" * 20 + "\n", "b\n", "c\n"],
            30,
            "x\n... (20 chars omitted)\n... (0 lines omitted)\nb\nc\n",
        ),
        (["a\n", "b\n", "ccccc\n"], 33, "a\nb\n... (0 lines omitted)\nccccc\n"),
        (["a\n", "b\n", "c\n"], 33, "a\n... (0 lines omitted)\nb\nc\n"),
        (["aaa\n", "bb\n", "cc\n"], 33, "aaa\n... (0 lines omitted)\nbb\ncc\n"),
        (
            ["aaaa\n", "bb\n", "cc\n"],
            33,
            "a\n... (4 chars omitted)\n... (0 lines omitted)\nbb\ncc\n",
        ),
        (
            ["a\n", "\n", "\n"],
            25,
            "a\n... (1 chars omitted)\n... (1 lines omitted)\n\n",
        ),
        (
            ["a\n", "b\n", "c" * 30 + "\n"],
            40,
            "a\nb\n... (1 lines omitted)\n",
        ),
    ],
    ids=[
        "the-end-gets-three-fifths",
        "the-end-is-all-but-the-first-line",
        "a-last-line-that-fills-the-end-is-kept",
        "no-line-is-in-both-ends",
        "a-first-line-that-fills-the-start-is-kept",
        "a-first-line-over-the-start-is-cut",
        "an-end-of-one-character-holds-one-line",
        "a-last-line-too-long-for-the-end-is-left-out",
    ],
)
def test_a_cut_output_keeps_its_start_and_its_end(
    lines: list[str],
    cap: int,
    expected: str,
) -> None:
    context = ToolContext(cwd=Path("/"), max_result_chars=50_000)
    assert _ends(context, lines, reserve=0, cap=cap, note="") == expected
    # With no cap of its own, the context's is the room.
    small = ToolContext(cwd=Path("/"), max_result_chars=cap)
    assert _ends(small, lines, reserve=0, cap=0, note="") == expected


def test_a_view_keeps_lines_that_fit_and_else_their_start_and_end() -> None:
    context = ToolContext(cwd=Path("/"), max_result_chars=60)
    assert context.view(["\n", "a\n", "b\n", "\n"]) == ("a\nb\n", False)
    assert context.view([]) == ("", False)
    lines = [f"line {i}\n" for i in range(20)]
    text, cut = context.view(lines, note="; see f")
    assert cut
    assert text == "line 0\n... (17 lines omitted; see f)\nline 18\nline 19\n"
    assert len(text) <= 60


def _bump_mtime(path: Path) -> None:
    """Move ``path``'s mtime well past what any earlier stat saw."""
    later = path.stat().st_mtime + 100
    os.utime(path, (later, later))


def test_the_origin_is_where_the_agent_started_else_where_it_works() -> None:
    assert ToolContext(cwd=Path("/w")).origin == Path("/w")
    assert ToolContext(cwd=Path("/w/sub"), root=Path("/w")).origin == Path("/w")


def test_a_context_given_nothing_works_here_and_fetches_through_the_shared_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    context = ToolContext()
    assert context.cwd == tmp_path
    assert context.fetch_session is default_session()
    given = FetchSession()
    assert ToolContext(fetch_session=given).fetch_session is given


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
