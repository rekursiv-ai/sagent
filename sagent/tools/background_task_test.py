"""Tests for background-job control keys and the BackgroundTask tool."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import re
import time

from treekle import PlainTree, from_plain

import pytest

from sagent.session.types import AnyToolResult, ToolCall, UncategorizedToolResult
from sagent.tools.background_task import (
    BackgroundTask,
    Offered,
    background_delay,
    backgroundable,
    offered,
    without_controls,
)
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from sagent.runtime.types import Job
    from sagent.tools.types import Tool


@dataclass(slots=True, kw_only=True)
class Echo:
    """A tool that says its arguments back."""

    name: str = "Echo"
    description: str = "Say it back."
    schema: Mapping[str, PlainTree] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    display: ToolDisplay = field(
        default_factory=lambda: ToolDisplay(output=DisplaySpec()),
    )

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        return f"Echo {sorted(arguments)}"

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        return f"{context.cwd}:" + ",".join(sorted(arguments))

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> AnyToolResult:
        return UncategorizedToolResult(
            call_id="",
            content=f"{context.cwd}: {sorted(arguments)}",
        )


@dataclass(slots=True, kw_only=True)
class FakeJob:
    """A job as the runtime lists it."""

    call: ToolCall
    started: float
    state: Literal["running", "sleeping", "detached"]
    cancelled: list[bool] = field(default_factory=list)

    def cancel(self) -> None:
        self.cancelled.append(True)


@dataclass(slots=True, kw_only=True)
class FakeTable:
    """The runtime's jobs, and the result a foreground call would get."""

    listed: dict[str, FakeJob] = field(default_factory=dict)
    results: dict[str, AnyToolResult] = field(default_factory=dict)
    waited: list[str] = field(default_factory=list)

    @property
    def jobs(self) -> Mapping[str, Job]:
        return self.listed

    async def foreground(self, call_id: str) -> AnyToolResult | None:
        self.waited.append(call_id)
        return self.results.get(call_id)


def _job(
    name: str,
    state: Literal["running", "sleeping", "detached"],
    age: float,
) -> FakeJob:
    return FakeJob(
        call=ToolCall(call_id="", name=name),
        started=time.time() - age,
        state=state,
    )


_CONTEXT = ToolContext(cwd=Path("/"))


# --- Control keys ---------------------------------------------------------------


def test_an_ordinary_tool_is_backgroundable() -> None:
    assert backgroundable(Echo())


@pytest.mark.parametrize("key", ["background", "delay"])
def test_a_tool_owning_a_control_key_is_not(key: str) -> None:
    assert not backgroundable(Echo(schema={"type": "object", "properties": {key: {}}}))


def test_background_task_is_not_backgroundable() -> None:
    assert not backgroundable(BackgroundTask())
    assert offered(BackgroundTask()).schema is BackgroundTask.schema


def test_a_schemaless_tool_is_backgroundable() -> None:
    tool = Echo(schema={})
    assert backgroundable(tool)
    assert set(_properties(offered(tool))) == {"background", "delay"}


def test_offered_adds_the_control_keys_and_keeps_the_rest() -> None:
    tool = Echo()
    shown = offered(tool)
    assert isinstance(shown, Offered)
    assert shown.schema["required"] == ["text"]
    assert shown.schema["type"] == "object"
    properties = _properties(shown)
    assert properties["text"] == {"type": "string"}
    assert properties["background"] == {
        "type": "boolean",
        "description": "Run in the background; the result arrives in a later message.",
    }
    assert properties["delay"] == {
        "type": "number",
        "minimum": 0,
        "description": "Seconds to wait before running; implies background.",
    }
    assert _properties(tool) == {"text": {"type": "string"}}


def test_a_tool_that_owns_a_key_is_offered_as_itself() -> None:
    tool = Echo(schema={"type": "object", "properties": {"delay": {}}})
    assert offered(tool) is tool


@pytest.mark.asyncio
async def test_offered_forwards_everything_without_the_control_keys() -> None:
    tool = Echo()
    shown = offered(tool)
    arguments: dict[str, PlainTree] = {"text": "hi", "background": True, "delay": 2}
    assert (shown.name, shown.description, shown.display) == (
        tool.name,
        tool.description,
        tool.display,
    )
    assert shown.summary(arguments) == "Echo ['text']"
    assert shown.serialize_key(_CONTEXT, arguments) == "/:text"
    result = await shown.call(_CONTEXT, arguments)
    assert result.text() == "/: ['text']"


@pytest.mark.parametrize(
    ("arguments", "delay"),
    [
        ({}, None),
        ({"background": False}, None),
        ({"delay": 0}, None),
        ({"background": True}, 0.0),
        ({"background": True, "delay": 0}, 0.0),
        ({"delay": 2}, 2.0),
        ({"delay": 0.5, "background": False}, 0.5),
    ],
)
def test_background_delay(arguments: dict[str, PlainTree], delay: float | None) -> None:
    assert background_delay(arguments) == delay


@pytest.mark.parametrize(
    ("arguments", "kind", "error"),
    [
        (
            {"background": "yes"},
            TypeError,
            "'background' must be a boolean, got 'yes'.",
        ),
        ({"background": 1}, TypeError, "'background' must be a boolean, got 1."),
        ({"delay": True}, TypeError, "'delay' must be seconds, at least 0, got True."),
        ({"delay": "5"}, TypeError, "'delay' must be seconds, at least 0, got '5'."),
        ({"delay": -0.5}, ValueError, "'delay' must be seconds, at least 0, got -0.5."),
        (
            {"delay": float("nan")},
            ValueError,
            "'delay' must be seconds, at least 0, got nan.",
        ),
    ],
)
def test_a_malformed_control_key_is_refused(
    arguments: dict[str, PlainTree],
    kind: type[Exception],
    error: str,
) -> None:
    with pytest.raises(kind, match=f"^{re.escape(error)}$"):
        _ = background_delay(arguments)


def test_without_controls_drops_only_the_control_keys() -> None:
    assert without_controls({"text": "a", "background": True, "delay": 1}) == {
        "text": "a",
    }
    assert without_controls({}) == {}


# --- BackgroundTask --------------------------------------------------------------


def test_background_task_describes_itself() -> None:
    tool = BackgroundTask()
    assert tool.name == "BackgroundTask"
    assert "`background: true`" in tool.description
    assert tool.schema["required"] == ["operation"]
    assert tool.summary({"operation": "list"}) == "BackgroundTask list"
    assert (
        tool.summary({"operation": "cancel", "id": "c1"}) == "BackgroundTask cancel c1"
    )
    assert tool.summary({}) == "BackgroundTask ?"
    assert tool.summary({"operation": "list", "id": ""}) == "BackgroundTask list"
    assert tool.serialize_key(_CONTEXT, {"operation": "list"}) is None


@pytest.mark.asyncio
async def test_background_task_alone_has_no_jobs_to_manage() -> None:
    result = await BackgroundTask().call(_CONTEXT, {"operation": "list"})
    assert result == _error("BackgroundTask runs only inside an agent's runtime.")


@pytest.mark.asyncio
async def test_list_without_jobs() -> None:
    table = FakeTable(listed={"c1": _job("Bash", "running", 0)})
    result = await BackgroundTask().manage(table, {"operation": "list"})
    assert result == UncategorizedToolResult(call_id="", content="No background jobs.")


@pytest.mark.asyncio
async def test_list_shows_each_background_job() -> None:
    table = FakeTable(
        listed={
            "fg": _job("Read", "running", 1),
            "call_long_id": _job("Bash", "detached", 12.4),
            "d1": _job("Grep", "sleeping", 3),
        },
    )
    result = await BackgroundTask().manage(table, {"operation": "list"})
    assert result.call_id == ""
    assert result.text().splitlines() == [
        f"{'ID':<12}  {'TOOL':<20}  {'PHASE':<8}  ELAPSED",
        f"{'call_long_id':<12}  {'Bash':<20}  {'running':<8}  12s",
        f"{'d1':<12}  {'Grep':<20}  {'sleeping':<8}  3s",
    ]


@pytest.mark.asyncio
async def test_cancel_cancels_the_job() -> None:
    job = _job("Bash", "detached", 0)
    table = FakeTable(listed={"c1": job})
    result = await BackgroundTask().manage(table, {"operation": "cancel", "id": "c1"})
    assert result == UncategorizedToolResult(
        call_id="",
        content="Cancelling Bash (c1).",
    )
    assert job.cancelled == [True]


@pytest.mark.asyncio
async def test_foreground_returns_the_jobs_result() -> None:
    real = UncategorizedToolResult(call_id="c1", content="done", is_error=True)
    table = FakeTable(listed={"c1": _job("Bash", "sleeping", 0)}, results={"c1": real})
    result = await BackgroundTask().manage(
        table,
        {"operation": "foreground", "id": "c1"},
    )
    assert result is real
    assert table.waited == ["c1"]


@pytest.mark.parametrize("operation", ["cancel", "foreground"])
@pytest.mark.asyncio
async def test_an_operation_on_a_job_needs_its_id(operation: str) -> None:
    result = await BackgroundTask().manage(FakeTable(), {"operation": operation})
    assert result == _error(f"{operation} requires an id.")


@pytest.mark.parametrize("operation", ["cancel", "foreground"])
@pytest.mark.asyncio
async def test_only_background_jobs_can_be_managed(operation: str) -> None:
    job = _job("Read", "running", 0)
    table = FakeTable(listed={"fg": job})
    result = await BackgroundTask().manage(table, {"operation": operation, "id": "fg"})
    assert result == _error("No background job 'fg'; list shows them.")
    assert job.cancelled == []
    assert table.waited == []


@pytest.mark.asyncio
async def test_a_job_that_ends_with_no_result_is_unknown() -> None:
    table = FakeTable(listed={"c1": _job("Bash", "detached", 0)})
    result = await BackgroundTask().manage(
        table,
        {"operation": "foreground", "id": "c1"},
    )
    assert result.is_error is True
    assert table.waited == ["c1"]


@pytest.mark.asyncio
async def test_an_unknown_operation_is_an_argument_error() -> None:
    with pytest.raises(ValueError, match="'operation' must be one of"):
        _ = await BackgroundTask().manage(FakeTable(), {"operation": "fg"})


def _error(text: str) -> UncategorizedToolResult:
    return UncategorizedToolResult(call_id="", content=text, is_error=True)


def _properties(tool: Tool) -> Mapping[str, PlainTree]:
    none: dict[str, PlainTree] = {}
    return from_plain(tool.schema.get("properties"), dict[str, PlainTree], default=none)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
