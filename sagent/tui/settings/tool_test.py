"""Tests for a tool's display as settings, changed where the agent reads it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import re

from treekle import PlainTree

import pytest

from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tui.settings.harvest import harvest
from sagent.tui.settings.tool import ToolDisplaySettings


if TYPE_CHECKING:
    from sagent.tools.types import Tool, ToolContext
    from sagent.tui.settings.types import Setting


@dataclass(frozen=True, slots=True, kw_only=True)
class _Echo:
    """A tool whose display may change."""

    name: str = "Echo"
    description: str = "Echo."
    schema: Mapping[str, PlainTree] = field(default_factory=dict[str, "PlainTree"])
    display: ToolDisplay = field(
        default_factory=lambda: ToolDisplay(output=DisplaySpec(lines=(2, 2))),
    )

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return self.name

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        del context, arguments
        return UncategorizedToolResult(call_id="", content="ok")


class _Calls:
    """What every tool here does; each says how it holds its display."""

    name = "Echo"
    description = "Echo."

    @property
    def schema(self) -> Mapping[str, PlainTree]:
        schema: dict[str, PlainTree] = {}
        return schema

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        del arguments
        return self.name

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        del context, arguments
        return UncategorizedToolResult(call_id="", content="ok")


class _Fixed(_Calls):
    """A tool that is no dataclass, so it cannot be copied changed."""

    display = ToolDisplay()


@dataclass(frozen=True, slots=True, kw_only=True)
class _Given(_Calls):
    """A tool built with whatever display it is given: it declares no default."""

    display: ToolDisplay


@dataclass(frozen=True, slots=True, kw_only=True)
class _Plain(_Calls):
    """A tool whose display default is a value, not a factory."""

    display: ToolDisplay = field(default=ToolDisplay(output=DisplaySpec(width=40)))


def _settings(tools: dict[str, Tool]) -> dict[str, Setting]:
    owner = ToolDisplaySettings(tools=tools, name="Echo")
    return {".".join(s.name): s for s in harvest([(("tool", "Echo"), owner)])}


def test_a_tool_exports_each_display_setting() -> None:
    settings = _settings({"Echo": _Echo()})
    assert list(settings) == [
        "tool.Echo.display.command.lines",
        "tool.Echo.display.command.width",
        "tool.Echo.display.command.overflow",
        "tool.Echo.display.output.lines",
        "tool.Echo.display.output.width",
        "tool.Echo.display.output.overflow",
        "tool.Echo.display.command_language",
    ]
    shown = {
        name: (s.value() if s.value else "", tuple(s.options()))
        for name, s in settings.items()
    }
    assert shown["tool.Echo.display.command.lines"] == ("all (default)", ("default",))
    assert shown["tool.Echo.display.output.lines"] == ("2,2 (default)", ("default",))
    assert shown["tool.Echo.display.output.width"] == ("0 (default)", ("default",))
    assert shown["tool.Echo.display.output.overflow"] == (
        "fold (default)",
        ("default", "fold", "ellipsis"),
    )
    assert shown["tool.Echo.display.command_language"] == (
        " (default)",
        ("default",),
    )
    assert all(s.cli for s in settings.values())


@pytest.mark.asyncio
async def test_a_change_replaces_the_tool_where_the_agent_reads_it() -> None:
    before = _Echo()
    tools: dict[str, Tool] = {"Echo": before}
    settings = _settings(tools)
    assert await settings["tool.Echo.display.output.lines"].apply("3,1") == "3,1"
    language = settings["tool.Echo.display.command_language"]
    assert await language.apply("bash") == "bash"
    after = tools["Echo"]
    assert after.display.output == DisplaySpec(lines=(3, 1))
    assert after.display.command_language == "bash"
    assert before.display.output == DisplaySpec(lines=(2, 2))
    output_lines = settings["tool.Echo.display.output.lines"]
    assert await output_lines.apply("default") == "2,2 (default)"
    assert tools["Echo"].display.output == DisplaySpec(lines=(2, 2))


@pytest.mark.asyncio
async def test_a_value_that_does_not_fit_changes_nothing() -> None:
    tools: dict[str, Tool] = {"Echo": _Echo()}
    with pytest.raises(ValueError, match=re.escape("lines must be HEAD,TAIL or all")):
        await _settings(tools)["tool.Echo.display.output.lines"].apply("3")
    assert tools["Echo"].display.output == DisplaySpec(lines=(2, 2))


@pytest.mark.parametrize(
    "tool",
    [
        _Fixed(),
        _Given(display=ToolDisplay(output=DisplaySpec(width=80))),
        _Plain(),
    ],
    ids=["no-dataclass", "no-declared-default", "declared-value"],
)
def test_a_tool_s_default_display_is_the_one_it_is_built_with(
    tool: Tool,
) -> None:
    shown = _settings({"Echo": tool})["tool.Echo.display.output.width"]
    assert shown.value is not None
    assert shown.value() == f"{tool.display.output.width} (default)"


@pytest.mark.asyncio
async def test_a_tool_that_cannot_be_copied_says_so() -> None:
    tools: dict[str, Tool] = {"Echo": _Fixed()}
    with pytest.raises(ValueError, match=r"^Echo cannot change its display$"):
        await _settings(tools)["tool.Echo.display.output.width"].apply("80")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
