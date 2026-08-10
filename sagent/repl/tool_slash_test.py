"""Tests for the ``/tool NAME.key=value`` live-reconfiguration path."""

from __future__ import annotations

from sagent.agent.agent import Agent
from sagent.agent.agent_test import StubModel
from sagent.repl.input_pane import _dispatch_tool
from sagent.repl.render import RecordingPrinter
from sagent.tools.bash import Bash
from sagent.tools.display import Displayable


def _agent() -> Agent:
    return Agent(model=StubModel(), tools=[Bash()], compactor=None)


def _output_of(agent: Agent) -> str:
    """Read the live Bash tool's display setting."""
    tool = agent.tools_map["Bash"]
    assert isinstance(tool, Displayable)
    return tool.output


def test_tool_override_applies_to_the_live_tool() -> None:
    """Tools are frozen, so the swap must go through ``replace_tool``.

    Mutating the instance in place raises ``FrozenInstanceError``; doing
    it via ``object.__setattr__`` would skip the version bump that
    invalidates the cached provider-facing tool list.
    """
    agent = _agent()
    printer = RecordingPrinter()
    _dispatch_tool(agent, "Bash.output=off", printer)
    assert _output_of(agent) == "off"
    assert not printer.tool_errors
    assert printer.slash_blocks == ["[/tool] Bash: output=off"]


def test_tool_override_reaches_the_provider_facing_surface() -> None:
    """``live_tools`` is cached; a swap must invalidate that cache."""
    agent = _agent()
    before = agent.live_tools()
    _dispatch_tool(agent, "Bash.output=off", RecordingPrinter())
    after = agent.live_tools()
    assert before is not after
    assert _output_of(agent) == "off"


def test_unknown_key_reports_the_settable_ones() -> None:
    agent = _agent()
    printer = RecordingPrinter()
    _dispatch_tool(agent, "Bash.otuput=off", printer)
    assert printer.tool_errors
    assert "output" in printer.tool_errors[0]
    assert _output_of(agent) == "on"


def test_unknown_tool_reports_the_loaded_ones() -> None:
    agent = _agent()
    printer = RecordingPrinter()
    _dispatch_tool(agent, "Nope.output=off", printer)
    assert printer.tool_errors
    assert "Bash" in printer.tool_errors[0]


def test_invalid_value_names_the_choices() -> None:
    agent = _agent()
    printer = RecordingPrinter()
    _dispatch_tool(agent, "Bash.output=bogus", printer)
    assert printer.tool_errors
    assert "on, off" in printer.tool_errors[0]
    assert _output_of(agent) == "on"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
