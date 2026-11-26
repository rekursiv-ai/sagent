"""The inspector reads full records and respects hidden tool displays."""

import asyncio
import io

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.plain_text import PlainTextOutput

import pytest

from sagent.agent.agent import Agent
from sagent.journal.types import EditResult, Splice, ToolCall, ToolMessage, WriteResult
from sagent.model.scripted import ScriptedProvider
from sagent.tools.bash import Bash
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tools.edit import Edit
from sagent.tools.write import Write
from sagent.tui.details import entries, inspect


def test_entries_keep_the_full_command_and_diff() -> None:
    agent = Agent(model=ScriptedProvider()["scripted"], tools=[Bash(), Edit()])
    command = "\n".join(f"echo {n}" for n in range(20))
    agent.session.append(
        ToolCall(name="Bash", call_id="b", arguments={"command": command}),
        ToolCall(name="Edit", call_id="e"),
        ToolMessage(
            call_id="e",
            state="done",
            stdout=(Splice(before="old", after="new"),),
            result=EditResult(path="a.py"),
        ),
    )
    assert entries(agent) == [
        ("Bash", command),
        ("Edit a.py", "@@ -1,1 +1,1 @@\n-old\n+new"),
    ]


def test_hidden_commands_stay_out_of_the_viewer() -> None:
    bash = Bash(display=ToolDisplay(command=DisplaySpec(lines=(0, 0))))
    agent = Agent(model=ScriptedProvider()["scripted"], tools=[bash])
    agent.session.append(
        ToolCall(name="Bash", call_id="b", arguments={"command": "private"}),
    )
    assert entries(agent) == []


def test_write_diff_keeps_every_line() -> None:
    tool = Write(display=ToolDisplay(output=DisplaySpec(lines=(1, 0))))
    agent = Agent(model=ScriptedProvider()["scripted"], tools=[tool])
    agent.session.append(
        ToolMessage(
            call_id="w",
            state="done",
            result=WriteResult(path="a", content="one\ntwo"),
        ),
    )
    assert entries(agent) == [("Write a", "@@ -0,0 +1,2 @@\n+one\n+two")]


@pytest.mark.asyncio
async def test_viewer_moves_between_originals_and_searches() -> None:
    screen = io.StringIO()
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=PlainTextOutput(screen)),
    ):
        pipe.send_text("\x10\x0e\x06needle\r" + "q")
        await asyncio.wait_for(
            inspect([("first", "one"), ("second", "a needle")], 2),
            timeout=3,
        )
    assert "second" in screen.getvalue()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
