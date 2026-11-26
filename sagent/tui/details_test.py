"""The inspector reads full records and respects hidden tool displays."""

from sagent.agent.agent import Agent
from sagent.journal.types import EditResult, Splice, ToolCall, ToolMessage
from sagent.model.scripted import ScriptedProvider
from sagent.tools.bash import Bash
from sagent.tools.display import DisplaySpec, ToolDisplay
from sagent.tools.edit import Edit
from sagent.tui.details import entries


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
        ToolCall(name="Bash", call_id="b", arguments={"command": "private"})
    )
    assert entries(agent) == []


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
