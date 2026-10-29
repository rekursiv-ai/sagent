"""Tests for reading the tool calls a local model writes into its text."""

from __future__ import annotations

import pytest

from sagent.providers.selfhosted.calls import read_calls


_OFFERED = {"Read", "Bash"}


def _ds(*calls: str) -> str:
    """Return a DeepSeek block of ``calls``, each ``name`` and its JSON."""
    inner = "".join(f"<｜tool▁call▁begin｜>{call}<｜tool▁call▁end｜>" for call in calls)
    return f"<｜tool▁calls▁begin｜>{inner}<｜tool▁calls▁end｜>"


def _call(name: str, arguments: str) -> str:
    return f"function\n{name}\n```json\n{arguments}\n```"


def _named(text: str) -> list[tuple[str, dict[str, object]]]:
    calls, _ = read_calls(text, offered=_OFFERED)
    return [(call.name, dict(call.arguments)) for call in calls]


def test_qwen_calls_are_taken_out_of_the_text_in_order() -> None:
    text = (
        'Reading. <tool_call>{"name": "Read", "arguments": {"file_path": "a"}}</tool_call>'
        ' then <tool_call>\n{"name": "Bash", "arguments": {"command": "ls"}}\n</tool_call> done.'
    )
    calls, said = read_calls(text, offered=_OFFERED)
    assert [(call.name, dict(call.arguments)) for call in calls] == [
        ("Read", {"file_path": "a"}),
        ("Bash", {"command": "ls"}),
    ]
    assert said == "Reading.  then  done."
    ids = [call.call_id for call in calls]
    assert len(set(ids)) == 2
    assert all(call_id.startswith("call_") and len(call_id) == 17 for call_id in ids)


def test_a_call_takes_the_offered_tools_own_spelling() -> None:
    # The runtime looks tools up by their exact name.
    assert _named('<tool_call>{"name": "read", "arguments": {}}</tool_call>') == [
        ("Read", {}),
    ]
    assert _named(_ds(_call("BASH", '{"command": "ls"}'))) == [
        ("Bash", {"command": "ls"}),
    ]


def test_a_call_with_null_arguments_has_none() -> None:
    assert _named('<tool_call>{"name": "Read", "arguments": null}</tool_call>') == [
        ("Read", {}),
    ]
    assert _named('<tool_call>{"name": "Read"}</tool_call>') == [("Read", {})]


@pytest.mark.parametrize(
    "block",
    [
        "<tool_call>{not json}</tool_call>",
        '<tool_call>["Read"]</tool_call>',
        '<tool_call>{"name": "", "arguments": {}}</tool_call>',
        '<tool_call>{"name": 7, "arguments": {}}</tool_call>',
        '<tool_call>{"name": "Read", "arguments": [1]}</tool_call>',
        '<tool_call>{"name": "Write", "arguments": {}}</tool_call>',
        _ds(),
        _ds(_call("Read", '{"file_path": "a"}'), _call("Write", "{}")),
        _ds(_call("Read", "{bad json}")),
        _ds(_call("Read", "[1]")),
        _ds("function\nRead\nno fence"),
        _ds("no function line\n```json\n{}\n```"),
    ],
)
def test_a_block_that_is_no_call_to_an_offered_tool_stays_in_the_text(
    block: str,
) -> None:
    calls, said = read_calls(f"before {block} after", offered=_OFFERED)
    assert calls == []
    assert said == f"before {block} after"


def test_deepseek_blocks_yield_every_call_they_hold() -> None:
    text = (
        "a "
        + _ds(_call("Read", '{"file_path": "x"}'), _call("Bash", '{"command": "pwd"}'))
        + " b"
    )
    calls, said = read_calls(text, offered=_OFFERED)
    assert [(call.name, dict(call.arguments)) for call in calls] == [
        ("Read", {"file_path": "x"}),
        ("Bash", {"command": "pwd"}),
    ]
    assert said == "a  b"


def test_both_dialects_in_one_reply_are_read_qwen_first() -> None:
    text = (
        _ds(_call("Bash", "{}"))
        + '<tool_call>{"name": "Read", "arguments": {}}</tool_call>'
    )
    assert [name for name, _ in _named(text)] == ["Read", "Bash"]


def test_text_without_calls_is_returned_trimmed() -> None:
    assert read_calls("  just words \n", offered=set()) == ([], "just words")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
