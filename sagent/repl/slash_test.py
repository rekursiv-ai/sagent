"""Tests for ``repl.slash``: line -> SlashAction parser."""

from __future__ import annotations

import pytest

from sagent.repl.slash import (
    Clear,
    Compact,
    Defer,
    Halt,
    Help,
    Kill,
    Login,
    ModelSwitch,
    Quit,
    Recompact,
    Send,
    Tasks,
    Text,
    Thinking,
    Unknown,
    parse_slash,
)


@pytest.mark.parametrize("line", ["", "  ", "\t\n"])
def test_parse_slash_empty_returns_none(line: str) -> None:
    assert parse_slash(line) is None


@pytest.mark.parametrize(
    "line", ["/quit", "  /quit  ", "/QUIT", "/exit", "  /exit  ", "/EXIT"]
)
def test_parse_slash_quit(line: str) -> None:
    assert isinstance(parse_slash(line), Quit)


def test_parse_slash_help() -> None:
    assert isinstance(parse_slash("/help"), Help)


def test_parse_slash_tasks() -> None:
    assert isinstance(parse_slash("/tasks"), Tasks)


def test_parse_slash_clear() -> None:
    assert isinstance(parse_slash("/clear"), Clear)


def test_parse_slash_login() -> None:
    assert isinstance(parse_slash("/login"), Login)


def test_parse_slash_compact_no_args() -> None:
    action = parse_slash("/compact")
    assert isinstance(action, Compact)
    assert action.args == ""


def test_parse_slash_compact_with_args() -> None:
    action = parse_slash("/compact keep recent stuff")
    assert isinstance(action, Compact)
    assert action.args == "keep recent stuff"


def test_parse_slash_recompact_with_args() -> None:
    action = parse_slash("/recompact extra hints")
    assert isinstance(action, Recompact)
    assert action.args == "extra hints"


def test_recompact_action_docstring_describes_compact_alias() -> None:
    assert Recompact.__doc__ is not None
    assert "alias" in Recompact.__doc__.lower()
    assert "/compact" in Recompact.__doc__
    assert "reload" not in Recompact.__doc__.lower()


def test_parse_slash_model_no_args() -> None:
    action = parse_slash("/model")
    assert isinstance(action, ModelSwitch)
    assert action.args == ""


def test_parse_slash_model_with_args() -> None:
    action = parse_slash("/model claude-opus-4-7")
    assert isinstance(action, ModelSwitch)
    assert action.args == "claude-opus-4-7"


def test_parse_slash_provider_translates_to_model_switch() -> None:
    action = parse_slash("/provider Anthropic")
    assert isinstance(action, ModelSwitch)
    assert action.args == "--provider Anthropic"


def test_parse_slash_provider_no_args() -> None:
    action = parse_slash("/provider")
    assert isinstance(action, ModelSwitch)
    assert action.args == ""


@pytest.mark.parametrize(
    "command",
    [
        "adaptive-show",
        "adaptive-hide",
        "on-show",
        "on-hide",
        "off-hide",
        "redact-hide",
        "adaptive",
        "on",
        "off",
        "redact",
        "show",
        "hide",
    ],
)
def test_parse_slash_thinking_commands(command: str) -> None:
    action = parse_slash(f"/thinking {command}")
    assert isinstance(action, Thinking)
    assert action.command == command


def test_parse_slash_thinking_missing_mode_returns_unknown() -> None:
    action = parse_slash("/thinking")
    assert isinstance(action, Unknown)
    assert "show" in action.text


def test_parse_slash_thinking_unknown_mode_returns_unknown() -> None:
    action = parse_slash("/thinking nope")
    assert isinstance(action, Unknown)
    assert "redact-hide" in action.text


def test_parse_slash_halt_no_target() -> None:
    action = parse_slash("/halt")
    assert isinstance(action, Halt)
    assert action.target == ""


def test_parse_slash_halt_with_target() -> None:
    action = parse_slash("/halt agent-A")
    assert isinstance(action, Halt)
    assert action.target == "agent-A"


def test_parse_slash_kill_target() -> None:
    action = parse_slash("/kill q1")
    assert isinstance(action, Kill)
    assert action.target == "q1"


def test_parse_slash_kill_all() -> None:
    action = parse_slash("/kill all")
    assert isinstance(action, Kill)
    assert action.target == "all"


def test_parse_slash_kill_no_target_is_unknown() -> None:
    action = parse_slash("/kill")
    assert isinstance(action, Unknown)
    assert "requires" in action.text


def test_parse_slash_send_with_message() -> None:
    action = parse_slash("/send fix-tools continue please")
    assert isinstance(action, Send)
    assert action.target == "fix-tools"
    assert action.content == "continue please"


def test_parse_slash_send_accepts_brace_target() -> None:
    action = parse_slash("/send {fix-tools,fix-compact} continue")
    assert isinstance(action, Send)
    assert action.target == "{fix-tools,fix-compact}"
    assert action.content == "continue"


def test_parse_slash_send_accepts_regex_target() -> None:
    action = parse_slash("/send /fix-.*/ continue")
    assert isinstance(action, Send)
    assert action.target == "/fix-.*/"
    assert action.content == "continue"


def test_parse_slash_send_missing_message_is_unknown() -> None:
    action = parse_slash("/send fix-tools")
    assert isinstance(action, Unknown)
    assert "requires" in action.text


def test_parse_slash_unknown_command() -> None:
    action = parse_slash("/foobar")
    assert isinstance(action, Unknown)
    assert "/foobar" in action.text


def test_parse_slash_text_is_plain() -> None:
    action = parse_slash("Hello, world!")
    assert isinstance(action, Text)
    assert action.content == "Hello, world!"


def test_parse_slash_text_strips_whitespace() -> None:
    action = parse_slash("   hi   ")
    assert isinstance(action, Text)
    assert action.content == "hi"


def test_parse_slash_defer_with_text() -> None:
    action = parse_slash("/defer when you have a moment")
    assert isinstance(action, Defer)
    assert action.content == "when you have a moment"


def test_parse_slash_defer_no_text_returns_unknown() -> None:
    """``/defer`` with no payload is invalid; surfaces as Unknown."""
    action = parse_slash("/defer")
    assert isinstance(action, Unknown)
    assert "requires" in action.text


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
