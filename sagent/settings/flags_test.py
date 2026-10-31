"""Tests for splitting a command line into argparse's flags and settings."""

from __future__ import annotations

from typing import Protocol, cast

import re
import shlex

import pytest

from sagent.settings.exceptions import HelpRequestedError, SettingError
from sagent.settings.flags import ActionParser, flags


_STATIC = frozenset({"--resume", "--tools", "--name", "--headless", "-h", "--help"})


@pytest.mark.parametrize(
    ("argv", "rest", "given"),
    [
        ([], [], []),
        (["login"], ["login"], []),
        (["--effort", "high"], [], [("effort", "high")]),
        (["--effort=high"], [], [("effort", "high")]),
        (["--model.thinking.effort", "max"], [], [("model.thinking.effort", "max")]),
        (
            ["--resume", "3f", "--model", "work@oaisub/gpt+1m", "--headless"],
            ["--resume", "3f", "--headless"],
            [("model", "work@oaisub/gpt+1m")],
        ),
        (["--name=Bob", "--context", "1m"], ["--name=Bob"], [("context", "1m")]),
        (["--output", "text", "-h"], ["-h"], [("output", "text")]),
        (["login", "--account", "", "x"], ["login", "x"], [("account", "")]),
        # After ``--``, every word is argparse's, a setting's spelling included.
        (
            ["--effort", "high", "--", "--model", "x"],
            ["--", "--model", "x"],
            [("effort", "high")],
        ),
    ],
)
def test_a_flag_argparse_does_not_know_is_a_setting_and_takes_one_value(
    argv: list[str],
    rest: list[str],
    given: list[tuple[str, str]],
) -> None:
    assert flags(argv, static=_STATIC) == (rest, given)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--effort"], "--effort needs a value"),
        (["--effort", "--model", "x"], "--effort needs a value"),
        (["--=x"], "'--=x' names no setting"),
    ],
)
def test_a_setting_without_a_value_is_refused(argv: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        flags(argv, static=_STATIC)


class _Added(Protocol):
    """What ``_parser`` reads."""

    name: str
    target: str
    args: list[str]
    env: list[str]


def _parser() -> ActionParser:
    parser = ActionParser(("mcp", "add"), description="Add an MCP server.")
    parser.add_argument("name")
    parser.add_argument("target")
    parser.add_argument("args", nargs="*")
    parser.add_argument("-e", "--env", action="append", default=[])
    return parser


def test_what_follows_an_actions_name_is_read_as_a_command_line_would_be() -> None:
    argv = ["srv", "-e", "A=1", "-e", "B=two words", "--", "npx", "pkg", "--flag"]
    read = _parser().read(shlex.join(argv))
    assert read is not None
    parsed = cast(_Added, read)
    assert (parsed.name, parsed.target, parsed.args) == (
        "srv",
        "npx",
        ["pkg", "--flag"],
    )
    assert parsed.env == ["A=1", "B=two words"]


@pytest.mark.parametrize("asked", ["-h", "--help", "srv x --help"])
def test_an_actions_help_is_returned_not_printed_or_exited(
    asked: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = _parser()
    assert parser.read(asked) is None
    help_text = parser.format_help()
    assert help_text.startswith("usage: sagent mcp add [-h] [-e ENV]")
    assert "Add an MCP server." in help_text
    assert capsys.readouterr() == ("", "")


def test_printing_an_actions_help_raises_rather_than_prints(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(HelpRequestedError):
        _parser().print_help()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    ("given", "said"),
    [
        ("", "the following arguments are required: name, target"),
        ("srv x --nope", "unrecognized arguments: --nope"),
        ("srv 'unclosed", "No closing quotation"),
    ],
)
def test_a_mistake_in_an_actions_arguments_raises_and_never_exits(
    given: str,
    said: str,
) -> None:
    with pytest.raises(SettingError, match=said) as error:
        _parser().read(given)
    assert str(error.value).endswith("; see sagent mcp add --help")


def test_an_exit_argparse_asks_for_raises_instead() -> None:
    with pytest.raises(SettingError, match=r"^see sagent mcp add --help$"):
        _parser().exit()


def test_a_help_word_after_the_separator_is_an_argument() -> None:
    read = _parser().read("srv -- npx --help")
    assert read is not None
    assert cast(_Added, read).args == ["--help"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
