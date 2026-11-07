"""Tests for splitting a command line into argparse's flags and settings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import re
import shlex

import pytest

from sagent.settings.exceptions import HelpRequestedError, SettingError
from sagent.settings.flags import ActionParser, flags


if TYPE_CHECKING:
    import argparse


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


async def _added(parsed: argparse.Namespace) -> str:
    added = cast(_Added, parsed)
    return f"added {added.name} -> {added.target}"


@pytest.mark.asyncio
async def test_an_actions_setting_runs_on_what_it_parsed_or_returns_its_help() -> None:
    parser = _parser()

    setting = parser.setting(_added, name=("add",), help_short="add an MCP server")

    assert (setting.name, setting.help_short, setting.cli) == (
        ("add",),
        "add an MCP server",
        True,
    )
    assert setting.help_long == parser.format_help()
    assert setting.usage == "[-h] [-e ENV] name target [args ...]"
    assert setting.flags == {"-h", "--help", "-e", "--env"}
    assert await setting.apply("srv npx") == "added srv -> npx"
    assert await setting.apply("--help") == parser.format_help()


def test_a_help_word_after_the_separator_is_an_argument() -> None:
    read = _parser().read("srv -- npx --help")
    assert read is not None
    assert cast(_Added, read).args == ["--help"]


@pytest.mark.asyncio
async def test_an_action_setting_exports_its_parser_and_runs_only_without_help() -> (
    None
):
    parser = _parser()
    seen: list[_Added] = []

    async def run(parsed: object) -> str:
        seen.append(cast(_Added, parsed))
        return f"added {seen[-1].name}"

    setting = parser.setting(run, name=("add",), help_short="add a server")
    assert setting.name == ("add",)
    assert setting.help_short == "add a server"
    assert setting.help_long == parser.format_help()
    assert setting.usage == "[-h] [-e ENV] name target [args ...]"
    assert setting.cli is True
    assert setting.flags == frozenset({"-h", "--help", "-e", "--env"})
    assert await setting.apply("--help") == parser.format_help()
    assert seen == []
    assert await setting.apply("srv -e A=1 -- npx pkg") == "added srv"
    (parsed,) = seen
    assert (parsed.name, parsed.target, parsed.args, parsed.env) == (
        "srv",
        "npx",
        ["pkg"],
        ["A=1"],
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
