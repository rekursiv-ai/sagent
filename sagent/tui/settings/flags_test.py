"""Tests for splitting a command line into argparse's flags and settings."""

from __future__ import annotations

import re

import pytest

from sagent.tui.settings.flags import flags


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


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
