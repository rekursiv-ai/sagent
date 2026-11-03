"""Tests for ``sagent.lib.testing.checks``."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import os
import re
import signal
import subprocess
import sys

import pytest

from sagent.lib.testing import checks
from sagent.lib.testing.checks import (
    assert_help_without_docstring,
    bash_path,
    default_signals,
    smuggle,
)


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.cli_python_subprocess
@pytest.mark.parametrize(("prefixed", "seen"), [(True, "default"), (False, "ignored")])
def test_a_child_starts_with_the_signal_at_its_default_action(
    *,
    prefixed: bool,
    seen: str,
) -> None:
    # Unprefixed is the control: it proves the parent below really hands its
    # children SIGINT ignored, as a suite started with ``nohup ... &`` does.
    child = [
        *(default_signals(signal.SIGINT) if prefixed else []),
        sys.executable,
        "-c",
        (
            "import signal\n"
            "ignored = signal.getsignal(signal.SIGINT) is signal.SIG_IGN\n"
            "print('ignored' if ignored else 'default')\n"
        ),
    ]
    parent = (
        "import signal, subprocess, sys\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "subprocess.run(sys.argv[1:], check=True)\n"
    )

    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and inline scripts.
        [sys.executable, "-c", parent, *child],
        capture_output=True,
        check=True,
        text=True,
    )

    assert completed.stdout.strip() == seen


@pytest.mark.cli_python_subprocess
def test_the_command_keeps_the_pid_it_was_started_with() -> None:
    # Callers signal the child's pid and process group, so the reset must exec
    # the command in place rather than run it as a grandchild.
    process = subprocess.Popen(  # noqa: S603 -- fixed interpreter and inline script.
        [
            *default_signals(signal.SIGHUP),
            sys.executable,
            "-c",
            "import os; print(os.getpid())",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    stdout, _ = process.communicate(timeout=60)

    assert int(stdout) == process.pid


def test_the_prefix_runs_this_interpreter() -> None:
    # What the prefix DOES is the subprocess tests above; this pins the shape
    # they rely on, in the tier that runs on every commit.
    assert default_signals(signal.SIGINT, signal.SIGHUP) == [
        sys.executable,
        "-c",
        (
            "import os, signal, sys\n"
            "for number in (2, 1,):\n"
            "    signal.signal(number, signal.SIG_DFL)\n"
            "os.execvp(sys.argv[1], sys.argv[1:])\n"
        ),
    ]


def test_naming_no_signal_is_an_error() -> None:
    with pytest.raises(ValueError, match=r"^name at least one signal to reset$"):
        default_signals()


def test_help_that_exits_zero_without_a_docstring_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv: list[list[str]] = []

    def entry() -> None:
        argv.append(list(sys.argv))
        sys.exit(0)

    assert_help_without_docstring(monkeypatch, checks, entry)
    assert checks.__doc__ is None
    assert argv == [[checks.__name__, "--help"]]


def test_help_that_fails_without_a_docstring_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match=r"^Expected exit_info\.value\.code == 0\.$"):
        assert_help_without_docstring(monkeypatch, checks, partial(sys.exit, 2))


def test_smuggle_returns_the_value_itself() -> None:
    value = object()
    assert smuggle(value, int) is value


@pytest.mark.cli_bash
def test_the_bash_found_runs_what_bash_3_2_cannot() -> None:
    script = (
        "set -u; declare -A seen=([a]=1); empty=(); mapfile -t <<<x; echo ${empty[@]}"
    )
    ran = subprocess.run(  # noqa: S603 -- The bash under test, a fixed script.
        [bash_path(), "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )
    assert (ran.returncode, ran.stderr) == (0, "")


def test_an_older_bash_earlier_on_path_is_passed_over(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    old, new = _bash_in(tmp_path / "old", exits=1), _bash_in(tmp_path / "new", exits=0)
    monkeypatch.setenv("PATH", f"{old.parent}{os.pathsep}{new.parent}")
    assert bash_path() == str(new)


def test_without_a_path_the_default_search_path_is_used(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    new = _bash_in(tmp_path / "default", exits=0)
    monkeypatch.delenv("PATH", raising=False)
    monkeypatch.setattr(os, "defpath", str(new.parent))
    assert bash_path() == str(new)


def test_no_new_enough_bash_on_path_fails_naming_the_fix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    old = _bash_in(tmp_path / "old", exits=1)
    monkeypatch.setenv("PATH", str(old.parent))
    with pytest.raises(
        pytest.fail.Exception,
        match=f"has {re.escape(str(old))}: brew install bash$",
    ):
        _ = bash_path()


def _bash_in(directory: Path, *, exits: int) -> Path:
    """Write a stand-in bash whose version probe exits ``exits``."""
    directory.mkdir()
    bash = directory / "bash"
    _ = bash.write_text(f"#!/bin/sh\nexit {exits}\n")
    bash.chmod(0o755)
    return bash


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
