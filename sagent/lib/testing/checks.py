"""Assertions and helpers a single test calls.

- ``assert_help_without_docstring``: a CLI's ``--help`` survives ``python -OO``.
- ``bash_path``: the bash, 4.4 or newer, a test runs a Linux host's script with.
- ``default_signals``: start a child with default signal handling, whatever the
  suite inherited. A process inherits an ignored signal across ``exec``, so a
  suite started as ``nohup pytest ... &`` hands its children SIGHUP and SIGINT
  ignored, and a non-interactive bash cannot trap a signal it was started with
  ignored. A test that signals such a child then measures how the suite was
  launched, not the code under test::

      subprocess.Popen([*default_signals(signal.SIGINT), bash_path(), script])

- ``smuggle``: pass a deliberately ill-typed value through a typed API, so a
  test can reach the runtime check that rejects it.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, cast

import os
import shutil
import subprocess
import sys

import pytest


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import signal


__all__ = ["assert_help_without_docstring", "bash_path", "default_signals", "smuggle"]


def assert_help_without_docstring(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    entry: Callable[[], object],
) -> None:
    """Assert ``--help`` exits 0 when ``python -OO`` has stripped ``__doc__``.

    Args:
      monkeypatch: The test's monkeypatch fixture.
      module: Module whose docstring feeds ``description=``.
      entry: Calls the entry point; it reads ``sys.argv``.

    """
    monkeypatch.setattr(module, "__doc__", None)
    monkeypatch.setattr(sys, "argv", [module.__name__, "--help"])
    with pytest.raises(SystemExit) as exit_info:
        entry()
    if exit_info.value.code != 0:
        raise ValueError("Expected exit_info.value.code == 0.")


def bash_path() -> str:
    """Return the first bash on ``PATH`` that is 4.4 or newer.

    It runs the scripts of Linux hosts, which use ``mapfile``, associative arrays,
    ``source <(...)`` and empty arrays under ``set -u``, none of which bash 3.2 runs.
    macOS keeps 3.2 at ``/bin/bash``, and its ``path_helper`` puts Homebrew's newer
    bash after it, so neither ``/bin/bash`` nor the first ``bash`` on ``PATH`` will
    do. A script an operator's Mac runs keeps ``/bin/bash`` in its test, which so
    checks it against the 3.2 that Mac runs it with.

    Returns:
      path: The bash's absolute path.

    """
    return _bash_on(os.environ.get("PATH", os.defpath))


def default_signals(*signums: signal.Signals) -> list[str]:
    """Return an argv prefix that runs the rest with ``signums`` at their default action.

    A short Python interpreter resets each signal and then ``exec``s the command
    in its own place, so the command keeps the pid, process group and session
    ``subprocess`` gave it.

    Args:
      signums: The signals the command must start with at their default action.

    Returns:
      prefix: Arguments to put before the command's own argv.

    """
    if not signums:
        raise ValueError("name at least one signal to reset")
    numbers = ", ".join(str(int(signum)) for signum in signums)
    return [
        sys.executable,
        "-c",
        (
            "import os, signal, sys\n"
            f"for number in ({numbers},):\n"
            "    signal.signal(number, signal.SIG_DFL)\n"
            "os.execvp(sys.argv[1], sys.argv[1:])\n"
        ),
    ]


def smuggle[T](value: object, as_type: type[T]) -> T:
    """Return ``value`` typed as ``as_type``, unchecked, for a negative test."""
    del as_type
    return cast("T", value)


@cache
def _bash_on(search: str) -> str:
    """Return the first bash 4.4 or newer in the ``search`` path, probed once."""
    found = [
        bash
        for directory in search.split(os.pathsep)
        if (bash := shutil.which("bash", path=directory)) is not None
    ]
    for bash in found:
        probe = subprocess.run(  # noqa: S603 -- A bash found on PATH, a fixed script.
            [bash, "-c", "(( BASH_VERSINFO[0] * 100 + BASH_VERSINFO[1] >= 404 ))"],
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            return bash
    pytest.fail(
        f"needs bash >= 4.4 on PATH, which has {', '.join(found) or 'none'}: "
        "brew install bash",
        pytrace=False,
    )
