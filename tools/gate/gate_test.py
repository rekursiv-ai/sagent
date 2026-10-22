"""Tests for the generic gate commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import argparse
import sys

import pytest

from tools.gate import fakes, gate, spool


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@pytest.fixture
def processes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> fakes.Processes:
    """Record what ``main`` runs, under no gate's settings."""
    return fakes.isolate(tmp_path, monkeypatch=monkeypatch)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({"WORKERS_DEFAULT": "3"}, 3),
        ({"WORKERS_DEFAULT": "3", "WORKERS_PYREFLY_CHECK": "7"}, 7),
        ({"WORKERS_DEFAULT": "3", "WORKERS_PYREFLY_CHECK": ""}, 3),
        ({"WORKERS_DEFAULT": "3", "WORKERS_PYREFLY_CHECK": "junk"}, 1),
    ],
)
def test_a_gate_honors_its_own_worker_override(
    environ: dict[str, str],
    expected: int,
) -> None:
    """CI pins one gate's workers without touching the others."""
    assert gate.workers("pyrefly-check", environ=environ) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("3", 3),
        ("10", 10),
        ("0", 0),
        ("00", 0),
        ("01", 0),
        ("", 0),
        ("-1", 0),
        (" 1 ", 0),
        ("true", 0),
        ("1x", 0),
    ],
)
def test_only_a_positive_integer_turns_the_graph_on(value: str, expected: int) -> None:
    """``0`` must mean off; the bash gate once read ``-n "0"`` as on."""
    assert gate.positive(value) == expected


def _run_gate(
    tmp_path: Path,
    tools: fakes.Recorder,
    files: Sequence[str],
    *,
    when: str | None = None,
    **environ: str,
) -> int:
    context = fakes.context(tmp_path, tools, **environ)
    spool.append(context.directory, gate="leaks", key=9, paths=files)
    return gate.run(
        "leaks",
        context=context,
        when=when,
        command=["pkg.leaks", "--strict"],
    )


def test_run_hands_a_module_the_spool_at_its_worker_count(tmp_path: Path) -> None:
    tools = fakes.Recorder(status=3)

    assert _run_gate(tmp_path, tools, ["a b.py", "c.py"], WORKERS_LEAKS="2") == 3
    assert tools.calls[0][1:] == [
        "-m",
        "pkg.leaks",
        "--strict",
        "--workers",
        "2",
        "--",
        "a b.py",
        "c.py",
    ]


def test_run_with_nothing_spooled_runs_nothing(tmp_path: Path) -> None:
    """With no files the mutation runner mutates `.`: 5940 files."""
    tools = fakes.Recorder()

    assert _run_gate(tmp_path, tools, []) == 0
    assert tools.calls == []


@pytest.mark.parametrize(
    ("value", "runs"),
    [
        ("1", True),
        ("2", True),
        ("0", False),
        ("", False),
        ("01", False),
        ("true", False),
        ("yes", False),
        (" 1 ", False),
        ("1x", False),
    ],
)
def test_run_waits_for_its_opt_in(tmp_path: Path, value: str, runs: bool) -> None:
    tools = fakes.Recorder()

    _run_gate(tmp_path, tools, ["a.py"], when="OPT_IN", OPT_IN=value)

    assert bool(tools.calls) is runs
    assert spool.drain(tmp_path / "spool", gate="leaks", key=9) is None


def test_run_without_its_spool_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tools = fakes.Recorder()

    assert (
        gate.run(
            "leaks",
            context=fakes.context(tmp_path, tools),
            when=None,
            command=["pkg.leaks"],
        )
        == 1
    )
    assert tools.calls == []
    assert "leaks-spool" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "spool_name", "when", "module"),
    [
        (
            ["run", "--when", "OPT", "mutation", "--", "pkg.m", "--explain"],
            "mutation",
            "OPT",
            ["pkg.m", "--explain"],
        ),
        (["run", "leaks", "--", "pkg.leaks"], "leaks", None, ["pkg.leaks"]),
    ],
)
def test_run_parses_its_options_before_the_module(
    argv: list[str],
    spool_name: str,
    when: str | None,
    module: list[str],
) -> None:
    """The module's own flags follow ``--``; ``--when`` once became the module."""
    flags = gate._parse_args(argparse.ArgumentParser(), argv)

    assert (flags.gate, flags.when, flags.argv) == (spool_name, when, module)


def test_run_without_a_module_is_a_usage_error() -> None:
    with pytest.raises(SystemExit, match="2"):
        gate._parse_args(argparse.ArgumentParser(), ["run", "leaks", "--"])


def test_the_cli_spools_then_runs_one_module(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside pre-commit every call shares key 1, so the spool meets its gate."""
    for name in ("a.py", "b.py"):
        monkeypatch.setattr(sys, "argv", ["gate", "spool", "spooled", name])
        assert gate.main() == 0
    monkeypatch.setattr(
        sys,
        "argv",
        ["gate", "run", "spooled", "--", "pkg.tool", "--flag"],
    )

    assert gate.main() == 4
    assert processes.calls[-1][1:] == [
        "-m",
        "pkg.tool",
        "--flag",
        "--workers",
        "1",
        "--",
        "a.py",
        "b.py",
    ]


def test_the_cli_runs_a_threaded_tool(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["gate", "threaded", "tool", "--", "tool", "--threads={workers}"],
    )

    assert gate.main() == 4
    assert processes.calls[-1] == ["tool", "--threads=1"]


def test_every_command_runs_below_interactive_priority(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["gate", "spool", "x", "a.py"])

    assert gate.main() == 0
    assert processes.niceness == [19]


@pytest.mark.parametrize(("status", "expected"), [(0, "a\n"), (1, None)])
def test_output_is_stdout_only_on_success(
    processes: fakes.Processes,
    status: int,
    expected: str | None,
) -> None:
    processes.status, processes.stdout = status, "a\n"

    assert gate._output(["tool"]) == expected
    assert gate.run_argv(["tool"]) == status


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [("  77 /usr/bin/pre-commit run\n", (77, "/usr/bin/pre-commit run")), ("", None)],
)
def test_a_process_is_its_parent_and_command(
    processes: fakes.Processes,
    stdout: str,
    expected: tuple[int, str] | None,
) -> None:
    processes.stdout = stdout

    assert gate._process(5) == expected


def test_installed_finds_importable_modules() -> None:
    assert gate._installed("argparse")
    assert not gate._installed("no_such_module_anywhere")
    assert not gate._installed("no_such_module_anywhere.child")


def test_a_missing_tool_is_exit_127_not_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A commit must not die on a tool a machine lacks, as ``npm`` once did."""
    assert gate.run_argv(["no-such-tool-anywhere"]) == 127
    assert gate._output(["no-such-tool-anywhere"]) is None
    assert "no-such-tool-anywhere" in capsys.readouterr().err


def test_threaded_fills_in_the_gate_worker_count(tmp_path: Path) -> None:
    tools = fakes.Recorder()

    gate.threaded(
        "pyrefly-check-global",
        context=fakes.context(tmp_path, tools, WORKERS_PYREFLY_CHECK_GLOBAL="2"),
        command=["pyrefly", "check", "--threads={workers}"],
    )

    assert tools.calls == [["pyrefly", "check", "--threads=2"]]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
