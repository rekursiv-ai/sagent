"""Tests for the generic gate commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import argparse
import os
import shutil
import sys

import pytest

from sagent.lib.testing.checks import assert_help_without_docstring
from tools.gate import fakes, gate, spool


if TYPE_CHECKING:
    from collections.abc import Sequence


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
        (None, 0),
    ],
)
def test_only_a_positive_integer_turns_the_graph_on(
    value: str | None,
    expected: int,
) -> None:
    """``0`` must mean off; the bash gate once read ``-n "0"`` as on."""
    assert gate.positive(value) == expected


@pytest.mark.parametrize(
    ("environ", "on"),
    [({}, False), ({"X": ""}, False), ({"X": "0"}, True), ({"X": "1"}, True)],
    ids=["unset", "empty", "zero", "one"],
)
def test_a_flag_is_on_whenever_it_holds_anything(
    environ: dict[str, str],
    on: bool,
) -> None:
    """``0`` is on too: a flag is presence, and a count reads ``positive``."""
    assert gate.flagged(environ, "X") is on


def _run_gate(
    tmp_path: Path,
    tools: fakes.Recorder,
    files: Sequence[str],
    *,
    when: str | None = None,
    **environ: str,
) -> int:
    context = fakes.context(tmp_path, tools, **environ)
    fakes.touch(tmp_path, files)
    spool.append(context.directory, gate="leaks", key=9, paths=files)
    return gate.run(
        "leaks",
        context=context,
        when=when,
        command=["pkg.leaks", "--strict"],
    )


def test_a_file_deleted_after_spooling_is_not_handed_on(tmp_path: Path) -> None:
    """A probe file removed mid-run crashed ty and pyrefly on a missing path."""
    tools = fakes.Recorder()
    context = fakes.context(tmp_path, tools)
    (context.root / "kept.py").touch()
    spool.append(
        context.directory,
        gate="ty-check",
        key=9,
        paths=["kept.py", "gone_test.py"],
    )

    assert gate.drained("ty-check", context=context) == ["kept.py"]
    assert gate.drained("ty-check", context=context) is None


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
        Path(name).touch()
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

    assert gate.output_argv(["tool"]) == expected
    assert gate.run_argv(["tool"]) == status


def test_output_lets_the_tools_diagnostics_through(
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A gate that falls back must say why on the hook's own stderr."""
    assert (
        gate.output_argv(
            [sys.executable, "-c", "import sys; print('out'); sys.stderr.write('why')"],
        )
        == "out\n"
    )
    assert capfd.readouterr().err == "why"


def test_a_failing_tools_output_is_none_not_a_raise() -> None:
    assert gate.output_argv([sys.executable, "-c", "raise SystemExit(3)"]) is None


def test_a_missing_tool_is_named_by_output_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert gate.output_argv(["no-such-tool-anywhere"]) is None
    assert capsys.readouterr().err == "no-such-tool-anywhere: command not found\n"


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
    assert gate.output_argv(["no-such-tool-anywhere"]) is None
    assert "no-such-tool-anywhere" in capsys.readouterr().err


def test_run_skips_a_gate_whose_opt_in_is_unset(tmp_path: Path) -> None:
    tools = fakes.Recorder()

    assert _run_gate(tmp_path, tools, ["a.py"], when="OPT_IN") == 0
    assert tools.calls == []


def test_the_cli_runs_only_when_its_opt_in_is_set(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Path("a.py").touch()
    for argv in (
        ["gate", "spool", "spooled", "a.py"],
        ["gate", "run", "--when", "OPT", "spooled", "--", "pkg.tool"],
    ):
        monkeypatch.setattr(sys, "argv", argv)
        assert gate.main() == 0
    assert {call[0] for call in processes.calls} == {"ps"}


@pytest.mark.parametrize(
    ("argv", "shown"),
    [
        (
            ["--help"],
            [
                "Add files to a gate's spool.",
                "Run a module once over a spool.",
                "Run a command at the worker count.",
            ],
        ),
        (["spool", "--help"], ["The gate whose spool to add to.", "Files pre-commit"]),
        (
            ["run", "--help"],
            ["The gate, naming its spool and worker override.", "--when VARIABLE"],
        ),
        (["threaded", "--help"], ["The gate, naming its worker override."]),
    ],
    ids=["commands", "spool", "run", "threaded"],
)
def test_each_command_explains_itself(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    shown: list[str],
) -> None:
    del processes
    monkeypatch.setattr(sys, "argv", ["gate", *argv])
    with pytest.raises(SystemExit, match=r"^0$"):
        gate.main()
    out = " ".join(capsys.readouterr().out.split())
    assert [text for text in shown if text not in out] == []


def test_the_help_opens_with_the_module_docstring(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del processes
    monkeypatch.setattr(sys, "argv", ["gate", "--help"])
    with pytest.raises(SystemExit):
        gate.main()
    assert (
        capsys.readouterr().out.split("\n\n")[1]
        == (gate.__doc__ or "").split(
            "\n\n",
        )[0]
    )


def test_without_a_docstring_the_help_has_no_description(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``python -OO`` strips ``__doc__``; the help must still run, and say nothing."""
    del processes
    assert_help_without_docstring(monkeypatch, gate, gate.main)
    assert capsys.readouterr().out.split("\n\n")[1].startswith("positional arguments")


@pytest.mark.parametrize("command", ["run", "threaded"])
def test_a_command_with_nothing_after_its_separator_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    with pytest.raises(SystemExit, match=r"^2$"):
        gate._parse_args(argparse.ArgumentParser(), [command, "leaks", "--"])
    assert f"{command}: name the command after --" in capsys.readouterr().err


def test_naming_no_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit, match=r"^2$"):
        gate._parse_args(argparse.ArgumentParser(), [])


def test_a_second_separator_reaches_the_command() -> None:
    flags = gate._parse_args(
        argparse.ArgumentParser(),
        ["threaded", "tool", "--", "--", "x"],
    )
    assert flags.argv == ["--", "x"]


def test_the_context_runs_real_tools_from_here(
    processes: fakes.Processes,
    tmp_path: Path,
) -> None:
    processes.stdout = "1 /usr/bin/python -m pre_commit run\n"
    context = gate.context()

    assert (context.root, context.environ) == (tmp_path, os.environ)
    assert context.directory == tmp_path / f"precommit-spool-{os.getuid()}"
    assert context.key == spool.run_key(os.getpid(), process=gate._process)
    assert (context.run, context.output, context.installed, context.which) == (
        gate.run_argv,
        gate.output_argv,
        gate._installed,
        shutil.which,
    )


def test_a_real_process_is_its_parent_and_command() -> None:
    found = gate._process(os.getpid())

    assert found is not None
    assert found[0] == os.getppid()
    # Xdist retitles its workers, and ``ps`` may cut a retitled line short.
    argv = Path(f"/proc/{os.getpid()}/cmdline").read_bytes().split(b"\0")
    title = " ".join(arg.decode() for arg in argv if arg)
    assert found[1]
    assert title.startswith(found[1])


def test_a_process_that_is_gone_is_none() -> None:
    assert gate._process(2**22 + 1) is None


def test_a_process_is_read_from_ps(processes: fakes.Processes) -> None:
    processes.stdout = "1 x\n"
    gate._process(5)

    assert processes.calls == [["ps", "-o", "ppid=,command=", "-p", "5"]]


def test_a_failing_command_reports_its_own_status() -> None:
    assert gate.run_argv([sys.executable, "-c", "raise SystemExit(3)"]) == 3


def test_a_missing_tool_run_names_itself(capsys: pytest.CaptureFixture[str]) -> None:
    assert gate.run_argv(["no-such-tool-anywhere"]) == 127
    assert capsys.readouterr().err == "no-such-tool-anywhere: command not found\n"


@pytest.mark.parametrize(
    ("files", "dependents", "hops"),
    [([], "pkg.deps", 1), (["a.py"], None, 1), (["a.py"], "pkg.deps", 0)],
    ids=["no-files", "no-module", "no-hops"],
)
def test_a_walk_that_is_off_runs_nothing(
    tmp_path: Path,
    files: list[str],
    dependents: str | None,
    hops: int,
) -> None:
    tools = fakes.Recorder(output="found.py\n")

    assert (
        gate.walk(
            fakes.context(tmp_path, tools),
            files=files,
            dependents=dependents,
            hops=hops,
            emit="files",
            workers=2,
        )
        == []
    )
    assert tools.calls == []


@pytest.mark.parametrize(
    ("output", "found"),
    [("x.py\ny_test.py::t\n", ["x.py", "y_test.py::t"]), (None, [])],
    ids=["found", "failed"],
)
def test_a_walk_asks_the_module_for_what_lies_hops_away(
    tmp_path: Path,
    output: str | None,
    found: list[str],
) -> None:
    tools = fakes.Recorder(output=output)

    assert (
        gate.walk(
            fakes.context(tmp_path, tools),
            files=["a.py", "b.py"],
            dependents="pkg.deps",
            hops=2,
            emit="nodeids",
            workers=3,
        )
        == found
    )
    assert tools.calls == [
        [
            sys.executable,
            "-m",
            "pkg.deps",
            "--emit",
            "nodeids",
            "--depth",
            "2",
            "--workers",
            "3",
            "--",
            "a.py",
            "b.py",
        ],
    ]


class _Dependents(argparse.Namespace):
    dependents: str | None = None


def test_dependents_is_an_optional_module_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser()
    gate.add_dependents(parser)

    assert parser.parse_args([], namespace=_Dependents()).dependents is None
    named = parser.parse_args(["--dependents", "pkg.deps"], namespace=_Dependents())
    assert named.dependents == "pkg.deps"
    parser.print_help()
    out = " ".join(capsys.readouterr().out.split())
    assert (
        "--dependents MODULE Widen to dependents with this module's CLI when opted in."
        in out
    )


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
