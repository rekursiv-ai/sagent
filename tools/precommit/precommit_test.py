"""Tests for the gates ``.pre-commit-config.yaml`` runs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, override

import argparse
import os
import subprocess
import sys
import tempfile

import coverage
import pytest

from tools.precommit import precommit, spool
from tools.precommit.tiers import TIERS


if TYPE_CHECKING:
    from collections.abc import Sequence


class _Recorder:
    """Stands in for running tools: records each argv, returns canned results."""

    def __init__(
        self,
        status: int = 0,
        output: str | None = None,
        *,
        installed: bool = True,
        on_path: bool = True,
        statuses: Sequence[int] = (),
    ) -> None:
        self.status = status
        self.statuses = list(statuses)
        self.output = output
        self.installed = installed
        self.on_path = on_path
        self.calls: list[list[str]] = []
        self.argfiles: list[list[str]] = []

    def run(self, argv: Sequence[str]) -> int:
        self.calls.append(list(argv))
        self.argfiles.extend(
            Path(arg[1:]).read_text().splitlines()
            for arg in argv
            if arg.startswith("@")
        )
        return self.statuses.pop(0) if self.statuses else self.status

    def read(self, argv: Sequence[str]) -> str | None:
        self.calls.append(list(argv))
        return self.output

    def has(self, module: str) -> bool:
        del module
        return self.installed

    def which(self, name: str) -> str | None:
        return f"/bin/{name}" if self.on_path else None


def _context(tmp_path: Path, tools: _Recorder, **environ: str) -> precommit.Context:
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text() if pyproject.is_file() else ""
    if "[tool.pyrefly]" not in text:
        pyproject.write_text(
            f'{text}[tool.pyrefly]\nproject-excludes = ["**/third_party/"]\n',
        )
    return precommit.Context(
        directory=tmp_path / "spool",
        key=9,
        root=root,
        environ={"WORKERS_DEFAULT": "3", **environ},
        run=tools.run,
        output=tools.read,
        installed=tools.has,
        which=tools.which,
    )


def _gate(
    name: str,
    tmp_path: Path,
    tools: _Recorder,
    *,
    dependents: str | None = None,
    fixer: str | None = None,
    **environ: str,
) -> int:
    return precommit.typecheck(
        name,
        context=_context(tmp_path, tools, **environ),
        dependents=dependents,
        fixer=fixer,
    )


def _spool(tmp_path: Path, *paths: str) -> None:
    """Spool ``paths`` as one pre-commit run does: to every checker's twin."""
    for gate in ("ty-check", "pyrefly-check", "basedpyright"):
        spool.append(tmp_path / "spool", gate=gate, key=9, paths=paths)


def test_ty_checks_every_spooled_file_once_minus_excludes(tmp_path: Path) -> None:
    _spool(tmp_path, "a.py")
    _spool(tmp_path, "x/third_party/v.py", "b.py", "a.py")
    tools = _Recorder()

    assert _gate("ty-check", tmp_path, tools) == 0
    assert [[Path(call[0]).name, *call[1:]] for call in tools.calls] == [
        ["ty", "check", "--force-exclude", "a.py", "b.py"],
    ]


def test_pyrefly_checks_what_ty_checked(tmp_path: Path) -> None:
    _spool(tmp_path, "a.py")
    _gate("ty-check", tmp_path, _Recorder())
    tools = _Recorder()

    assert _gate("pyrefly-check", tmp_path, tools) == 0
    assert [[Path(call[0]).name, *call[1:]] for call in tools.calls] == [
        ["pyrefly", "check", "--threads=3", "a.py"],
    ]


def test_pyrefly_checks_the_spool_when_ty_was_skipped(tmp_path: Path) -> None:
    """``SKIP=ty-check`` leaves nothing checked; pyrefly must not pass on nothing.

    It widens on its own then, at its own worker count.
    """
    _spool(tmp_path, "a.py", "third_party/v.py")
    tools = _Recorder(output="a.py\nb.py\nthird_party/v.py\n")

    status = _gate(
        "pyrefly-check",
        tmp_path,
        tools,
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_TYPECHECK="2",
        WORKERS_PYREFLY_CHECK="5",
    )

    walk, check = tools.calls
    assert status == 0
    assert walk[2:4] == ["pkg.affected", "--emit"]
    assert walk[walk.index("--workers") + 1] == "5"
    assert check[-2:] == ["a.py", "b.py"]


def test_pyrefly_checks_nothing_when_ty_checked_nothing(tmp_path: Path) -> None:
    """Every spooled file excluded: an empty hand-off, not a missing one."""
    _spool(tmp_path, "third_party/v.py")
    _gate("ty-check", tmp_path, _Recorder())
    _spool(tmp_path, "a.py")
    tools = _Recorder()

    assert _gate("pyrefly-check", tmp_path, tools) == 0
    assert tools.calls == []


@pytest.mark.parametrize(
    ("name", "checker"),
    [("pyrefly-check", "pyrefly"), ("basedpyright", "basedpyright")],
)
def test_the_fixer_runs_the_checker_in_its_place(
    tmp_path: Path,
    name: str,
    checker: str,
) -> None:
    _spool(tmp_path, "a.py")
    tools = _Recorder()

    _gate(name, tmp_path, tools, fixer="pkg.fixer")

    assert tools.calls[0][1:] == [
        "-m",
        "pkg.fixer",
        "--checker",
        checker,
        "--fix",
        "--threads",
        "3",
        "--",
        "a.py",
    ]


@pytest.mark.parametrize("name", ["ty-check", "pyrefly-check", "basedpyright"])
def test_nothing_to_check_runs_nothing(tmp_path: Path, name: str) -> None:
    spool.write(tmp_path / "spool", gate=name, key=9, paths=[])
    tools = _Recorder()

    assert _gate(name, tmp_path, tools) == 0
    assert tools.calls == []


@pytest.mark.parametrize("name", ["ty-check", "pyrefly-check", "basedpyright"])
def test_a_checker_without_its_spool_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    """``pre-commit run ty-check`` once ran without its twin and passed a type error."""
    tools = _Recorder()

    assert _gate(name, tmp_path, tools) == 1
    assert tools.calls == []
    assert f"{name}-spool" in capsys.readouterr().err


def test_basedpyright_checks_its_spool_at_its_thread_count(tmp_path: Path) -> None:
    _spool(tmp_path, "a.py", "b.py")
    tools = _Recorder()

    assert _gate("basedpyright", tmp_path, tools) == 0
    assert [[Path(call[0]).name, *call[1:]] for call in tools.calls] == [
        ["basedpyright", "--threads=3", "a.py", "b.py"],
    ]


@pytest.mark.parametrize("name", ["ty-check", "pyrefly-check", "basedpyright"])
def test_a_failing_tool_fails_the_gate(tmp_path: Path, name: str) -> None:
    _spool(tmp_path, "a.py")

    assert _gate(name, tmp_path, _Recorder(status=1)) == 1


def test_dependents_widen_the_files_then_excludes_apply(tmp_path: Path) -> None:
    _spool(tmp_path, "a.py")
    tools = _Recorder(output="a.py\nb.py\nthird_party/c.py\n")

    _gate(
        "ty-check",
        tmp_path,
        tools,
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_TYPECHECK="3",
    )

    walk, check = tools.calls
    assert walk[1:] == [
        "-m",
        "pkg.affected",
        "--emit",
        "files",
        "--depth",
        "3",
        "--workers",
        "3",
        "--",
        "a.py",
    ]
    assert check[-2:] == ["a.py", "b.py"]


@pytest.mark.parametrize("output", [None, ""])
def test_a_failed_or_empty_walk_checks_the_spooled_files(
    tmp_path: Path,
    output: str | None,
) -> None:
    """A broken index must not shrink the check to nothing."""
    _spool(tmp_path, "a.py")
    tools = _Recorder(output=output)

    _gate(
        "ty-check",
        tmp_path,
        tools,
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_TYPECHECK="3",
    )

    assert tools.calls[-1][-1] == "a.py"


@pytest.mark.parametrize(
    ("dependents", "value"),
    [(None, "3"), ("pkg.affected", "0"), ("pkg.affected", "")],
)
def test_the_walk_needs_both_the_module_and_the_opt_in(
    tmp_path: Path,
    dependents: str | None,
    value: str,
) -> None:
    _spool(tmp_path, "a.py")
    tools = _Recorder(output="b.py\n")

    _gate(
        "ty-check",
        tmp_path,
        tools,
        dependents=dependents,
        PRECOMMIT_CODEGRAPH_TYPECHECK=value,
    )

    assert [call[-1] for call in tools.calls] == ["a.py"]


def test_one_gates_depth_never_widens_the_other(tmp_path: Path) -> None:
    """Each gate reads its own variable; the other's is not an opt-in."""
    _spool(tmp_path, "a.py")
    checker = _Recorder(output="b.py\n")
    _gate(
        "ty-check",
        tmp_path,
        checker,
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_PYTEST="3",
    )
    tests = _Recorder(output="b_test.py::t\n", installed=False)
    _pytest(
        tmp_path,
        tests,
        ["a.py"],
        existing=["a_test.py"],
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_TYPECHECK="3",
    )

    assert [call[-1] for call in checker.calls] == ["a.py"]
    assert tests.argfiles == [["a_test.py"]]


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
    assert precommit.workers("pyrefly-check", environ=environ) == expected


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
    assert precommit.positive(value) == expected


def _pytest(
    tmp_path: Path,
    tools: _Recorder,
    files: Sequence[str],
    *,
    existing: Sequence[str] = (),
    dependents: str | None = None,
    **environ: str,
) -> int:
    context = _context(tmp_path, tools, **environ)
    for name in (*files, *existing):
        (context.root / name).parent.mkdir(parents=True, exist_ok=True)
        (context.root / name).touch()
    spool.append(context.directory, gate="pytest", key=9, paths=files)
    return precommit.pytest(context, dependents=dependents)


def test_pytest_runs_each_changed_files_pairs_with_its_directory_measured(
    tmp_path: Path,
) -> None:
    tools = _Recorder()

    _pytest(
        tmp_path,
        tools,
        ["pkg/a.py", "pkg/b_test.py", "top.py"],
        existing=["pkg/a_test.py", "pkg/tests/test_a.py", "pkg/b.py"],
    )

    erase, test = tools.calls
    assert erase[1:] == ["-m", "coverage", "erase"]
    assert test[1:4] == ["-m", "pytest", "-q"]
    assert tools.argfiles == [
        [
            "--cov=.",
            "--cov=pkg",
            "--cov-append",
            "--no-cov-on-fail",
            "--cov-fail-under=0",
            "--cov-report=",
            "pkg/a_test.py",
            "pkg/b_test.py",
            "pkg/tests/test_a.py",
        ],
    ]


def test_pytest_without_pytest_cov_measures_nothing(tmp_path: Path) -> None:
    tools = _Recorder(installed=False)

    _pytest(tmp_path, tools, ["a.py"], existing=["a_test.py"])

    assert [call[1:3] for call in tools.calls] == [["-m", "pytest"]]
    assert tools.argfiles == [["a_test.py"]]


def test_pytest_without_a_pair_runs_nothing(tmp_path: Path) -> None:
    tools = _Recorder()

    assert _pytest(tmp_path, tools, ["a.py"]) == 0
    assert [call[1:3] for call in tools.calls] == [["-m", "coverage"]]


@pytest.mark.parametrize(("status", "expected"), [(1, 1), (5, 0), (2, 2)])
def test_pytest_passes_an_empty_collection_and_fails_a_failure(
    tmp_path: Path,
    status: int,
    expected: int,
) -> None:
    """Exit 5 is a selection whose every test carries a deselected marker."""
    tools = _Recorder(statuses=[0, status])

    assert _pytest(tmp_path, tools, ["a.py"], existing=["a_test.py"]) == expected


def test_a_failed_erase_fails_before_any_test_runs(tmp_path: Path) -> None:
    tools = _Recorder(status=19)

    assert _pytest(tmp_path, tools, ["a.py"], existing=["a_test.py"]) == 19
    assert len(tools.calls) == 1


@pytest.mark.parametrize(("executed", "expected"), [(2, 1), (4, 0)])
def test_pytest_fails_changed_sources_below_the_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    executed: int,
    expected: int,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[tool.coverage.report]\nfail_under = 100\n")
    (root / "a.py").write_text("".join(f"x{i} = {i}\n" for i in range(4)))
    data = coverage.CoverageData(basename=str(root / ".coverage"))
    data.add_lines({str(root / "a.py"): list(range(1, executed + 1))})
    data.write()
    monkeypatch.chdir(root)

    assert _pytest(tmp_path, _Recorder(), ["a.py", "a_test.py"]) == expected
    assert ("below fail_under=100" in capsys.readouterr().err) is bool(expected)


def test_pytest_without_coverage_data_skips_the_summary(tmp_path: Path) -> None:
    assert _pytest(tmp_path, _Recorder(), ["a.py", "a_test.py"]) == 0


def test_pytest_without_its_spool_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tools = _Recorder()

    assert precommit.pytest(_context(tmp_path, tools), dependents=None) == 1
    assert tools.calls == []
    assert "pytest-spool" in capsys.readouterr().err


@pytest.mark.parametrize("threshold", ["x", "0", "01"])
def test_a_junk_xdist_threshold_keeps_the_default(
    tmp_path: Path,
    threshold: str,
) -> None:
    tools = _Recorder(output="".join(f"t{i}_test.py::a\n" for i in range(15)))

    _pytest(
        tmp_path,
        tools,
        ["a.py"],
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_PYTEST="1",
        SIDECAR_XDIST_MIN=threshold,
    )

    assert "-n" not in tools.calls[-1]


@pytest.mark.parametrize(("count", "spread"), [(15, False), (16, True)])
def test_pytest_spreads_only_across_enough_test_files(
    tmp_path: Path,
    count: int,
    spread: bool,
) -> None:
    """Xdist's ~5s startup loses below 16 files; nodeids count by file."""
    tools = _Recorder(
        output="".join(f"t{i}_test.py::a\nt{i}_test.py::b\n" for i in range(count)),
    )

    _pytest(
        tmp_path,
        tools,
        ["a.py"],
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_PYTEST="1",
    )

    assert ("-n" in tools.calls[-1]) is spread


def test_pytest_prefers_the_graphs_tests_to_the_pairs(tmp_path: Path) -> None:
    tools = _Recorder(output="b_test.py::t\n", installed=False)

    _pytest(
        tmp_path,
        tools,
        ["a.py"],
        existing=["a_test.py"],
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_PYTEST="1",
    )

    walk, _ = tools.calls
    assert walk[3:8] == ["--emit", "nodeids", "--depth", "1", "--workers"]
    assert tools.argfiles == [["b_test.py::t"]]


@pytest.mark.parametrize("output", [None, ""])
def test_pytest_falls_back_to_the_pairs_when_the_graph_names_nothing(
    tmp_path: Path,
    output: str | None,
) -> None:
    """A stale index once ran zero tests and passed."""
    tools = _Recorder(output=output, installed=False)

    _pytest(
        tmp_path,
        tools,
        ["a.py"],
        existing=["a_test.py"],
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_PYTEST="1",
    )

    assert tools.argfiles == [["a_test.py"]]


def test_pytest_asks_for_missed_lines_on_request(tmp_path: Path) -> None:
    tools = _Recorder()

    _pytest(
        tmp_path,
        tools,
        ["a.py"],
        existing=["a_test.py"],
        PRECOMMIT_COVERAGE_DETAIL="1",
    )

    assert "--cov-report=term-missing:skip-covered" in tools.argfiles[0]


def _run_gate(
    tmp_path: Path,
    tools: _Recorder,
    files: Sequence[str],
    *,
    when: str | None = None,
    **environ: str,
) -> int:
    context = _context(tmp_path, tools, **environ)
    spool.append(context.directory, gate="leaks", key=9, paths=files)
    return precommit.run(
        "leaks",
        context=context,
        when=when,
        command=["pkg.leaks", "--strict"],
    )


def test_run_hands_a_module_the_spool_at_its_worker_count(tmp_path: Path) -> None:
    tools = _Recorder(status=3)

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
    tools = _Recorder()

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
    tools = _Recorder()

    _run_gate(tmp_path, tools, ["a.py"], when="OPT_IN", OPT_IN=value)

    assert bool(tools.calls) is runs
    assert spool.drain(tmp_path / "spool", gate="leaks", key=9) is None


def test_run_without_its_spool_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tools = _Recorder()

    assert (
        precommit.run(
            "leaks",
            context=_context(tmp_path, tools),
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
    flags = precommit._parse_args(argparse.ArgumentParser(), argv)

    assert (flags.gate, flags.when, flags.argv) == (spool_name, when, module)


def test_run_without_a_module_is_a_usage_error() -> None:
    with pytest.raises(SystemExit, match="2"):
        precommit._parse_args(argparse.ArgumentParser(), ["run", "leaks", "--"])


class _Processes:
    """Stands in for ``subprocess.run``: records argvs, answers ``status``/``stdout``."""

    def __init__(self, status: int, stdout: str) -> None:
        self.status = status
        self.stdout = stdout
        self.calls: list[list[str]] = []
        self.niceness: list[int] = []

    def nice(self, increment: int) -> int:
        self.niceness.append(increment)
        return 0

    def __call__(
        self,
        argv: Sequence[str],
        **_: object,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, self.status, stdout=self.stdout)


@pytest.fixture
def processes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Processes:
    """Run ``main`` in ``tmp_path`` with every child process recorded."""
    fake = _Processes(status=4, stdout="")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("WORKERS_DEFAULT", "1")
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(os, "nice", fake.nice)
    monkeypatch.setattr(subprocess, "run", fake)
    return fake


def _on_path(name: str) -> str:
    return f"/bin/{name}"


@pytest.mark.parametrize(
    ("spooled", "gate", "expected"),
    [
        (
            "spooled",
            ["run", "spooled", "--", "pkg.tool", "--flag"],
            ["-m", "pkg.tool", "--flag", "--workers", "1", "--", "a.py", "b.py"],
        ),
        ("ty-check", ["ty-check"], ["check", "--force-exclude", "a.py", "b.py"]),
        (
            "pyrefly-check",
            ["pyrefly-check", "--fixer", "pkg.fixer"],
            ["-m", "pkg.fixer", "--checker", "pyrefly", "--fix", "--threads", "1"],
        ),
    ],
)
def test_the_cli_spools_then_runs_one_gate(
    processes: _Processes,
    monkeypatch: pytest.MonkeyPatch,
    spooled: str,
    gate: list[str],
    expected: list[str],
) -> None:
    """Outside pre-commit every call shares key 1, so the spool meets its gate."""
    for name in ("a.py", "b.py"):
        monkeypatch.setattr(sys, "argv", ["precommit", "spool", spooled, name])
        assert precommit.main() == 0
    monkeypatch.setattr(sys, "argv", ["precommit", *gate])

    assert precommit.main() == 4
    assert processes.calls[-1][1 : 1 + len(expected)] == expected


def test_the_cli_runs_pytest(
    processes: _Processes,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "a_test.py").write_text("")
    monkeypatch.setattr(sys, "argv", ["precommit", "spool", "pytest", "a.py"])
    precommit.main()
    monkeypatch.setattr(sys, "argv", ["precommit", "pytest"])

    assert precommit.main() == 4
    assert processes.calls[-1][1:4] == ["-m", "coverage", "erase"]


def test_the_cli_runs_a_tier(
    processes: _Processes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_FORCE_ALL", "1")
    monkeypatch.setattr(sys, "argv", ["precommit", "tier", "pytest-ci-smoke-global"])

    assert precommit.main() == 4
    assert processes.calls[-1][-2:] == ["-m", "ci_smoke"]


def test_the_cli_runs_a_threaded_tool(
    processes: _Processes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["precommit", "threaded", "tool", "--", "tool", "--threads={workers}"],
    )

    assert precommit.main() == 4
    assert processes.calls[-1] == ["tool", "--threads=1"]


def test_every_command_runs_below_interactive_priority(
    processes: _Processes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["precommit", "spool", "x", "a.py"])

    assert precommit.main() == 0
    assert processes.niceness == [19]


@pytest.mark.parametrize(("status", "expected"), [(0, "a\n"), (1, None)])
def test_output_is_stdout_only_on_success(
    processes: _Processes,
    status: int,
    expected: str | None,
) -> None:
    processes.status, processes.stdout = status, "a\n"

    assert precommit._output(["tool"]) == expected
    assert precommit._run(["tool"]) == status


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [("  77 /usr/bin/pre-commit run\n", (77, "/usr/bin/pre-commit run")), ("", None)],
)
def test_a_process_is_its_parent_and_command(
    processes: _Processes,
    stdout: str,
    expected: tuple[int, str] | None,
) -> None:
    processes.stdout = stdout

    assert precommit._process(5) == expected


def test_the_gates_import_without_coverage_installed(tmp_path: Path) -> None:
    """An export without pytest-cov still type-checks and spools; madcatter has none."""
    blocker = tmp_path / "coverage.py"
    blocker.write_text("raise ModuleNotFoundError(\"No module named 'coverage'\")\n")
    done = subprocess.run(  # noqa: S603 -- this interpreter; fixed argv.
        [sys.executable, "-c", f"import {precommit.__name__}"],
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])},
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0, done.stderr


def test_installed_finds_importable_modules() -> None:
    assert precommit._installed("argparse")
    assert not precommit._installed("no_such_module_anywhere")
    assert not precommit._installed("no_such_module_anywhere.child")


def test_a_missing_tool_is_exit_127_not_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A commit must not die on a tool a machine lacks, as ``npm`` once did."""
    assert precommit._run(["no-such-tool-anywhere"]) == 127
    assert precommit._output(["no-such-tool-anywhere"]) is None
    assert "no-such-tool-anywhere" in capsys.readouterr().err


def test_a_tool_beside_the_interpreter_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "ty").write_text("")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))

    assert precommit._tool("ty") == str(tmp_path / "ty")
    assert precommit._tool("no-such-tool-anywhere") == "no-such-tool-anywhere"


class _Tiers(_Recorder):
    """A recorder whose ``has`` answers per module."""

    def __init__(
        self,
        *,
        missing: frozenset[str] = frozenset(),
        on_path: bool = True,
    ) -> None:
        super().__init__(on_path=on_path)
        self.missing = missing

    @override
    def has(self, module: str) -> bool:
        return module not in self.missing


_PLUGIN = "pkg.nodeids"


def _tier(
    tmp_path: Path,
    name: str,
    tools: _Recorder,
    **environ: str,
) -> list[str]:
    """Run tier ``name``; return the pytest argv after ``-m pytest -n=N``."""
    precommit.tier(
        TIERS[name],
        context=_context(tmp_path, tools, **environ),
        plugin=_PLUGIN,
    )
    (argv,) = tools.calls
    assert argv[:3] == ["env", "PYTHONUNBUFFERED=1", "COVERAGE_CORE=ctrace"]
    return argv[argv.index("pytest") + 2 :]


@pytest.fixture(params=sorted(TIERS))
def name(request: pytest.FixtureRequest) -> str:
    """Each tier."""
    return str(request.param)


def test_an_unsharded_local_tier_selects_with_testmon(
    tmp_path: Path,
    name: str,
) -> None:
    """Coverage is how testmon sees which tests touched changed code: keep it."""
    flags = _tier(tmp_path, name, _Tiers())

    assert "--testmon-forceselect" in flags
    assert "--no-cov" not in flags


@pytest.mark.parametrize(
    "environ",
    [{"PYTEST_FORCE_ALL": "1"}, {"PYTEST_SPLIT_ARGS": "--splits 4 --group 1"}],
    ids=["force-all", "sharded"],
)
def test_a_forced_or_sharded_tier_runs_everything(
    tmp_path: Path,
    name: str,
    environ: dict[str, str],
) -> None:
    """Testmon and pytest-split partition one collection; a shard ran 8 of 118."""
    flags = _tier(tmp_path, name, _Tiers(), **environ)

    split = environ.get("PYTEST_SPLIT_ARGS", "").split()
    assert "--testmon-forceselect" not in flags
    assert "--no-cov" in flags
    assert flags[len(flags) - len(split) :] == split


def test_only_the_ci_testmon_tier_selects_in_github_actions(
    tmp_path: Path,
    name: str,
) -> None:
    flags = _tier(tmp_path, name, _Tiers(), GITHUB_ACTIONS="true")

    assert ("--testmon-forceselect" in flags) is TIERS[name].ci_testmon


def test_pr_preselection_collects_with_testmon_and_the_plugin(
    tmp_path: Path,
    name: str,
) -> None:
    flags = _tier(
        tmp_path,
        name,
        _Tiers(),
        GITHUB_ACTIONS="true",
        PYTEST_COLLECT_SELECTED="selected.txt",
    )

    assert "--testmon-forceselect" in flags
    assert "--collect-only" in flags
    assert flags[flags.index("-p") + 1] == _PLUGIN


def test_a_tree_without_the_plugins_passes_none_of_their_flags(
    tmp_path: Path,
    name: str,
) -> None:
    """Every export runs these tiers; a missing plugin's flag is a hard error."""
    tools = _Tiers(missing=frozenset({"testmon", "pytest_cov", _PLUGIN}))

    flags = _tier(tmp_path, name, tools, PYTEST_COLLECT_SELECTED="selected.txt")

    assert not {"--testmon-forceselect", "--no-cov", "-p"} & set(flags)


def test_selected_nodeids_run_without_testmon(tmp_path: Path, name: str) -> None:
    listing = tmp_path / "nodeids.txt"
    listing.write_text("== chrome ==\na_test.py::test_one\n")

    flags = _tier(
        tmp_path,
        name,
        _Tiers(),
        PYTEST_FORCE_ALL="1",
        PYTEST_NODEIDS_FILE=str(listing),
    )

    assert flags[-1] == "a_test.py::test_one"
    assert "== chrome ==" not in flags


def test_a_selection_of_nothing_runs_nothing(tmp_path: Path, name: str) -> None:
    """``--collect-only`` banners are never node ids; running them collects zero."""
    listing = tmp_path / "nodeids.txt"
    listing.write_text("0/33757 tests collected (33757 deselected) in 47.02s\n")
    tools = _Tiers()

    assert (
        precommit.tier(
            TIERS[name],
            context=_context(tmp_path, tools, PYTEST_NODEIDS_FILE=str(listing)),
            plugin=None,
        )
        == 0
    )
    assert tools.calls == []


def test_progress_and_testmon_args_reach_pytest(tmp_path: Path, name: str) -> None:
    flags = _tier(
        tmp_path,
        name,
        _Tiers(),
        PYTEST_PROGRESS_ARGS="-p x --loop-ci-progress",
        PYTEST_TESTMON_ARGS="--testmon-noselect",
        PYTEST_SPLIT_ARGS="--group 2",
    )

    assert "--testmon-noselect" in flags
    assert "--no-cov" not in flags
    assert flags[flags.index("--loop-ci-progress") - 2 :][:2] == ["-p", "x"]


def test_each_tier_passes_its_marker(tmp_path: Path, name: str) -> None:
    flags = _tier(tmp_path, name, _Tiers())

    marker = TIERS[name].marker
    assert (flags[flags.index("-m") + 1] if "-m" in flags else "") == marker


def test_a_coverage_report_measures_the_project_in_place_of_no_cov(
    tmp_path: Path,
) -> None:
    flags = _tier(
        tmp_path,
        "pytest-global",
        _Tiers(),
        PYTEST_FORCE_ALL="1",
        PYTEST_COVERAGE_REPORT="1",
    )

    assert "--no-cov" not in flags
    assert flags[:2] == ["--cov", "--cov-report=term:skip-covered"]


@pytest.mark.parametrize(("status", "allowed"), [(5, True), (1, False), (0, True)])
def test_only_declared_tiers_pass_an_empty_collection(
    tmp_path: Path,
    name: str,
    status: int,
    allowed: bool,
) -> None:
    """The whole suite collecting nothing is a broken tree, not an empty tier."""
    tools = _Tiers()
    tools.status = status

    code = precommit.tier(TIERS[name], context=_context(tmp_path, tools), plugin=None)

    assert (code == 0) is (allowed and (status == 0 or TIERS[name].allow_empty))


@pytest.mark.parametrize("status", [0, 5, 2])
def test_a_collection_counts_an_empty_one_as_success(
    tmp_path: Path,
    name: str,
    status: int,
) -> None:
    tools = _Tiers()
    tools.status = status

    code = precommit.tier(
        TIERS[name],
        context=_context(tmp_path, tools, PYTEST_COLLECT_SELECTED="s.txt"),
        plugin=None,
    )

    assert code == (2 if status == 2 else 0)


def test_a_tier_without_its_capability_runs_nothing(tmp_path: Path) -> None:
    tools = _Tiers(on_path=False)

    assert (
        precommit.tier(
            TIERS["pytest-integration-global"],
            context=_context(tmp_path, tools),
            plugin=None,
        )
        == 0
    )
    assert tools.calls == []


def test_threaded_fills_in_the_gate_worker_count(tmp_path: Path) -> None:
    tools = _Recorder()

    precommit.threaded(
        "pyrefly-check-global",
        context=_context(tmp_path, tools, WORKERS_PYREFLY_CHECK_GLOBAL="2"),
        command=["pyrefly", "check", "--threads={workers}"],
    )

    assert tools.calls == [["pyrefly", "check", "--threads=2"]]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
