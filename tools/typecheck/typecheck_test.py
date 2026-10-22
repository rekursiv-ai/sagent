"""Tests for the type-checker gates."""

from __future__ import annotations

from pathlib import Path

import sys

import pytest

from tools.gate import fakes, gate, spool
from tools.typecheck import typecheck


@pytest.fixture
def processes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> fakes.Processes:
    """Record what ``main`` runs, under no gate's settings."""
    return fakes.isolate(tmp_path, monkeypatch=monkeypatch)


def _typecheck(
    name: str,
    tmp_path: Path,
    tools: fakes.Recorder,
    *,
    dependents: str | None = None,
    fixer: str | None = None,
    **environ: str,
) -> int:
    return typecheck.typecheck(
        name,
        context=fakes.context(tmp_path, tools, **environ),
        dependents=dependents,
        fixer=fixer,
    )


def _spool(tmp_path: Path, *paths: str) -> None:
    """Spool ``paths`` as one pre-commit run does: to every checker's twin."""
    for name in ("ty-check", "pyrefly-check", "basedpyright"):
        spool.append(tmp_path / "spool", gate=name, key=9, paths=paths)


def test_ty_checks_every_spooled_file_once_minus_excludes(tmp_path: Path) -> None:
    _spool(tmp_path, "a.py")
    _spool(tmp_path, "x/third_party/v.py", "b.py", "a.py")
    tools = fakes.Recorder()

    assert _typecheck("ty-check", tmp_path, tools) == 0
    assert [[Path(call[0]).name, *call[1:]] for call in tools.calls] == [
        ["ty", "check", "--force-exclude", "a.py", "b.py"],
    ]


def test_pyrefly_checks_what_ty_checked(tmp_path: Path) -> None:
    _spool(tmp_path, "a.py")
    _typecheck("ty-check", tmp_path, fakes.Recorder())
    tools = fakes.Recorder()

    assert _typecheck("pyrefly-check", tmp_path, tools) == 0
    assert [[Path(call[0]).name, *call[1:]] for call in tools.calls] == [
        ["pyrefly", "check", "--threads=3", "a.py"],
    ]


def test_pyrefly_checks_the_spool_when_ty_was_skipped(tmp_path: Path) -> None:
    """``SKIP=ty-check`` leaves nothing checked; pyrefly must not pass on nothing.

    It widens on its own then, at its own worker count.
    """
    _spool(tmp_path, "a.py", "third_party/v.py")
    tools = fakes.Recorder(output="a.py\nb.py\nthird_party/v.py\n")

    status = _typecheck(
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
    _typecheck("ty-check", tmp_path, fakes.Recorder())
    _spool(tmp_path, "a.py")
    tools = fakes.Recorder()

    assert _typecheck("pyrefly-check", tmp_path, tools) == 0
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
    tools = fakes.Recorder()

    _typecheck(name, tmp_path, tools, fixer="pkg.fixer")

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
    tools = fakes.Recorder()

    assert _typecheck(name, tmp_path, tools) == 0
    assert tools.calls == []


@pytest.mark.parametrize("name", ["ty-check", "pyrefly-check", "basedpyright"])
def test_a_checker_without_its_spool_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    """``pre-commit run ty-check`` once ran without its twin and passed a type error."""
    tools = fakes.Recorder()

    assert _typecheck(name, tmp_path, tools) == 1
    assert tools.calls == []
    assert f"{name}-spool" in capsys.readouterr().err


def test_basedpyright_checks_its_spool_at_its_thread_count(tmp_path: Path) -> None:
    _spool(tmp_path, "a.py", "b.py")
    tools = fakes.Recorder()

    assert _typecheck("basedpyright", tmp_path, tools) == 0
    assert [[Path(call[0]).name, *call[1:]] for call in tools.calls] == [
        ["basedpyright", "--threads=3", "a.py", "b.py"],
    ]


@pytest.mark.parametrize("name", ["ty-check", "pyrefly-check", "basedpyright"])
def test_a_failing_tool_fails_the_typecheck(tmp_path: Path, name: str) -> None:
    _spool(tmp_path, "a.py")

    assert _typecheck(name, tmp_path, fakes.Recorder(status=1)) == 1


def test_dependents_widen_the_files_then_excludes_apply(tmp_path: Path) -> None:
    _spool(tmp_path, "a.py")
    tools = fakes.Recorder(output="a.py\nb.py\nthird_party/c.py\n")

    _typecheck(
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
    tools = fakes.Recorder(output=output)

    _typecheck(
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
    tools = fakes.Recorder(output="b.py\n")

    _typecheck(
        "ty-check",
        tmp_path,
        tools,
        dependents=dependents,
        PRECOMMIT_CODEGRAPH_TYPECHECK=value,
    )

    assert [call[-1] for call in tools.calls] == ["a.py"]


def test_a_tool_beside_the_interpreter_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "ty").write_text("")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))

    assert typecheck._tool("ty") == str(tmp_path / "ty")
    assert typecheck._tool("no-such-tool-anywhere") == "no-such-tool-anywhere"


@pytest.mark.parametrize(
    ("checker", "argv", "expected"),
    [
        ("ty-check", ["ty-check"], ["check", "--force-exclude", "a.py", "b.py"]),
        (
            "pyrefly-check",
            ["pyrefly-check", "--fixer", "pkg.fixer"],
            ["-m", "pkg.fixer", "--checker", "pyrefly", "--fix", "--threads", "1"],
        ),
    ],
)
def test_the_cli_checks_what_the_gate_spooled(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
    checker: str,
    argv: list[str],
    expected: list[str],
) -> None:
    """The spool command and the checker are separate CLIs sharing one spool."""
    for name in ("a.py", "b.py"):
        monkeypatch.setattr(sys, "argv", ["gate", "spool", checker, name])
        assert gate.main() == 0
    monkeypatch.setattr(sys, "argv", ["typecheck", *argv])

    assert typecheck.main() == 4
    assert processes.calls[-1][1 : 1 + len(expected)] == expected


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
