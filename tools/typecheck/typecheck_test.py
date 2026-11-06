"""Tests for the type-checker gates."""

from __future__ import annotations

from pathlib import Path

import argparse
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
    fakes.touch(tmp_path, paths)
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
@pytest.mark.parametrize(
    ("value", "enabled"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("01", False),
        ("1", True),
        ("2", True),
    ],
)
def test_the_fixer_runs_the_checker_in_its_place(
    tmp_path: Path,
    name: str,
    checker: str,
    value: str | None,
    *,
    enabled: bool,
) -> None:
    _spool(tmp_path, "a.py")
    tools = fakes.Recorder()

    _typecheck(
        name,
        tmp_path,
        tools,
        fixer="pkg.fixer",
        **({"PRECOMMIT_BASEDPYRIGHT": value} if value is not None else {}),
    )

    [[_, *head, named]] = tools.calls
    assert head == [
        "-m",
        "pkg.fixer",
        "--checker",
        checker,
        "--fix",
        "--threads",
        "3",
        "--typechecker",
        "pyrefly",
        *(["basedpyright"] if enabled or name == "basedpyright" else []),
        "--",
    ]
    assert named.startswith("@")
    assert tools.argfiles == [["a.py"]]


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
    assert walk[1:-1] == [
        "-m",
        "pkg.affected",
        "--emit",
        "files",
        "--depth",
        "3",
        "--workers",
        "3",
        "--",
    ]
    assert tools.argfiles == [["a.py"]]
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
        Path(name).touch()
        monkeypatch.setattr(sys, "argv", ["gate", "spool", checker, name])
        assert gate.main() == 0
    monkeypatch.setattr(sys, "argv", ["typecheck", *argv])

    assert typecheck.main() == 4
    assert processes.calls[-1][1 : 1 + len(expected)] == expected


def test_the_cli_widens_through_its_dependents_module(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Path("a.py").touch()
    monkeypatch.setattr(sys, "argv", ["gate", "spool", "ty-check", "a.py"])
    assert gate.main() == 0
    monkeypatch.setenv("PRECOMMIT_CODEGRAPH_TYPECHECK", "2")
    monkeypatch.setattr(
        sys,
        "argv",
        ["typecheck", "ty-check", "--dependents", "pkg.affected"],
    )

    typecheck.main()

    walks = [call for call in processes.calls if "--emit" in call]
    assert [call[1:6] for call in walks] == [
        ["-m", "pkg.affected", "--emit", "files", "--depth"],
    ]


@pytest.mark.parametrize("name", ["ty-check", "basedpyright"])
def test_only_pyrefly_takes_what_ty_checked(tmp_path: Path, name: str) -> None:
    """A hand-off left from another run must not replace the other checkers' spool."""
    fakes.touch(tmp_path, ["b.py"])
    spool.write(tmp_path / "spool", gate="typecheck-checked", key=9, paths=["b.py"])
    _spool(tmp_path, "a.py")
    tools = fakes.Recorder()

    assert _typecheck(name, tmp_path, tools) == 0
    assert [call[-1] for call in tools.calls] == ["a.py"]


@pytest.mark.parametrize("name", ["ty-check", "pyrefly-check", "basedpyright"])
def test_each_checker_is_a_command_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    # The flags come from the argv given, not the process's.
    monkeypatch.setattr(sys, "argv", ["typecheck", "--bogus"])

    flags = typecheck._parse_args(argparse.ArgumentParser(), [name])

    assert (flags.command, flags.dependents, flags.fixer) == (name, None, None)
    assert flags.handler is typecheck._check


@pytest.mark.parametrize("value", [None, "", "0", "false", "01", "1", "2"])
def test_cli_environment_gate(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    Path("a.py").touch()
    monkeypatch.setattr(sys, "argv", ["gate", "spool", "basedpyright", "a.py"])
    assert gate.main() == 0
    processes.calls.clear()
    if value is not None:
        monkeypatch.setenv("PRECOMMIT_BASEDPYRIGHT", value)
    monkeypatch.setattr(
        sys,
        "argv",
        ["typecheck", "basedpyright", "--when", "PRECOMMIT_BASEDPYRIGHT"],
    )

    assert typecheck.main() == (4 if value in {"1", "2"} else 0)
    checks = [call for call in processes.calls if "--threads=1" in call]
    assert len(checks) == (1 if value in {"1", "2"} else 0)
    assert list(Path().glob("precommit-spool-*/basedpyright.*")) == []


def test_a_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        typecheck._parse_args(argparse.ArgumentParser(), [])

    assert exited.value.code == 2
    assert "required: command" in capsys.readouterr().err


def test_ty_check_has_no_fixer() -> None:
    with pytest.raises(SystemExit) as exited:
        typecheck._parse_args(
            argparse.ArgumentParser(),
            ["ty-check", "--fixer", "pkg.fixer"],
        )

    assert exited.value.code == 2


def test_the_help_names_each_checker_and_the_fixer_module(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        typecheck._parse_args(argparse.ArgumentParser(), ["--help"])
    top = " ".join(capsys.readouterr().out.split())
    with pytest.raises(SystemExit):
        typecheck._parse_args(argparse.ArgumentParser(), ["pyrefly-check", "--help"])
    pyrefly = capsys.readouterr().out

    for name in ("ty-check", "pyrefly-check", "basedpyright"):
        assert f"{name} Run {name} over its spool." in top
    assert "--fixer MODULE" in pyrefly
    assert "--when VARIABLE" in pyrefly


@pytest.mark.parametrize(
    ("docstring", "after_usage"),
    [
        (typecheck.__doc__, "Run one type checker once"),
        (None, "positional arguments:"),
    ],
    ids=["docstring", "optimized away"],
)
def test_main_help_describes_the_gate_from_its_docstring(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    docstring: str | None,
    after_usage: str,
) -> None:
    """``python -OO`` strips the docstring; the help then has no description."""
    monkeypatch.setattr(typecheck, "__doc__", docstring)
    monkeypatch.setattr(sys, "argv", ["typecheck", "--help"])

    with pytest.raises(SystemExit):
        typecheck.main()

    assert capsys.readouterr().out.split("\n\n")[1].startswith(after_usage)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
