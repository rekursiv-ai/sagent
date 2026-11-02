"""Tests for the pytest gates."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import os
import subprocess
import sys

import coverage
import pytest

from tools.gate import fakes, gate, spool
from tools.testing import coverage_summary, testing
from tools.testing.tiers import TIERS
from tools.typecheck import typecheck


if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType


@pytest.fixture
def processes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> fakes.Processes:
    """Record what ``main`` runs, under no gate's settings."""
    return fakes.isolate(tmp_path, monkeypatch=monkeypatch)


@pytest.mark.parametrize("module", [gate, typecheck, testing])
def test_every_gate_imports_without_coverage_installed(
    tmp_path: Path,
    module: ModuleType,
) -> None:
    """An export without pytest-cov still type-checks and spools; madcatter has none.

    ``testing`` is the one that reaches ``coverage``, through ``coverage_summary``.
    """
    blocker = tmp_path / "coverage.py"
    blocker.write_text("raise ModuleNotFoundError(\"No module named 'coverage'\")\n")
    done = subprocess.run(  # noqa: S603 -- this interpreter; fixed argv.
        [sys.executable, "-c", f"import {module.__name__}"],
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])},
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0, done.stderr


def test_one_gates_depth_never_widens_the_other(tmp_path: Path) -> None:
    """Each gate reads its own variable; the other's is not an opt-in."""
    fakes.touch(tmp_path, ["a.py"])
    spool.append(tmp_path / "spool", gate="ty-check", key=9, paths=["a.py"])
    checker = fakes.Recorder(output="b.py\n")
    typecheck.typecheck(
        "ty-check",
        context=fakes.context(tmp_path, checker, PRECOMMIT_CODEGRAPH_PYTEST="3"),
        dependents="pkg.affected",
        fixer=None,
    )
    tests = fakes.Recorder(output="b_test.py::t\n", installed=False)
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


def _pytest(
    tmp_path: Path,
    tools: fakes.Recorder,
    files: Sequence[str],
    *,
    existing: Sequence[str] = (),
    dependents: str | None = None,
    **environ: str,
) -> int:
    context = fakes.context(tmp_path, tools, **environ)
    for name in (*files, *existing):
        (context.root / name).parent.mkdir(parents=True, exist_ok=True)
        (context.root / name).touch()
    spool.append(context.directory, gate="pytest", key=9, paths=files)
    return testing.pytest(context, dependents=dependents)


def test_pytest_runs_each_changed_files_pairs_with_its_directory_measured(
    tmp_path: Path,
) -> None:
    tools = fakes.Recorder()

    _pytest(
        tmp_path,
        tools,
        ["pkg/a.py", "pkg/b_test.py", "top.py"],
        existing=["pkg/a_test.py", "pkg/tests/test_a.py", "pkg/b.py"],
    )

    [test] = tools.calls
    assert test[3:6] == ["-m", "pytest", "-q"]
    assert tools.argfiles == [
        [
            "--cov=.",
            "--cov=pkg",
            "--no-cov-on-fail",
            "--cov-fail-under=0",
            "--cov-report=",
            "pkg/a_test.py",
            "pkg/b_test.py",
            "pkg/tests/test_a.py",
        ],
    ]


def test_pytest_without_pytest_cov_measures_nothing(tmp_path: Path) -> None:
    tools = fakes.Recorder(installed=False)

    _pytest(tmp_path, tools, ["a.py"], existing=["a_test.py"])

    assert [call[3:5] for call in tools.calls] == [["-m", "pytest"]]
    assert tools.argfiles == [["a_test.py"]]


def test_pytest_without_a_pair_runs_nothing(tmp_path: Path) -> None:
    tools = fakes.Recorder()

    assert _pytest(tmp_path, tools, ["a.py"]) == 0
    assert tools.calls == []


@pytest.mark.parametrize(("status", "expected"), [(1, 1), (5, 0), (2, 2)])
def test_pytest_passes_an_empty_collection_and_fails_a_failure(
    tmp_path: Path,
    status: int,
    expected: int,
) -> None:
    """Exit 5 is a selection whose every test carries a deselected marker."""
    tools = fakes.Recorder(status=status)

    assert _pytest(tmp_path, tools, ["a.py"], existing=["a_test.py"]) == expected


def test_a_concurrent_runs_coverage_data_survives_the_gate(tmp_path: Path) -> None:
    """``coverage erase`` here once deleted a live ``pytest -n24 --cov``'s workers."""
    tools = fakes.Recorder()
    live = tmp_path / "repo" / ".coverage.host.pid7.x"
    live.parent.mkdir()
    live.write_text("in use")

    _pytest(tmp_path, tools, ["a.py"], existing=["a_test.py"])

    [test] = tools.calls
    data = Path(test[1].removeprefix("COVERAGE_FILE="))
    assert test[:2] == ["env", f"COVERAGE_FILE={data}"]
    assert not data.is_relative_to(tmp_path / "repo")
    assert live.read_text() == "in use"


class _Measured(fakes.Recorder):
    """Writes ``lines`` of ``a.py`` to the data file pytest is pointed at."""

    def __init__(self, root: Path, *, executed: int) -> None:
        super().__init__()
        self.root = root
        self.executed = executed

    @override
    def run(self, argv: Sequence[str]) -> int:
        data = coverage.CoverageData(
            basename=argv[1].removeprefix("COVERAGE_FILE="),
        )
        data.add_lines({str(self.root / "a.py"): list(range(1, self.executed + 1))})
        data.write()
        return super().run(argv)


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
    monkeypatch.chdir(root)
    tools = _Measured(root, executed=executed)

    assert _pytest(tmp_path, tools, ["a.py", "a_test.py"]) == expected
    assert ("below fail_under=100" in capsys.readouterr().err) is bool(expected)


def test_pytest_without_coverage_data_skips_the_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No data file is a run that measured nothing, never a 0% shortfall."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[tool.coverage.report]\nfail_under = 100\n")
    (root / "a.py").write_text("x = 1\n")
    monkeypatch.chdir(root)

    assert _pytest(tmp_path, fakes.Recorder(), ["a.py", "a_test.py"]) == 0


def test_pytest_measures_only_with_pytest_cov_installed(tmp_path: Path) -> None:
    tools = _Tiers(missing=frozenset({"pytest_cov"}))

    _pytest(tmp_path, tools, ["a.py"], existing=["a_test.py"])

    assert tools.argfiles == [["a_test.py"]]


def test_pytest_without_its_spool_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tools = fakes.Recorder()

    assert testing.pytest(fakes.context(tmp_path, tools), dependents=None) == 1
    assert tools.calls == []
    assert "pytest-spool" in capsys.readouterr().err


@pytest.mark.parametrize("threshold", ["x", "0", "01"])
def test_a_junk_xdist_threshold_keeps_the_default(
    tmp_path: Path,
    threshold: str,
) -> None:
    tools = fakes.Recorder(output="".join(f"t{i}_test.py::a\n" for i in range(15)))

    _pytest(
        tmp_path,
        tools,
        ["a.py"],
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_PYTEST="1",
        SIDECAR_XDIST_MIN=threshold,
    )

    assert "-n" not in tools.calls[-1]


def test_the_xdist_threshold_can_be_lowered(tmp_path: Path) -> None:
    tools = fakes.Recorder(output="t0_test.py::a\nt1_test.py::a\n")

    _pytest(
        tmp_path,
        tools,
        ["a.py"],
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_PYTEST="1",
        SIDECAR_XDIST_MIN="2",
    )

    assert "-n" in tools.calls[-1]


@pytest.mark.parametrize(("count", "spread"), [(15, False), (16, True)])
def test_pytest_spreads_only_across_enough_test_files(
    tmp_path: Path,
    count: int,
    spread: bool,
) -> None:
    """Xdist's ~5s startup loses below 16 files; nodeids count by file."""
    tools = fakes.Recorder(
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


def test_pytest_reads_its_selection_from_an_argfile(tmp_path: Path) -> None:
    tools = fakes.Recorder(
        output="".join(f"t{i}_test.py::a\n" for i in range(16)),
        installed=False,
    )

    _pytest(
        tmp_path,
        tools,
        ["a.py"],
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_PYTEST="1",
    )

    *head, argfile = tools.calls[-1]
    assert head[3:] == ["-m", "pytest", "-n", "3", "-q", "--tb=short"]
    path = Path(argfile.removeprefix("@"))
    assert path.name == "pytest.args"
    assert path.parent.name.startswith("precommit-")


def test_an_unmeasured_run_ignores_a_stale_coverage_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without pytest-cov a leftover ``.coverage`` describes some other run."""
    read: list[object] = []

    def shortfall(sources: object, *, data: Path) -> list[str]:
        read.append((sources, data))
        return []

    monkeypatch.setattr(coverage_summary, "shortfall", shortfall)
    tools = fakes.Recorder(installed=False)
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / ".coverage").write_text("")

    assert _pytest(tmp_path, tools, ["a.py"], existing=["a_test.py"]) == 0
    assert read == []


def test_the_gate_sizes_the_walk_and_pytest_from_its_own_workers(
    tmp_path: Path,
) -> None:
    tools = fakes.Recorder(
        output="".join(f"t{i}_test.py::a\n" for i in range(16)),
        installed=False,
    )

    _pytest(
        tmp_path,
        tools,
        ["a.py"],
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_PYTEST="1",
        WORKERS_PYTEST="5",
    )

    [walk] = [call for call in tools.calls if "--emit" in call]
    assert walk[walk.index("--workers") + 1] == "5"
    test = tools.calls[-1]
    assert test[test.index("-n") + 1] == "5"


def test_pytest_prefers_the_graphs_tests_to_the_pairs(tmp_path: Path) -> None:
    tools = fakes.Recorder(output="b_test.py::t\n", installed=False)

    _pytest(
        tmp_path,
        tools,
        ["a.py"],
        existing=["a_test.py"],
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_PYTEST="1",
    )

    [walk] = [call for call in tools.calls if "--emit" in call]
    assert walk[3:8] == ["--emit", "nodeids", "--depth", "1", "--workers"]
    assert tools.argfiles == [["b_test.py::t"]]


@pytest.mark.parametrize("output", [None, ""])
def test_pytest_falls_back_to_the_pairs_when_the_graph_names_nothing(
    tmp_path: Path,
    output: str | None,
) -> None:
    """A stale index once ran zero tests and passed."""
    tools = fakes.Recorder(output=output, installed=False)

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
    tools = fakes.Recorder()

    _pytest(
        tmp_path,
        tools,
        ["a.py"],
        existing=["a_test.py"],
        PRECOMMIT_COVERAGE_DETAIL="1",
    )

    assert "--cov-report=term-missing:skip-covered" in tools.argfiles[0]


def test_the_cli_runs_pytest(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in ("a.py", "a_test.py"):
        (tmp_path / name).write_text("")
    monkeypatch.setattr(sys, "argv", ["gate", "spool", "pytest", "a.py"])
    gate.main()
    monkeypatch.setattr(sys, "argv", ["testing", "pytest"])

    assert testing.main() == 4
    assert processes.calls[-1][3:5] == ["-m", "pytest"]


def test_the_cli_runs_a_tier(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_FORCE_ALL", "1")
    monkeypatch.setattr(sys, "argv", ["testing", "tier", "pytest-ci-smoke-global"])

    assert testing.main() == 4
    assert processes.calls[-1][-2:] == ["-m", "ci_smoke"]


class _Tiers(fakes.Recorder):
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
    tools: fakes.Recorder,
    **environ: str,
) -> list[str]:
    """Run tier ``name``; return the pytest argv after ``-m pytest -n=N``."""
    testing.tier(
        TIERS[name],
        context=fakes.context(tmp_path, tools, **environ),
        plugin=_PLUGIN,
    )
    (argv,) = tools.calls
    assert argv[:3] == ["env", "PYTHONUNBUFFERED=1", "COVERAGE_CORE=ctrace"]
    return argv[argv.index("pytest") + 2 :]


@pytest.fixture(params=sorted(TIERS))
def name(request: pytest.FixtureRequest) -> str:
    """Each tier."""
    return cast(str, request.param)


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
        testing.tier(
            TIERS[name],
            context=fakes.context(tmp_path, tools, PYTEST_NODEIDS_FILE=str(listing)),
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

    code = testing.tier(
        TIERS[name],
        context=fakes.context(tmp_path, tools),
        plugin=None,
    )

    assert (code == 0) is (allowed and (status == 0 or TIERS[name].allow_empty))


@pytest.mark.parametrize("status", [0, 5, 2])
def test_a_collection_counts_an_empty_one_as_success(
    tmp_path: Path,
    name: str,
    status: int,
) -> None:
    tools = _Tiers()
    tools.status = status

    code = testing.tier(
        TIERS[name],
        context=fakes.context(tmp_path, tools, PYTEST_COLLECT_SELECTED="s.txt"),
        plugin=None,
    )

    assert code == (2 if status == 2 else 0)


def test_a_tier_without_its_capability_runs_nothing(tmp_path: Path) -> None:
    tools = _Tiers(on_path=False)

    assert (
        testing.tier(
            TIERS["pytest-integration-global"],
            context=fakes.context(tmp_path, tools),
            plugin=None,
        )
        == 0
    )
    assert tools.calls == []


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
