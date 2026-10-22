"""Tests for the pytest gates."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import sys

import coverage
import pytest

from tools import conftest
from tools.gate import gate, spool
from tools.testing import testing
from tools.testing.tiers import TIERS
from tools.typecheck import typecheck


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def test_one_gates_depth_never_widens_the_other(tmp_path: Path) -> None:
    """Each gate reads its own variable; the other's is not an opt-in."""
    spool.append(tmp_path / "spool", gate="ty-check", key=9, paths=["a.py"])
    checker = conftest.Recorder(output="b.py\n")
    typecheck.typecheck(
        "ty-check",
        context=conftest.context(tmp_path, checker, PRECOMMIT_CODEGRAPH_PYTEST="3"),
        dependents="pkg.affected",
        fixer=None,
    )
    tests = conftest.Recorder(output="b_test.py::t\n", installed=False)
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
    tools: conftest.Recorder,
    files: Sequence[str],
    *,
    existing: Sequence[str] = (),
    dependents: str | None = None,
    **environ: str,
) -> int:
    context = conftest.context(tmp_path, tools, **environ)
    for name in (*files, *existing):
        (context.root / name).parent.mkdir(parents=True, exist_ok=True)
        (context.root / name).touch()
    spool.append(context.directory, gate="pytest", key=9, paths=files)
    return testing.pytest(context, dependents=dependents)


def test_pytest_runs_each_changed_files_pairs_with_its_directory_measured(
    tmp_path: Path,
) -> None:
    tools = conftest.Recorder()

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
    tools = conftest.Recorder(installed=False)

    _pytest(tmp_path, tools, ["a.py"], existing=["a_test.py"])

    assert [call[1:3] for call in tools.calls] == [["-m", "pytest"]]
    assert tools.argfiles == [["a_test.py"]]


def test_pytest_without_a_pair_runs_nothing(tmp_path: Path) -> None:
    tools = conftest.Recorder()

    assert _pytest(tmp_path, tools, ["a.py"]) == 0
    assert [call[1:3] for call in tools.calls] == [["-m", "coverage"]]


@pytest.mark.parametrize(("status", "expected"), [(1, 1), (5, 0), (2, 2)])
def test_pytest_passes_an_empty_collection_and_fails_a_failure(
    tmp_path: Path,
    status: int,
    expected: int,
) -> None:
    """Exit 5 is a selection whose every test carries a deselected marker."""
    tools = conftest.Recorder(statuses=[0, status])

    assert _pytest(tmp_path, tools, ["a.py"], existing=["a_test.py"]) == expected


def test_a_failed_erase_fails_before_any_test_runs(tmp_path: Path) -> None:
    tools = conftest.Recorder(status=19)

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

    assert _pytest(tmp_path, conftest.Recorder(), ["a.py", "a_test.py"]) == expected
    assert ("below fail_under=100" in capsys.readouterr().err) is bool(expected)


def test_pytest_without_coverage_data_skips_the_summary(tmp_path: Path) -> None:
    assert _pytest(tmp_path, conftest.Recorder(), ["a.py", "a_test.py"]) == 0


def test_pytest_without_its_spool_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tools = conftest.Recorder()

    assert testing.pytest(conftest.context(tmp_path, tools), dependents=None) == 1
    assert tools.calls == []
    assert "pytest-spool" in capsys.readouterr().err


@pytest.mark.parametrize("threshold", ["x", "0", "01"])
def test_a_junk_xdist_threshold_keeps_the_default(
    tmp_path: Path,
    threshold: str,
) -> None:
    tools = conftest.Recorder(output="".join(f"t{i}_test.py::a\n" for i in range(15)))

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
    tools = conftest.Recorder(
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
    tools = conftest.Recorder(output="b_test.py::t\n", installed=False)

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
    tools = conftest.Recorder(output=output, installed=False)

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
    tools = conftest.Recorder()

    _pytest(
        tmp_path,
        tools,
        ["a.py"],
        existing=["a_test.py"],
        PRECOMMIT_COVERAGE_DETAIL="1",
    )

    assert "--cov-report=term-missing:skip-covered" in tools.argfiles[0]


def test_the_cli_runs_pytest(
    processes: conftest.Processes,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "a_test.py").write_text("")
    monkeypatch.setattr(sys, "argv", ["gate", "spool", "pytest", "a.py"])
    gate.main()
    monkeypatch.setattr(sys, "argv", ["testing", "pytest"])

    assert testing.main() == 4
    assert processes.calls[-1][1:4] == ["-m", "coverage", "erase"]


def test_the_cli_runs_a_tier(
    processes: conftest.Processes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CI runs this suite inside a shard; its ``PYTEST_*`` must not reach the tier."""
    for name in (
        "PYTEST_SPLIT_ARGS",
        "PYTEST_NODEIDS_FILE",
        "PYTEST_COLLECT_SELECTED",
        "PYTEST_TESTMON_ARGS",
        "PYTEST_PROGRESS_ARGS",
        "PYTEST_COVERAGE_REPORT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTEST_FORCE_ALL", "1")
    monkeypatch.setattr(sys, "argv", ["testing", "tier", "pytest-ci-smoke-global"])

    assert testing.main() == 4
    assert processes.calls[-1][-2:] == ["-m", "ci_smoke"]


class _Tiers(conftest.Recorder):
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
    tools: conftest.Recorder,
    **environ: str,
) -> list[str]:
    """Run tier ``name``; return the pytest argv after ``-m pytest -n=N``."""
    testing.tier(
        TIERS[name],
        context=conftest.context(tmp_path, tools, **environ),
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
        testing.tier(
            TIERS[name],
            context=conftest.context(tmp_path, tools, PYTEST_NODEIDS_FILE=str(listing)),
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
        context=conftest.context(tmp_path, tools),
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
        context=conftest.context(tmp_path, tools, PYTEST_COLLECT_SELECTED="s.txt"),
        plugin=None,
    )

    assert code == (2 if status == 2 else 0)


def test_a_tier_without_its_capability_runs_nothing(tmp_path: Path) -> None:
    tools = _Tiers(on_path=False)

    assert (
        testing.tier(
            TIERS["pytest-integration-global"],
            context=conftest.context(tmp_path, tools),
            plugin=None,
        )
        == 0
    )
    assert tools.calls == []


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
