"""Tests for the pytest gates."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast, override
from unittest.mock import Mock

import argparse
import os
import subprocess
import sys

import coverage
import pytest

from sagent.lib.testing.checks import assert_help_without_docstring
from tools.gate import fakes, gate, spool
from tools.testing import coverage_summary, listing, testing
from tools.testing.tiers import TIERS, Tier
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

    *head, test_files, argfile = tools.calls[-1]
    assert head[3:] == [
        *("-m", "pytest", "-n", "3", "-q", "--tb=short"),
        *("-p", listing.__name__),
    ]
    selection = Path(test_files.removeprefix("--test-files="))
    assert selection.name == "selected.txt"
    assert selection.parent.name.startswith("precommit-")
    path = Path(argfile.removeprefix("@"))
    assert path.name == "args"
    assert path.parent.name.startswith("gate-")


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
    assert tools.argfiles == [["a.py"], ["b_test.py"]]
    assert tools.listings == [["b_test.py::t"]]


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

    # The walk's own argfile, then pytest's.
    assert tools.argfiles == [["a.py"], ["a_test.py"]]


class _Runs(fakes.Recorder):
    """Records each command, then runs it for real in ``root``."""

    def __init__(self, root: Path, *, output: str) -> None:
        super().__init__(output=output, installed=False)
        self.root = root
        self.returned: list[int] = []

    @override
    def run(self, argv: Sequence[str]) -> int:
        _ = super().run(argv)
        self.returned.append(
            subprocess.run(  # noqa: S603 -- the gate's own pytest argv.
                argv,
                cwd=self.root,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(sys.path),
                },
                capture_output=True,
                check=False,
            ).returncode,
        )
        return self.returned[-1]


def test_a_subprocess_keeps_its_import_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.syspath_prepend(str(tmp_path))

    assert _Runs(tmp_path, output="").run(["pytest"]) == 0
    path = cast(dict[str, str], run.call_args.kwargs["env"])["PYTHONPATH"]
    assert path == os.pathsep.join(sys.path)


def _skipped_module(root: Path) -> None:
    root.mkdir()
    (root / "skip_test.py").write_text(
        "import pytest\n\n"
        'pytest.skip("off", allow_module_level=True)\n\n\n'
        "def test_a():\n    pass\n",
    )


def test_a_module_skipped_test_does_not_drop_the_rest_of_the_walk(
    tmp_path: Path,
) -> None:
    """Pytest 9 runs nothing else once one nodeid points into a skipped module.

    Each walked test runs, and no unwalked test beside it does.
    """
    root = tmp_path / "repo"
    _skipped_module(root)
    for module in ("b", "c"):
        (root / f"{module}_test.py").write_text(
            "from pathlib import Path\n"
            + "".join(
                f'\n\ndef test_{name}():\n    Path("ran_{name}").touch()\n'
                for name in (module, f"{module}_unwalked")
            ),
        )
    tools = _Runs(
        root,
        output="b_test.py::test_b\nskip_test.py::test_a\nc_test.py::test_c\n",
    )

    status = _pytest(
        tmp_path,
        tools,
        ["a.py"],
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_PYTEST="1",
    )

    assert sorted(path.name for path in root.glob("ran_*")) == ["ran_b", "ran_c"]
    assert status == 0


def test_a_walk_reaching_only_a_skipped_module_passes(tmp_path: Path) -> None:
    """Pytest exits 5 when its every test skips at module level: nothing failed.

    A live-acceptance module skips unless its variable is set, and a change
    may reach no other test.
    """
    root = tmp_path / "repo"
    _skipped_module(root)
    tools = _Runs(root, output="skip_test.py::test_a\n")

    status = _pytest(
        tmp_path,
        tools,
        ["a.py"],
        dependents="pkg.affected",
        PRECOMMIT_CODEGRAPH_PYTEST="1",
    )

    assert tools.returned == [5]
    assert status == 0


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


@pytest.mark.parametrize(
    ("name", "test"),
    [
        ("conftest.py", True),
        ("a_test.py", True),
        ("test_a.py", True),
        ("a.py", False),
        ("test.py", False),
        ("pytest_a.py", False),
    ],
)
def test_a_test_file_is_named_as_pytest_collects_one(name: str, test: bool) -> None:
    assert testing._is_test(f"pkg/{name}") is test


def test_a_spooled_test_file_named_test_first_runs_itself(tmp_path: Path) -> None:
    tools = fakes.Recorder()

    _pytest(tmp_path, tools, ["pkg/test_c.py"])

    assert tools.argfiles == [["pkg/test_c.py"]]


@pytest.mark.parametrize(
    ("argv", "shown"),
    [
        (
            ["--help"],
            [
                "Run the tests paired with the spool.",
                "Run one whole-suite pytest tier.",
            ],
        ),
        (
            ["tier", "--help"],
            ["The tier.", "--plugin MODULE", "pytest-global"],
        ),
    ],
    ids=["commands", "tier"],
)
def test_each_command_explains_itself(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    shown: list[str],
) -> None:
    del processes
    monkeypatch.setattr(sys, "argv", ["testing", *argv])
    with pytest.raises(SystemExit, match=r"^0$"):
        testing.main()
    out = " ".join(capsys.readouterr().out.split())
    assert [text for text in shown if text not in out] == []


def test_the_help_opens_with_the_module_docstring(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del processes
    monkeypatch.setattr(sys, "argv", ["testing", "--help"])
    with pytest.raises(SystemExit):
        testing.main()
    assert (
        capsys.readouterr().out.split("\n\n")[1]
        == (testing.__doc__ or "").split(
            "\n\n",
        )[0]
    )


def test_without_a_docstring_the_help_has_no_description(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del processes
    assert_help_without_docstring(monkeypatch, testing, testing.main)
    assert capsys.readouterr().out.split("\n\n")[1].startswith("positional arguments")


@pytest.mark.parametrize(
    ("argv", "error"),
    [
        ([], "the following arguments are required: command"),
        (["tier", "nosuch"], "invalid choice: 'nosuch'"),
    ],
    ids=["no-command", "unknown-tier"],
)
def test_a_bad_command_line_is_a_usage_error(
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    error: str,
) -> None:
    with pytest.raises(SystemExit, match=r"^2$"):
        testing._parse_args(argparse.ArgumentParser(), argv)
    assert error in capsys.readouterr().err


def test_parse_args_reads_the_argv_it_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["testing", "pytest"])
    flags = testing._parse_args(
        argparse.ArgumentParser(),
        ["tier", "pytest-global", "--plugin", "pkg.p"],
    )
    assert (flags.tier, flags.plugin) == ("pytest-global", "pkg.p")


def test_the_cli_passes_its_plugin_to_a_tier_collection(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTEST_COLLECT_SELECTED", "selected.txt")
    monkeypatch.setattr(
        sys,
        "argv",
        ["testing", "tier", "pytest-ci-smoke-global", "--plugin", "argparse"],
    )

    testing.main()

    assert processes.calls[-1][processes.calls[-1].index("-p") + 1] == "argparse"


def test_the_cli_walks_the_graph_with_its_dependents_module(
    processes: fakes.Processes,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "a.py").write_text("")
    monkeypatch.setenv("PRECOMMIT_CODEGRAPH_PYTEST", "1")
    monkeypatch.setattr(sys, "argv", ["gate", "spool", "pytest", "a.py"])
    gate.main()
    monkeypatch.setattr(sys, "argv", ["testing", "pytest", "--dependents", "pkg.deps"])

    testing.main()

    walked = [call for call in processes.calls if "pkg.deps" in call]
    assert [call[1:3] for call in walked] == [["-m", "pkg.deps"]]


def test_a_tier_s_full_run_and_collection_are_exact(tmp_path: Path) -> None:
    tools = _Tiers()
    context = fakes.context(
        tmp_path,
        tools,
        PYTEST_FORCE_ALL="1",
        PYTEST_TESTMON_ARGS="--t",
        PYTEST_PROGRESS_ARGS="--p",
    )
    testing.tier(TIERS["pytest-ci-smoke-global"], context=context, plugin=None)
    collecting = fakes.context(tmp_path, tools, PYTEST_COLLECT_SELECTED="s.txt")
    testing.tier(TIERS["pytest-ci-smoke-global"], context=collecting, plugin=None)

    head = ["env", "PYTHONUNBUFFERED=1", "COVERAGE_CORE=ctrace"]
    pytest_argv = [*head, sys.executable, "-m", "pytest", "-n=3"]
    assert tools.calls == [
        [
            *pytest_argv,
            "--t",
            "-q",
            "--tb=short",
            "--durations=20",
            "--p",
            "-m",
            "ci_smoke",
        ],
        [
            *pytest_argv,
            "--testmon-forceselect",
            "-q",
            "--tb=short",
            "--durations=20",
            "--collect-only",
            "-m",
            "ci_smoke",
        ],
    ]


def test_a_full_run_without_testmon_or_progress_args_adds_none(
    tmp_path: Path,
) -> None:
    tools = _Tiers()

    testing.tier(
        TIERS["pytest-ci-smoke-global"],
        context=fakes.context(tmp_path, tools, PYTEST_FORCE_ALL="1"),
        plugin=None,
    )

    head = ["env", "PYTHONUNBUFFERED=1", "COVERAGE_CORE=ctrace"]
    assert tools.calls == [
        [
            *(*head, sys.executable, "-m", "pytest", "-n=3"),
            *("--no-cov", "-q", "--tb=short", "--durations=20", "-m", "ci_smoke"),
        ],
    ]


def test_a_tier_with_no_markers_passes_no_marker(tmp_path: Path) -> None:
    tools = _Tiers()

    testing.tier(
        Tier(name="bare"),
        context=fakes.context(tmp_path, tools, PYTEST_FORCE_ALL="1"),
        plugin=None,
    )

    [argv] = tools.calls
    assert argv.count("-m") == 1


def test_a_collection_that_fails_reports_its_status(tmp_path: Path) -> None:
    tools = _Tiers()
    tools.status = 1

    code = testing.tier(
        TIERS["pytest-global"],
        context=fakes.context(tmp_path, tools, PYTEST_COLLECT_SELECTED="s.txt"),
        plugin=None,
    )

    assert code == 1


def test_a_full_run_without_pytest_cov_passes_no_cov_flag(tmp_path: Path) -> None:
    tools = _Tiers(missing=frozenset({"pytest_cov"}))

    flags = _tier(tmp_path, "pytest-global", tools, PYTEST_FORCE_ALL="1")

    assert "--no-cov" not in flags


class _Locates(_Tiers):
    """A recorder that finds only ``pg_config`` on PATH."""

    @override
    def which(self, name: str) -> str | None:
        return f"/bin/{name}" if name == "pg_config" else None


def test_a_tier_runs_where_its_capability_is_on_path(tmp_path: Path) -> None:
    tools = _Locates()

    testing.tier(
        TIERS["pytest-integration-global"],
        context=fakes.context(tmp_path, tools, PYTEST_FORCE_ALL="1"),
        plugin=None,
    )

    assert len(tools.calls) == 1


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
