"""Run the tests a change reaches, or one whole-suite pytest tier.

Commands:
  pytest     The tests paired with the spooled files, then their coverage.
  tier NAME  One whole-suite pytest tier from ``TIERS``.

Examples:
  python -m tools.gate spool pytest a.py b.py
  python -m tools.testing pytest
  python -m tools.testing tier pytest-global

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import shlex
import sys
import tempfile

from tools.gate import gate
from tools.testing import coverage_summary, listing
from tools.testing.tiers import TIERS, Tier


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def main() -> int:
    """Run the paired tests or a tier.

    Returns:
      exit_code: pytest's, or the coverage summary's.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    flags = _parse_args(parser)
    return flags.handler(flags, gate.context())


def pytest(context: gate.Context, *, dependents: str | None) -> int:
    """Run the tests paired with the ``pytest`` spool, then gate their coverage.

    Pairs ``x.py`` with ``x_test.py`` or ``test_x.py``, beside it or under
    ``tests/``. Green with no pair found means "untested", not "tested".

    Args:
      context: What the gate reads and runs.
      dependents: Module whose CLI names the tests reaching the spooled files
        ``PRECOMMIT_CODEGRAPH_PYTEST`` hops away; the pairs remain the fallback
        when it names none.

    Returns:
      exit_code: pytest's, 0 for an empty collection, or the coverage summary's;
        1 when the spool was never written.

    """
    name = "pytest"
    files = gate.drained(name, context=context)
    if files is None:
        return gate.unspooled(name)
    count = gate.workers(name, environ=context.environ)
    tests = _selected(context, files=files, dependents=dependents, workers=count)
    if not tests:
        return 0
    # Files, not nodeids: the threshold was measured in files, and dozens of
    # nodeids from one file paid xdist's ~5s startup for nothing.
    spread = len({test.partition("::")[0] for test in tests}) >= (
        gate.positive(context.environ.get("SIDECAR_XDIST_MIN")) or 16
    )
    sources = [path for path in files if not _is_test(path)]
    directories = sorted({Path(path).parent.as_posix() for path in sources})
    measured = context.installed("pytest_cov")
    cov: list[str] = (
        [
            *(f"--cov={directory}" for directory in directories),
            "--no-cov-on-fail",
            "--cov-fail-under=0",
            "--cov-report=term-missing:skip-covered"
            if context.environ.get("PRECOMMIT_COVERAGE_DETAIL") == "1"
            else "--cov-report=",
        ]
        if measured and directories
        else []
    )
    # A private data file: `coverage erase` or a combine in the checkout deletes
    # every `.coverage.*` there, killing a concurrent `pytest --cov`'s workers.
    with tempfile.TemporaryDirectory(prefix="precommit-") as temp:
        # pragma: no mutate start -- any name in a private dir.
        data = Path(temp) / ".coverage"
        # pragma: no mutate end
        # Files as arguments and the nodeids as a listing: once one nodeid
        # argument points into a module that skips at module level, pytest
        # drops every other argument and the gate passes on "1 skipped".
        selection = Path(temp) / "selected.txt"
        selection.write_text("".join(f"{test}\n" for test in tests))
        files_run = sorted({test.partition("::")[0] for test in tests})
        # An @argfile: thousands of files or one --cov per file overflow ARG_MAX.
        argfile = Path(temp) / "pytest.args"
        argfile.write_text("".join(f"{arg}\n" for arg in [*cov, *files_run]))
        status = context.run(
            [
                "env",
                f"COVERAGE_FILE={data}",
                sys.executable,
                "-m",
                "pytest",
                *(["-n", str(count)] if spread else []),
                "-q",
                "--tb=short",
                "-p",
                listing.__name__,
                f"--test-files={selection}",
                f"@{argfile}",
            ],
        )
        if status not in {0, 5}:
            return status
        if not cov or not data.is_file():
            return 0
        below = coverage_summary.shortfall(sources, data=data)
    sys.stderr.writelines(f"{line}\n" for line in below)
    return int(bool(below))


def tier(spec: Tier, *, context: gate.Context, plugin: str | None) -> int:
    """Run a whole-suite pytest tier, selecting with testmon when that is safe.

    Testmon and pytest-split both partition one collection, and which wins is
    plugin order: a sharded run with a warm database once ran 8 of 118 tests and
    stayed green. So selection is derived, never configured: a non-empty
    ``PYTEST_SPLIT_ARGS`` (a shard) or ``PYTEST_FORCE_ALL`` turns it off. It is
    ``--testmon-forceselect``, not ``--testmon``: the tier passes ``-m``, and
    plain ``--testmon`` then selects nothing (tarpas/pytest-testmon#238).
    ``COVERAGE_CORE=ctrace`` because sysmon cannot carry testmon's per-test
    contexts, and testmon reads only the environment.

    ``PYTHONUNBUFFERED=1``: pre-commit reads the hook through a pipe, so a tier
    killed for memory otherwise leaves no progress at all.

    Args:
      spec: The tier.
      context: What the gate reads and runs.
      plugin: Module loaded with ``-p`` when collecting a selection, if
        importable; it writes the selected node ids.

    Returns:
      exit_code: pytest's; 0 when the tier is skipped, its selection is empty,
        or, under ``allow_empty``, nothing was collected.

    """
    environ = context.environ
    if spec.requires is not None and context.which(spec.requires) is None:
        return 0
    nodeids: list[str] = []
    if gate.flagged(environ, "PYTEST_NODEIDS_FILE"):
        listed = Path(environ["PYTEST_NODEIDS_FILE"]).read_text().splitlines()
        nodeids = [line for line in listed if "::" in line]
        if not nodeids:
            return 0
    split = shlex.split(environ.get("PYTEST_SPLIT_ARGS", ""))
    selects = (
        not gate.flagged(environ, "PYTEST_FORCE_ALL")
        and not split
        and (
            spec.ci_testmon
            or gate.flagged(environ, "PYTEST_COLLECT_SELECTED")
            or not gate.flagged(environ, "GITHUB_ACTIONS")
        )
        and context.installed("testmon")
    )
    select = ["--testmon-forceselect"] if selects else []
    measured = context.installed("pytest_cov")
    cov: list[str] = []
    if not selects and not gate.flagged(environ, "PYTEST_TESTMON_ARGS") and measured:
        cov = ["--no-cov"]
    # Bare `--cov` measures the project, as `[tool.coverage.run]` scopes it.
    if gate.flagged(environ, "PYTEST_COVERAGE_REPORT") and measured:
        cov = ["--cov", "--cov-report=term:skip-covered"]
    marker = ["-m", spec.marker] if spec.marker else []
    pytest_argv = [
        "env",
        "PYTHONUNBUFFERED=1",
        "COVERAGE_CORE=ctrace",
        sys.executable,
        "-m",
        "pytest",
        f"-n={gate.workers(spec.name, environ=environ)}",
    ]
    if gate.flagged(environ, "PYTEST_COLLECT_SELECTED"):
        loaded = (
            ["-p", plugin] if plugin is not None and context.installed(plugin) else []
        )
        status = context.run(
            [
                *pytest_argv,
                *select,
                *loaded,
                "-q",
                "--tb=short",
                *spec.extra,
                "--collect-only",
                *marker,
            ],
        )
        return 0 if status in {0, 5} else status
    status = context.run(
        [
            *pytest_argv,
            *cov,
            *select,
            *shlex.split(environ.get("PYTEST_TESTMON_ARGS", "")),
            "-q",
            "--tb=short",
            *spec.extra,
            *shlex.split(environ.get("PYTEST_PROGRESS_ARGS", "")),
            *marker,
            *split,
            *nodeids,
        ],
    )
    return 0 if spec.allow_empty and status == 5 else status


class Flags(Protocol):
    """Parsed command-line flags."""

    handler: Callable[[Flags, gate.Context], int]
    tier: str
    dependents: str | None
    plugin: str | None


def _parse_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> Flags:
    """Register ``pytest`` and ``tier`` and parse ``argv``."""
    commands = parser.add_subparsers(dest="command", required=True)
    tests = commands.add_parser("pytest", help="Run the tests paired with the spool.")
    tests.set_defaults(handler=_test)
    gate.add_dependents(tests)
    suite = commands.add_parser("tier", help="Run one whole-suite pytest tier.")
    suite.set_defaults(handler=_run_tier)
    suite.add_argument("tier", choices=sorted(TIERS), help="The tier.")
    suite.add_argument(
        "--plugin",
        metavar="MODULE",
        help="Load this module with -p when collecting a selection.",
    )
    return cast(Flags, parser.parse_args(argv))


def _test(flags: Flags, context: gate.Context) -> int:
    return pytest(context, dependents=flags.dependents)


def _run_tier(flags: Flags, context: gate.Context) -> int:
    return tier(TIERS[flags.tier], context=context, plugin=flags.plugin)


def _selected(
    context: gate.Context,
    *,
    files: Sequence[str],
    dependents: str | None,
    workers: int,
) -> list[str]:
    """Return the tests for ``files``: the graph's, else each file's pairs."""
    walked = gate.walk(
        context,
        files=files,
        dependents=dependents,
        hops=gate.positive(context.environ.get("PRECOMMIT_CODEGRAPH_PYTEST")),
        emit="nodeids",
        workers=workers,
    )
    if walked:
        return walked
    paired: set[str] = set()
    for path in map(Path, files):
        stem = path.stem.removesuffix("_test").removeprefix("test_")
        paired.update(
            candidate.as_posix()
            for folder in (path.parent, path.parent / "tests")
            for candidate in (folder / f"{stem}_test.py", folder / f"test_{stem}.py")
            if (context.root / candidate).is_file()
        )
    return sorted(paired)


def _is_test(path: str) -> bool:
    name = Path(path).name
    return (
        name == "conftest.py" or name.endswith("_test.py") or name.startswith("test_")
    )
