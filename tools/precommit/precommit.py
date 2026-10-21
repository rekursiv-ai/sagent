"""Run one pre-commit gate over the files pre-commit hands it.

A heavy hook is two entries in ``.pre-commit-config.yaml``: ``spool GATE`` takes
pre-commit's file chunks, then the gate runs once over all of them, sized to
this machine and below interactive priority. The spool hook's ``alias`` is the
gate's id, so ``pre-commit run GATE`` selects both; a gate whose spool was never
written fails rather than check nothing.

Gates:
  ty-check       ty over the spooled files, minus the project's excludes.
  pyrefly-check  pyrefly over the files ty-check checked, or its own spool.
  basedpyright   basedpyright over its spool, widened as ty-check widens.
  pytest         The tests paired with the spooled files, then their coverage.
  run GATE       A module once over GATE's files: ``-m MODULE ... --workers N
                 -- FILES``.
  codegraph-sync The CodeGraph index, when a gate opted in to walking it.
  tier NAME      One whole-suite pytest tier from ``TIERS``.
  threaded NAME  A command, each ``{workers}`` in it set to the count.

Examples:
  python -m tools.precommit spool ty-check a.py b.py
  python -m tools.precommit ty-check
  python -m tools.precommit run --when OPT_IN leaks -- pkg.leaks --strict

"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import util
from pathlib import Path
from typing import Protocol, cast

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

from tools.precommit import (
    codegraph_sync as sync,
    coverage_summary,
    scope,
    spool,
)
from tools.precommit.tiers import TIERS, Tier
from tools.precommit.worker_count import default_workers, worker_count


def main() -> int:
    """Spool files, or run a gate.

    Returns:
      exit_code: 0 when the gate passed.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    flags = _parse_args(parser)
    os.nice(19)
    context = Context(
        directory=Path(tempfile.gettempdir()) / f"precommit-spool-{os.getuid()}",
        key=spool.run_key(os.getpid(), process=_process),
        root=Path.cwd(),
        environ=os.environ,
        run=_run,
        output=_output,
        installed=_installed,
        which=shutil.which,
    )
    return flags.handler(flags, context)


@dataclass(frozen=True, slots=True, kw_only=True)
class Context:
    """What a gate reads and runs.

    Attributes:
      directory: The spool directory.
      key: The pre-commit run.
      root: The project root, holding ``pyproject.toml``; the tools' cwd.
      environ: Environment the overrides are read from.
      run: Runs one argv; returns its exit code, 127 when it is not installed.
      output: Runs one argv; returns its stdout, or None when it failed.
      installed: Whether a module can be imported.
      which: Finds an executable on PATH.

    """

    directory: Path
    key: int
    root: Path
    environ: Mapping[str, str]
    run: Callable[[Sequence[str]], int]
    output: Callable[[Sequence[str]], str | None]
    installed: Callable[[str], bool]
    which: Callable[[str], str | None]


def typecheck(
    name: str,
    *,
    context: Context,
    dependents: str | None,
    fixer: str | None,
) -> int:
    """Run type checker ``name`` once over its spool.

    ``ty-check`` hands the files it checked to ``pyrefly-check``, so the graph
    walk runs once; ``pyrefly-check`` widens on its own only when ``ty-check``
    was skipped.

    Args:
      name: ``ty-check``, ``pyrefly-check`` or ``basedpyright``.
      context: What the gate reads and runs.
      dependents: Module whose CLI widens files to their dependents
        ``PRECOMMIT_CODEGRAPH_TYPECHECK`` hops away; None checks only the
        spooled files.
      fixer: Module that runs the checker and deletes each unnecessary
        ``assert isinstance``; None runs the checker alone.

    Returns:
      exit_code: The checker's; 0 when there was nothing to check; 1 when its
        spool was never written.

    """
    spooled = spool.drain(context.directory, gate=name, key=context.key)
    if spooled is None:
        return _unspooled(name)
    count = workers(name, environ=context.environ)
    handed = (
        spool.drain(context.directory, gate="typecheck-checked", key=context.key)
        if name == "pyrefly-check"
        else None
    )
    files = (
        _typechecked(context, files=spooled, dependents=dependents, workers=count)
        if handed is None
        else handed
    )
    if name == "ty-check":
        spool.write(
            context.directory,
            gate="typecheck-checked",
            key=context.key,
            paths=files,
        )
    if not files:
        return 0
    if fixer is not None:
        checker = name.removesuffix("-check")
        return context.run(
            [
                sys.executable,
                "-m",
                fixer,
                "--checker",
                checker,
                "--fix",
                "--threads",
                str(count),
                "--",
                *files,
            ],
        )
    tool, *flags = {
        "ty-check": ["ty", "check", "--force-exclude"],
        "pyrefly-check": ["pyrefly", "check", f"--threads={count}"],
        "basedpyright": ["basedpyright", f"--threads={count}"],
    }[name]
    return context.run([_tool(tool), *flags, *files])


def pytest(context: Context, *, dependents: str | None) -> int:
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
    files = spool.drain(context.directory, gate="pytest", key=context.key)
    if files is None:
        return _unspooled("pytest")
    count = workers("pytest", environ=context.environ)
    measured = context.installed("pytest_cov")
    if measured and (
        erased := context.run([sys.executable, "-m", "coverage", "erase"])
    ):
        return erased
    tests = _selected(context, files=files, dependents=dependents, workers=count)
    if not tests:
        return 0
    # Files, not nodeids: the threshold was measured in files, and dozens of
    # nodeids from one file paid xdist's ~5s startup for nothing.
    spread = len({test.partition("::")[0] for test in tests}) >= (
        positive(context.environ.get("SIDECAR_XDIST_MIN", "")) or 16
    )
    sources = [path for path in files if not _is_test(path)]
    directories = sorted({Path(path).parent.as_posix() for path in sources})
    cov: list[str] = (
        [
            *(f"--cov={directory}" for directory in directories),
            "--cov-append",
            "--no-cov-on-fail",
            "--cov-fail-under=0",
            "--cov-report=term-missing:skip-covered"
            if context.environ.get("PRECOMMIT_COVERAGE_DETAIL") == "1"
            else "--cov-report=",
        ]
        if measured and directories
        else []
    )
    # An @argfile: thousands of nodeids or one --cov per file overflow ARG_MAX.
    with tempfile.NamedTemporaryFile(
        "w",
        prefix="precommit-",
        suffix=".args",
    ) as argfile:
        argfile.write("".join(f"{arg}\n" for arg in [*cov, *tests]))
        argfile.flush()
        status = context.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *(["-n", str(count)] if spread else []),
                "-q",
                "--tb=short",
                f"@{argfile.name}",
            ],
        )
    if status not in {0, 5}:
        return status
    if not cov or not (context.root / ".coverage").is_file():
        return 0
    below = coverage_summary.shortfall(sources)
    sys.stderr.writelines(f"{line}\n" for line in below)
    return int(bool(below))


def codegraph_sync(context: Context, *, caches: Path) -> int:
    """Sync the CodeGraph index when a gate opted in to walking it.

    Args:
      context: What the gate reads and runs.
      caches: Where the index lives, one directory per worktree name.

    Returns:
      exit_code: Always 0; a gate without an index falls back to its floor.

    """
    if any(
        positive(context.environ.get(name, ""))
        for name in ("PRECOMMIT_CODEGRAPH_PYTEST", "PRECOMMIT_CODEGRAPH_TYPECHECK")
    ):
        sync.sync(context.root, caches=caches, run=context.run, which=context.which)
    return 0


def run(
    name: str,
    *,
    context: Context,
    when: str | None,
    command: Sequence[str],
) -> int:
    """Run ``python -m COMMAND --workers N -- FILES`` once over spool ``name``.

    Args:
      name: The gate, which names its spool and its ``WORKERS_<NAME>`` override.
      context: What the gate reads and runs.
      when: Variable that must hold a positive integer for the gate to run;
        None always runs.
      command: The module and its leading arguments.

    Returns:
      exit_code: The module's; 0 when skipped or nothing was spooled; 1 when
        the spool was never written.

    """
    files = spool.drain(context.directory, gate=name, key=context.key)
    if when is not None and not positive(context.environ.get(when, "")):
        return 0
    if files is None:
        return _unspooled(name)
    if not files:
        return 0
    count = workers(name, environ=context.environ)
    return context.run(
        [sys.executable, "-m", *command, "--workers", str(count), "--", *files],
    )


def tier(spec: Tier, *, context: Context, plugin: str | None) -> int:
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
    if listing := environ.get("PYTEST_NODEIDS_FILE", ""):
        nodeids = [
            line for line in Path(listing).read_text().splitlines() if "::" in line
        ]
        if not nodeids:
            return 0
    split = shlex.split(environ.get("PYTEST_SPLIT_ARGS", ""))
    selects = (
        not environ.get("PYTEST_FORCE_ALL", "")
        and not split
        and (
            spec.ci_testmon
            or bool(environ.get("PYTEST_COLLECT_SELECTED", ""))
            or not environ.get("GITHUB_ACTIONS", "")
        )
        and context.installed("testmon")
    )
    select = ["--testmon-forceselect"] if selects else []
    measured = context.installed("pytest_cov")
    cov: list[str] = []
    if not selects and not environ.get("PYTEST_TESTMON_ARGS", "") and measured:
        cov = ["--no-cov"]
    # Bare `--cov` measures the project, as `[tool.coverage.run]` scopes it.
    if environ.get("PYTEST_COVERAGE_REPORT", "") and measured:
        cov = ["--cov", "--cov-report=term:skip-covered"]
    marker = ["-m", spec.marker] if spec.marker else []
    pytest_argv = [
        "env",
        "PYTHONUNBUFFERED=1",
        "COVERAGE_CORE=ctrace",
        sys.executable,
        "-m",
        "pytest",
        f"-n={workers(spec.name, environ=environ)}",
    ]
    if environ.get("PYTEST_COLLECT_SELECTED", ""):
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


def threaded(name: str, *, context: Context, command: Sequence[str]) -> int:
    """Run ``command`` with each ``{workers}`` set to gate ``name``'s count.

    Args:
      name: The gate; also names the ``WORKERS_<NAME>`` override.
      context: What the gate reads and runs.
      command: The argv; pre-commit appends any file names after it.

    Returns:
      exit_code: The command's.

    """
    count = str(workers(name, environ=context.environ))
    return context.run([part.replace("{workers}", count) for part in command])


def workers(name: str, *, environ: Mapping[str, str]) -> int:
    """Return gate ``name``'s worker count.

    ``WORKERS_<NAME>`` wins, then whatever ``default_workers`` reads. Per gate
    because CI's runners fit fewer workers for some gates than others.

    Args:
      name: The gate.
      environ: Environment the overrides are read from.

    Returns:
      workers: At least 1.

    """
    override = environ.get(f"WORKERS_{name.replace('-', '_').upper()}", "")
    return worker_count(override) if override else default_workers(environ=environ)


def positive(value: str) -> int:
    """Return ``value`` as a positive integer, or 0 for anything else.

    An allowlist: only a positive integer without a leading zero counts. The
    opt-ins it reads cost minutes when they run by mistake, and ``0`` once read
    as on.

    Args:
      value: The variable's value.

    Returns:
      count: The value, or 0.

    """
    return int(value) if re.fullmatch(r"[1-9][0-9]*", value) else 0


class Flags(Protocol):
    """Parsed command-line flags."""

    handler: Callable[[Flags, Context], int]
    command: str
    gate: str
    tier: str
    paths: list[str]
    dependents: str | None
    fixer: str | None
    when: str | None
    argv: list[str]
    caches: Path
    plugin: str | None


def _parse_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> Flags:
    """Register the commands and parse ``argv``; a command follows ``--``."""
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args(argv))
    if flags.command in {"run", "threaded"}:
        flags.argv = flags.argv[flags.argv[:1] == ["--"] :]
        if not flags.argv:
            parser.error(f"{flags.command}: name the command after --")
    return flags


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``spool`` command and one command per gate."""
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("spool", help="Add files to a gate's spool.")
    collect.set_defaults(handler=_spool_files)
    collect.add_argument("gate", help="The gate whose spool to add to.")
    collect.add_argument("paths", nargs="*", help="Files pre-commit passed.")
    for name in ("ty-check", "pyrefly-check", "basedpyright"):
        checker = commands.add_parser(name, help=f"Run {name} over its spool.")
        checker.set_defaults(handler=_check, fixer=None)
        _add_dependents(checker)
        if name != "ty-check":
            checker.add_argument(
                "--fixer",
                metavar="MODULE",
                help="Run the checker through this module, which drops needless asserts.",
            )
    tests = commands.add_parser("pytest", help="Run the tests paired with the spool.")
    tests.set_defaults(handler=_test)
    _add_dependents(tests)
    index = commands.add_parser("codegraph-sync", help="Sync the CodeGraph index.")
    index.set_defaults(handler=_sync_index)
    index.add_argument(
        "--caches",
        type=Path,
        required=True,
        help="Where the index lives, one directory per worktree name.",
    )
    once = commands.add_parser("run", help="Run a module once over a spool.")
    once.set_defaults(handler=_run_once)
    once.add_argument("gate", help="The gate, naming its spool and worker override.")
    once.add_argument(
        "--when",
        metavar="VARIABLE",
        help="Run only when this variable holds a positive integer.",
    )
    once.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="The module, then its leading arguments.",
    )
    suite = commands.add_parser("tier", help="Run one whole-suite pytest tier.")
    suite.set_defaults(handler=_run_tier)
    suite.add_argument("tier", choices=sorted(TIERS), help="The tier.")
    suite.add_argument(
        "--plugin",
        metavar="MODULE",
        help="Load this module with -p when collecting a selection.",
    )
    sized = commands.add_parser("threaded", help="Run a command at the worker count.")
    sized.set_defaults(handler=_run_threaded)
    sized.add_argument("gate", help="The gate, naming its worker override.")
    sized.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="The command; {workers} becomes the worker count.",
    )


def _add_dependents(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dependents",
        metavar="MODULE",
        help="Widen to dependents with this module's CLI when opted in.",
    )


def _spool_files(flags: Flags, context: Context) -> int:
    spool.append(context.directory, gate=flags.gate, key=context.key, paths=flags.paths)
    return 0


def _check(flags: Flags, context: Context) -> int:
    return typecheck(
        flags.command,
        context=context,
        dependents=flags.dependents,
        fixer=flags.fixer,
    )


def _test(flags: Flags, context: Context) -> int:
    return pytest(context, dependents=flags.dependents)


def _sync_index(flags: Flags, context: Context) -> int:
    return codegraph_sync(context, caches=flags.caches)


def _run_once(flags: Flags, context: Context) -> int:
    return run(flags.gate, context=context, when=flags.when, command=flags.argv)


def _run_tier(flags: Flags, context: Context) -> int:
    return tier(TIERS[flags.tier], context=context, plugin=flags.plugin)


def _run_threaded(flags: Flags, context: Context) -> int:
    return threaded(flags.gate, context=context, command=flags.argv)


def _unspooled(name: str) -> int:
    """Report a gate run without its spool hook; return the failing status."""
    sys.stderr.write(
        f"{name}: nothing was spooled. Its `{name}-spool` hook collects the "
        f"files, and `pre-commit run {name}` selects both.\n",
    )
    return 1


def _typechecked(
    context: Context,
    *,
    files: Sequence[str],
    dependents: str | None,
    workers: int,
) -> list[str]:
    """Return the type checkers' files: ``files`` widened, minus excludes."""
    widened = _walk(
        context,
        files=files,
        dependents=dependents,
        hops=positive(context.environ.get("PRECOMMIT_CODEGRAPH_TYPECHECK", "")),
        emit="files",
        workers=workers,
    )
    return scope.kept(widened or files, excludes=scope.excludes(context.root))


def _selected(
    context: Context,
    *,
    files: Sequence[str],
    dependents: str | None,
    workers: int,
) -> list[str]:
    """Return the tests for ``files``: the graph's, else each file's pairs."""
    walked = _walk(
        context,
        files=files,
        dependents=dependents,
        hops=positive(context.environ.get("PRECOMMIT_CODEGRAPH_PYTEST", "")),
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


def _walk(
    context: Context,
    *,
    files: Sequence[str],
    dependents: str | None,
    hops: int,
    emit: str,
    workers: int,
) -> list[str]:
    """Return what ``dependents`` names ``hops`` from ``files``; [] when off or failed."""
    if not files or not hops or dependents is None:
        return []
    found = context.output(
        [
            sys.executable,
            "-m",
            dependents,
            "--emit",
            emit,
            "--depth",
            str(hops),
            "--workers",
            str(workers),
            "--",
            *files,
        ],
    )
    return [] if found is None else found.splitlines()


def _is_test(path: str) -> bool:
    name = Path(path).name
    return (
        name == "conftest.py" or name.endswith("_test.py") or name.startswith("test_")
    )


def _tool(name: str) -> str:
    """Return ``name``'s executable beside this interpreter, else from PATH."""
    beside = Path(sys.executable).parent / name
    return str(beside) if beside.is_file() else (shutil.which(name) or name)


def _run(argv: Sequence[str]) -> int:
    try:
        return subprocess.run(argv, check=False).returncode  # noqa: S603 -- gate argv.
    except FileNotFoundError:
        return _missing(argv[0])


def _output(argv: Sequence[str]) -> str | None:
    try:
        done = subprocess.run(  # noqa: S603 -- gate argv.
            argv,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        _missing(argv[0])
        return None
    return done.stdout if done.returncode == 0 else None


def _missing(tool: str) -> int:
    """Report a tool that is not installed; return the shell's status for it."""
    sys.stderr.write(f"{tool}: command not found\n")
    return 127


def _installed(module: str) -> bool:
    try:
        return util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _process(pid: int) -> tuple[int, str] | None:
    """Return ``pid``'s parent and command line, from ``ps``."""
    done = subprocess.run(  # noqa: S603 -- fixed command, numeric pid.
        ["ps", "-o", "ppid=,command=", "-p", str(pid)],  # noqa: S607 -- system ps.
        capture_output=True,
        text=True,
        check=False,
    )
    parent, _, command = done.stdout.strip().partition(" ")
    return (int(parent), command) if parent.isdigit() else None
