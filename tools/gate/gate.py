"""Collect a hook's files, or run a tool once over them at this machine's size.

A heavy hook is two entries in ``.pre-commit-config.yaml``: ``spool GATE`` takes
pre-commit's file chunks, then the gate runs once over all of them, sized to
this machine and below interactive priority. The spool hook's ``alias`` is the
gate's id, so ``pre-commit run GATE`` selects both; a gate whose spool was never
written fails rather than check nothing. The type checkers and the pytest gates
are their own commands; these are the generic ones.

Commands:
  spool GATE     Add pre-commit's files to GATE's spool.
  run GATE       A module once over GATE's files: ``-m MODULE ... --workers N
                 -- FILES``.
  threaded NAME  A command, each ``{workers}`` in it set to the count.

Examples:
  python -m tools.gate spool leaks a.py b.py
  python -m tools.gate run --when OPT_IN leaks -- pkg.leaks --strict
  python -m tools.gate threaded pyrefly -- pyrefly check --threads={workers}

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
import shutil
import subprocess
import sys
import tempfile

from tools.gate import spool
from tools.gate.worker_count import default_workers, worker_count


def main() -> int:
    """Spool files, or run a module or command over them.

    Returns:
      exit_code: 0 when the gate passed.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    flags = _parse_args(parser)
    return flags.handler(flags, context())


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


def context() -> Context:
    """Return this process's context, after dropping below interactive priority.

    Every gate's ``main`` starts here, so each one runs at ``nice 19``.

    Returns:
      context: The spool, root, and tool runners of this pre-commit run.

    """
    os.nice(19)
    return Context(
        directory=Path(tempfile.gettempdir()) / f"precommit-spool-{os.getuid()}",
        key=spool.run_key(os.getpid(), process=_process),
        root=Path.cwd(),
        environ=os.environ,
        run=run_argv,
        output=_output,
        installed=_installed,
        which=shutil.which,
    )


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
        return unspooled(name)
    if not files:
        return 0
    count = workers(name, environ=context.environ)
    return context.run(
        [sys.executable, "-m", *command, "--workers", str(count), "--", *files],
    )


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


def run_argv(argv: Sequence[str]) -> int:
    """Run ``argv``; return its exit code, or 127 and a message when not installed."""
    try:
        return subprocess.run(argv, check=False).returncode  # noqa: S603 -- gate argv.
    except FileNotFoundError:
        return _missing(argv[0])


def unspooled(name: str) -> int:
    """Report a gate run without its spool hook; return the failing status."""
    sys.stderr.write(
        f"{name}: nothing was spooled. Its `{name}-spool` hook collects the "
        f"files, and `pre-commit run {name}` selects both.\n",
    )
    return 1


def walk(
    context: Context,
    *,
    files: Sequence[str],
    dependents: str | None,
    hops: int,
    emit: str,
    workers: int,
) -> list[str]:
    """Return what ``dependents`` names ``hops`` from ``files``; [] when off or failed.

    Args:
      context: What the gate reads and runs.
      files: The spooled files.
      dependents: Module whose CLI walks the reverse dependencies; None is off.
      hops: How far to walk; 0 is off.
      emit: ``files`` or ``nodeids``.
      workers: Seeds walked at once.

    Returns:
      found: One path or nodeid per entry.

    """
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


def add_dependents(parser: argparse.ArgumentParser) -> None:
    """Register ``--dependents``, the module that widens a gate's files."""
    parser.add_argument(
        "--dependents",
        metavar="MODULE",
        help="Widen to dependents with this module's CLI when opted in.",
    )


class Flags(Protocol):
    """Parsed command-line flags."""

    handler: Callable[[Flags, Context], int]
    command: str
    gate: str
    paths: list[str]
    when: str | None
    argv: list[str]


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
    """Register ``spool``, ``run`` and ``threaded``."""
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("spool", help="Add files to a gate's spool.")
    collect.set_defaults(handler=_spool_files)
    collect.add_argument("gate", help="The gate whose spool to add to.")
    collect.add_argument("paths", nargs="*", help="Files pre-commit passed.")
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
    sized = commands.add_parser("threaded", help="Run a command at the worker count.")
    sized.set_defaults(handler=_run_threaded)
    sized.add_argument("gate", help="The gate, naming its worker override.")
    sized.add_argument(
        "argv",
        nargs=argparse.REMAINDER,
        help="The command; {workers} becomes the worker count.",
    )


def _spool_files(flags: Flags, context: Context) -> int:
    spool.append(context.directory, gate=flags.gate, key=context.key, paths=flags.paths)
    return 0


def _run_once(flags: Flags, context: Context) -> int:
    return run(flags.gate, context=context, when=flags.when, command=flags.argv)


def _run_threaded(flags: Flags, context: Context) -> int:
    return threaded(flags.gate, context=context, command=flags.argv)


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
