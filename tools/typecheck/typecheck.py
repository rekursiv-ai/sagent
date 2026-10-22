"""Run one type checker once over the files pre-commit hands its spool.

``ty-check`` hands the files it checked to ``pyrefly-check``, so the graph walk
runs once; ``pyrefly-check`` widens on its own only when ``ty-check`` was
skipped. Every checker drops what ``[tool.pyrefly] project-excludes`` names.

Checkers:
  ty-check       ty over the spooled files, minus the project's excludes.
  pyrefly-check  pyrefly over the files ty-check checked, or its own spool.
  basedpyright   basedpyright over its spool, widened as ty-check widens.

Examples:
  python -m tools.gate spool ty-check a.py b.py
  python -m tools.typecheck ty-check

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import shutil
import sys

from tools.gate import gate, spool
from tools.typecheck import scope


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def main() -> int:
    """Run one checker over its spool.

    Returns:
      exit_code: The checker's.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    flags = _parse_args(parser)
    return flags.handler(flags, gate.context())


def typecheck(
    name: str,
    *,
    context: gate.Context,
    dependents: str | None,
    fixer: str | None,
) -> int:
    """Run type checker ``name`` once over its spool.

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
        return gate.unspooled(name)
    count = gate.workers(name, environ=context.environ)
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


class Flags(Protocol):
    """Parsed command-line flags."""

    handler: Callable[[Flags, gate.Context], int]
    command: str
    dependents: str | None
    fixer: str | None


def _parse_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> Flags:
    """Register one command per checker and parse ``argv``."""
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("ty-check", "pyrefly-check", "basedpyright"):
        checker = commands.add_parser(name, help=f"Run {name} over its spool.")
        checker.set_defaults(handler=_check, fixer=None)
        gate.add_dependents(checker)
        if name != "ty-check":
            checker.add_argument(
                "--fixer",
                metavar="MODULE",
                help="Run the checker through this module, which drops needless asserts.",
            )
    return cast(Flags, parser.parse_args(argv))


def _check(flags: Flags, context: gate.Context) -> int:
    return typecheck(
        flags.command,
        context=context,
        dependents=flags.dependents,
        fixer=flags.fixer,
    )


def _typechecked(
    context: gate.Context,
    *,
    files: Sequence[str],
    dependents: str | None,
    workers: int,
) -> list[str]:
    """Return the type checkers' files: ``files`` widened, minus excludes."""
    widened = gate.walk(
        context,
        files=files,
        dependents=dependents,
        hops=gate.positive(context.environ.get("PRECOMMIT_CODEGRAPH_TYPECHECK", "")),
        emit="files",
        workers=workers,
    )
    return scope.kept(widened or files, excludes=scope.excludes(context.root))


def _tool(name: str) -> str:
    """Return ``name``'s executable beside this interpreter, else from PATH."""
    beside = Path(sys.executable).parent / name
    return str(beside) if beside.is_file() else (shutil.which(name) or name)
