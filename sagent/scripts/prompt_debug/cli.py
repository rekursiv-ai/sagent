"""List and run prompt-regression cases, or run an experiment's arm, on live models.

A case pins a behavior a prompt fix was proven to stop; ``run`` re-runs cases on
the system prompt the sagent CLI assembles and exits with how many failed. Cases
come from the built-in package, or any package of ``.md`` cases named with
``--cases``. ``experiment`` runs one arm of a module's ``run(arm, n=, model=)``;
a bare name is a module of the built-in experiments package.

Examples:
  python -m sagent.scripts.prompt_debug list
  python -m sagent.scripts.prompt_debug run 'verbosity-*' --n 4
  python -m sagent.scripts.prompt_debug experiment prune_note note 10

"""

from __future__ import annotations

from dataclasses import replace
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import asyncio
import importlib
import logging

from sagent.providers.catalog import ModelCatalog
from sagent.scripts.prompt_debug.harness import load_cases, run_case


if TYPE_CHECKING:
    from collections.abc import Awaitable

    from sagent.providers.types import Model
    from sagent.scripts.prompt_debug.harness import Case, CaseResult


__all__ = ["Flags", "main"]


def main() -> int:
    """Run the command the flags name.

    Returns:
      code: The process exit code; for ``run``, how many cases failed.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),  # pragma: no mutate[diagnostic-text]
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if flags.command == "experiment":
        result = asyncio.run(_experiment(flags))
        print(f"RESULT {flags.arm}: {result.flagged}/{len(result.verdicts)}")
        return 0
    cases = [
        case
        for package in flags.cases or [f"{__package__}.cases"]
        for case in load_cases(package)
        if not flags.names or any(fnmatch(case.name, name) for name in flags.names)
    ]
    if flags.command == "list":
        for case in cases:
            print(
                f"{case.name}  {type(case.mode).__name__}  n={case.n}"
                f"  max_flag_rate={case.max_flag_rate:g}",
            )
        return 0
    results = asyncio.run(_run(flags, cases))
    for result in results:
        case = result.case
        print(
            f"{'PASS' if result.passed else 'FAIL'} {case.name}:"
            f" {result.flagged}/{len(result.verdicts)} flagged of n={case.n}"
            f" (max {case.max_flag_rate:g})",
        )
    return sum(not result.passed for result in results)


class Flags(Protocol):
    """Parsed command-line flags."""

    command: str
    names: list[str]
    cases: list[str] | None
    provider: str
    trials: int | None
    n: int
    module: str
    arm: str


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the subcommands and their flags on ``parser``."""
    commands = parser.add_subparsers(dest="command", required=True)
    for name, about in (
        ("list", "List the cases."),
        ("run", "Run the cases; exit with how many failed."),
    ):
        command = commands.add_parser(name, help=about)
        command.add_argument(
            "names",
            nargs="*",
            help="Case names or globs; every case when none.",
        )
        command.add_argument(
            "--cases",
            action="append",
            help="A package of .md cases, in place of the built-in ones; repeatable.",
        )
        if name == "run":
            _add_model(command)
            command.add_argument(
                "--n",
                type=int,
                dest="trials",
                metavar="N",
                help="Trials per case, in place of each case's own n.",
            )
    experiment = commands.add_parser("experiment", help="Run one arm of an experiment.")
    experiment.add_argument(
        "module",
        help="A module with run(arm, n=, model=); a bare name is a built-in one.",
    )
    experiment.add_argument("arm", help="Which arm.")
    experiment.add_argument("n", type=int, help="How many trials.")
    _add_model(experiment)


def _add_model(parser: argparse.ArgumentParser) -> None:
    """Register the provider flag on ``parser``."""
    parser.add_argument(
        "--provider",
        default="anthropic",
        help="The catalog provider whose default model answers.",
    )


async def _run(flags: Flags, cases: list[Case]) -> list[CaseResult]:
    """Run each case, in turn, on the flagged provider's default model."""
    async with ModelCatalog() as catalog:
        model = catalog[flags.provider]["default"]
        return [
            await run_case(
                case if flags.trials is None else replace(case, n=flags.trials),
                model=model,
            )
            for case in cases
        ]


async def _experiment(flags: Flags) -> CaseResult:
    """Run the flagged arm of the flagged experiment module."""
    name = flags.module
    package = __package__ or ""
    module = importlib.import_module(
        name if "." in name else f"{package}.experiments.{name}",
    )
    run = cast(_Experiment, module.run)
    async with ModelCatalog() as catalog:
        return await run(flags.arm, n=flags.n, model=catalog[flags.provider]["default"])


class _Experiment(Protocol):
    """An experiment module's ``run``."""

    def __call__(self, arm: str, *, n: int, model: Model) -> Awaitable[CaseResult]:
        """Run ``n`` trials of ``arm`` on ``model``."""
        ...
