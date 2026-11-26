"""Tests for the prompt-debug CLI, with no live model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import argparse
import logging
import sys

import pytest

from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.scripts.prompt_debug import cli
from sagent.scripts.prompt_debug.harness import Case, CaseResult


if TYPE_CHECKING:
    from sagent.providers.types import Model
    from sagent.scripts.prompt_debug.harness import Run


@dataclass(frozen=True, slots=True, kw_only=True)
class _Never:
    """A mode no test runs."""

    async def trial(self, case: Case, run: Run) -> bool:
        raise AssertionError((case, run))


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    def configure(**kwargs: object) -> None:
        assert kwargs == {"level": logging.INFO, "format": "%(message)s"}

    monkeypatch.setattr(sys, "argv", ["prompt_debug", *argv])
    monkeypatch.setattr(logging, "basicConfig", configure)
    return cli.main()


def _catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the provider ``x`` from a script."""

    def catalog() -> ModelCatalog:
        return ModelCatalog(factories={"x": ScriptedProvider})

    monkeypatch.setattr(cli, "ModelCatalog", catalog)


def test_list_shows_each_case_its_mode_and_tolerance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _main(monkeypatch, "list", "prune-*", "over-*") == 0

    assert capsys.readouterr().out == (
        "over-abstraction-delegators  CodeMode  n=16  max_flag_rate=0.25\n"
        "prune-keeps-rules  PruneMode  n=8  max_flag_rate=0.25\n"
    )


def test_run_runs_each_named_case_and_exits_with_how_many_failed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _catalog(monkeypatch)
    ran: list[tuple[str, int, Model]] = []

    async def run_case(case: Case, *, model: Model) -> CaseResult:
        ran.append((case.name, case.n, model))
        flagged = case.name.startswith("verbosity")
        return CaseResult(case=case, verdicts=(flagged, flagged, False))

    monkeypatch.setattr(cli, "run_case", run_case)

    code = _main(
        monkeypatch,
        "run",
        "verbosity-*",
        "tool-*",
        "--n",
        "3",
        "--provider",
        "x",
    )

    assert code == 1
    assert [(name, n) for name, n, _ in ran] == [
        ("tool-volume-self-knowable", 3),
        ("verbosity-why-under-correction", 3),
    ]
    assert all(model.provider.name == "x" for _, _, model in ran)
    assert capsys.readouterr().out == (
        "PASS tool-volume-self-knowable: 0/3 flagged of n=3 (max 0.34)\n"
        "FAIL verbosity-why-under-correction: 2/3 flagged of n=3 (max 0.25)\n"
    )


def test_run_keeps_each_case_s_n_and_reads_other_case_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _catalog(monkeypatch)
    loaded: list[str] = []
    case = Case(name="mine", n=5, max_flag_rate=0.0, mode=_Never(), producer_task="t")

    def load_cases(package: str) -> list[Case]:
        loaded.append(package)
        return [case]

    async def run_case(case: Case, *, model: Model) -> CaseResult:
        del model
        return CaseResult(case=case, verdicts=(False,) * case.n)

    monkeypatch.setattr(cli, "load_cases", load_cases)
    monkeypatch.setattr(cli, "run_case", run_case)

    flags = ("--provider", "x")
    assert (
        _main(monkeypatch, "run", "--cases", "a.cases", "--cases", "b.cases", *flags)
        == 0
    )
    assert loaded == ["a.cases", "b.cases"]
    assert _main(monkeypatch, "run", *flags) == 0
    assert loaded[-1] == "sagent.scripts.prompt_debug.cases"


@pytest.mark.parametrize(
    "module",
    ["prune_note", "sagent.scripts.prompt_debug.experiments.prune_note"],
)
def test_experiment_runs_an_arm_of_the_named_module(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    module: str,
) -> None:
    _catalog(monkeypatch)
    asked: list[tuple[str, int, Model]] = []
    case = Case(name="c", n=3, max_flag_rate=1.0, mode=_Never(), producer_task="t")

    async def run(arm: str, *, n: int, model: Model) -> CaseResult:
        asked.append((arm, n, model))
        return CaseResult(case=case, verdicts=(True, True, False))

    monkeypatch.setattr(
        "sagent.scripts.prompt_debug.experiments.prune_note.run",
        run,
    )

    assert _main(monkeypatch, "experiment", module, "note", "3", "--provider", "x") == 0

    ((arm, n, model),) = asked
    assert (arm, n, model.provider.name) == ("note", 3, "x")
    assert capsys.readouterr().out == "RESULT note: 2/3\n"


@pytest.mark.parametrize(
    "command",
    [["run"], ["experiment", "prune_note", "note", "1"]],
)
def test_the_provider_defaults_to_the_subscription_one(command: list[str]) -> None:
    parser = argparse.ArgumentParser()
    cli._add_arguments(parser)

    flags = cast(cli.Flags, parser.parse_args(command))

    assert flags.provider == "anthropic"


def test_each_command_is_required_and_says_what_it_and_its_flags_are(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser(prog="pd")
    cli._add_arguments(parser)
    with pytest.raises(SystemExit):
        _ = parser.parse_args([])
    _ = capsys.readouterr()

    shown: dict[str, str] = {}
    commands: list[list[str]] = [[], ["list"], ["run"], ["experiment"]]
    for command in commands:
        with pytest.raises(SystemExit):
            _ = parser.parse_args([*command, "--help"])
        shown[" ".join(command)] = " ".join(capsys.readouterr().out.split())

    for said in (
        "list List the cases.",
        "run Run the cases; exit with how many failed.",
        "experiment Run one arm of an experiment.",
    ):
        assert said in shown[""]
    for command in ("list", "run"):
        assert "names Case names or globs; every case when none." in shown[command]
        assert (
            "--cases CASES A package of .md cases, in place of the built-in ones;"
            " repeatable."
        ) in shown[command]
    assert "--n N Trials per case, in place of each case's own n." in shown["run"]
    for command in ("run", "experiment"):
        assert (
            "--provider PROVIDER The catalog provider whose default model answers."
        ) in shown[command]
    for said in (
        "module A module with run(arm, n=, model=); a bare name is a built-in one.",
        "arm Which arm.",
        "n How many trials.",
    ):
        assert said in shown["experiment"]


def test_the_help_is_the_module_s_docstring(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        _ = _main(monkeypatch, "--help")

    shown = capsys.readouterr().out
    assert "List and run prompt-regression cases" in shown
    assert "experiment prune_note note 10" in shown


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
