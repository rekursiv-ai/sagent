"""Tests for the stand-ins the gate tests run tools through."""

from __future__ import annotations

from typing import TYPE_CHECKING

import os
import subprocess
import tempfile

import pytest

from tools.gate import fakes


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "name",
    [
        "WORKERS_TOOL",
        "WORKERS_PYTEST_GLOBAL",
        "PYTEST_SPLIT_ARGS",
        "PYTEST_NODEIDS_FILE",
        "PRECOMMIT_CODEGRAPH_PYTEST",
        "SIDECAR_XDIST_MIN",
        "GITHUB_ACTIONS",
    ],
)
def test_isolate_drops_every_variable_a_gate_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """CI runs these tests inside a gate, whose own settings must not reach them."""
    monkeypatch.setenv(name, "7")

    fakes.isolate(tmp_path, monkeypatch=monkeypatch)

    assert name not in os.environ
    assert os.environ["WORKERS_DEFAULT"] == "1"


def test_isolate_keeps_unrelated_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME_LIKE", "kept")

    fakes.isolate(tmp_path, monkeypatch=monkeypatch)

    assert os.environ["HOME_LIKE"] == "kept"


def test_isolate_records_every_process_and_priority_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = fakes.isolate(tmp_path, monkeypatch=monkeypatch)
    recorded.stdout = "out"

    done = subprocess.run(["tool", "a"], check=False)  # noqa: S607 -- recorded, never run.

    assert (done.returncode, done.stdout) == (4, "out")
    assert recorded.calls == [["tool", "a"]]
    assert os.nice(19) == 0
    assert recorded.niceness == [19]
    assert tempfile.gettempdir() == str(tmp_path)


def test_a_recorder_answers_its_canned_results_in_order(tmp_path: Path) -> None:
    (tmp_path / "args").write_text("a\nb\n")
    tools = fakes.Recorder(status=3, output="x\n", statuses=[1], on_path=False)

    assert tools.run(["tool", f"@{tmp_path / 'args'}"]) == 1
    assert tools.run(["tool"]) == 3
    assert tools.read(["walk"]) == "x\n"
    assert tools.has("anything")
    assert tools.which("tool") is None
    assert fakes.Recorder().which("tool") == "/bin/tool"
    assert tools.calls == [["tool", f"@{tmp_path / 'args'}"], ["tool"], ["walk"]]
    assert tools.argfiles == [["a", "b"]]


def test_a_context_runs_through_its_recorder_in_a_project_root(tmp_path: Path) -> None:
    tools = fakes.Recorder(installed=False)
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "pyproject.toml").write_text('[project]\nname = "x"\n')

    context = fakes.context(tmp_path, tools, EXTRA="1")
    fakes.context(tmp_path, tools)

    assert dict(context.environ) == {"WORKERS_DEFAULT": "3", "EXTRA": "1"}
    assert (context.directory, context.key) == (tmp_path / "spool", 9)
    assert (context.root / "pyproject.toml").read_text().count("[tool.pyrefly]") == 1
    assert context.run(["tool"]) == 0
    assert not context.installed("pytest_cov")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
