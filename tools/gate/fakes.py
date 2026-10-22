"""Stand-ins for the processes a gate runs, shared by the gates' tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import os
import re
import subprocess
import tempfile

from tools.gate import gate


if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest


class Recorder:
    """Stands in for running tools: records each argv, returns canned results."""

    def __init__(
        self,
        status: int = 0,
        output: str | None = None,
        *,
        installed: bool = True,
        on_path: bool = True,
        statuses: Sequence[int] = (),
    ) -> None:
        self.status = status
        self.statuses = list(statuses)
        self.output = output
        self.installed = installed
        self.on_path = on_path
        self.calls: list[list[str]] = []
        self.argfiles: list[list[str]] = []

    def run(self, argv: Sequence[str]) -> int:
        """Record ``argv`` and each ``@argfile`` it names; return the next status.

        Args:
          argv: The command a gate ran.

        Returns:
          status: The next of ``statuses``, else ``status``.

        """
        self.calls.append(list(argv))
        self.argfiles.extend(
            Path(arg[1:]).read_text().splitlines()
            for arg in argv
            if arg.startswith("@")
        )
        return self.statuses.pop(0) if self.statuses else self.status

    def read(self, argv: Sequence[str]) -> str | None:
        """Record ``argv``; return the canned stdout."""
        self.calls.append(list(argv))
        return self.output

    def has(self, module: str) -> bool:
        """Report every module as installed or none."""
        del module
        return self.installed

    def which(self, name: str) -> str | None:
        """Place every tool under ``/bin``, or none on PATH."""
        return f"/bin/{name}" if self.on_path else None


def context(tmp_path: Path, tools: Recorder, **environ: str) -> gate.Context:
    """Return a context running ``tools`` in ``tmp_path/repo``, spooling to key 9.

    Args:
      tmp_path: The test's directory.
      tools: The recorder every run goes through.
      **environ: Variables set on top of ``WORKERS_DEFAULT=3``.

    Returns:
      context: The gate context.

    """
    root = tmp_path / "repo"
    root.mkdir(exist_ok=True)
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text() if pyproject.is_file() else ""
    if "[tool.pyrefly]" not in text:
        pyproject.write_text(
            f'{text}[tool.pyrefly]\nproject-excludes = ["**/third_party/"]\n',
        )
    return gate.Context(
        directory=tmp_path / "spool",
        key=9,
        root=root,
        environ={"WORKERS_DEFAULT": "3", **environ},
        run=tools.run,
        output=tools.read,
        installed=tools.has,
        which=tools.which,
    )


class Processes:
    """Stands in for ``subprocess.run``: records argvs, answers ``status``/``stdout``."""

    def __init__(self, status: int, stdout: str) -> None:
        self.status = status
        self.stdout = stdout
        self.calls: list[list[str]] = []
        self.niceness: list[int] = []

    def nice(self, increment: int) -> int:
        """Record a priority change in place of ``os.nice``."""
        self.niceness.append(increment)
        return 0

    def __call__(
        self,
        argv: Sequence[str],
        **_: object,
    ) -> subprocess.CompletedProcess[str]:
        """Record ``argv``; answer with the canned status and stdout."""
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, self.status, stdout=self.stdout)


def isolate(tmp_path: Path, *, monkeypatch: pytest.MonkeyPatch) -> Processes:
    """Run a gate's ``main`` in ``tmp_path``, every child process recorded.

    Drops each variable a gate reads, so the run sees one worker and no opt-in
    whatever gate the test suite itself runs under.

    Args:
      tmp_path: The test's directory, made the cwd and ``TMPDIR``.
      monkeypatch: Restores everything this changes.

    Returns:
      recorded: What ``main`` ran.

    """
    for name in list(os.environ):
        if re.match(r"(WORKERS|PYTEST|PRECOMMIT|SIDECAR)_|GITHUB_ACTIONS$", name):
            monkeypatch.delenv(name)
    recorded = Processes(status=4, stdout="")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("WORKERS_DEFAULT", "1")
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setattr(os, "nice", recorded.nice)
    monkeypatch.setattr(subprocess, "run", recorded)
    return recorded
