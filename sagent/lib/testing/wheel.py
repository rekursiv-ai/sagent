"""Build a package's wheel and run code against it, outside the checkout.

A package's wheel tests use it::

    from sagent.lib.testing.wheel import built_wheel

    pytestmark = pytest.mark.cli_uv


    def test_the_wheel_imports_outside_the_checkout() -> None:
        built_wheel("sagent", anchor=__file__).run("import sagent")

``cli_uv`` puts the tests in the ``ci_smoke`` tier, which every package's
pre-push stage runs. A checkout whose project is not the package, as the
monorepo's is not, builds no such wheel, so there the tests skip; the
package's public CI is where they run.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import cast

import re
import subprocess
import tempfile
import tomllib
import zipfile

import pytest


@dataclass(frozen=True, slots=True, kw_only=True)
class Wheel:
    """One built wheel, and the project that built it.

    Attributes:
      path: The ``.whl`` file.
      project: The directory holding the project's ``pyproject.toml``.

    """

    path: Path
    project: Path

    def names(self) -> frozenset[str]:
        """Return every path the wheel holds."""
        with zipfile.ZipFile(self.path) as archive:
            return frozenset(archive.namelist())

    def read(self, name: str) -> str:
        """Return the text of the wheel's file ``name``."""
        with zipfile.ZipFile(self.path) as archive:
            return archive.read(name).decode()

    def run(self, code: str, *, extras: tuple[str, ...] = ()) -> str:
        """Run ``code`` where only this wheel and its dependencies are installed.

        Args:
          code: Python source for ``python -c``.
          extras: Extras of the wheel to install with it.

        Returns:
          stdout: What ``code`` printed.

        """
        return subprocess.run(  # noqa: S603 -- Fixed uv argv; code is the test's own.
            self.argv(code, extras=extras),
            # Outside the checkout, so nothing resolves from the source tree.
            cwd=tempfile.gettempdir(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def argv(self, code: str, *, extras: tuple[str, ...] = ()) -> list[str]:
        """Return the command ``run`` executes.

        Args:
          code: Python source for ``python -c``.
          extras: Extras of the wheel to install with it.

        Returns:
          argv: The ``uv run`` command line.

        """
        sources = _table(_pyproject(self.project), "tool", "uv", "sources")
        return [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            # Torch resolves from its CPU index, as the project's lock does; PyPI's
            # linux torch is the CUDA build, which fills a runner's disk.
            "--index",
            "https://download.pytorch.org/whl/cpu",
            "--index-strategy",
            "unsafe-best-match",
            "--with",
            f"{self.path}[{','.join(extras)}]" if extras else str(self.path),
            # A sibling the package tracks by git may not be on PyPI yet.
            *(
                part
                for name, source in sources.items()
                if isinstance(source, dict) and "git" in source
                for part in (
                    "--with",
                    f"{name} @ git+{source['git']}"
                    + (f"@{source['branch']}" if "branch" in source else ""),
                )
            ),
            "python",
            "-P",
            "-c",
            code,
        ]


def built_wheel(distribution: str, *, anchor: str | Path) -> Wheel:
    """Return ``distribution``'s wheel, built once per process.

    Args:
      distribution: The project name the wheel is built from.
      anchor: A file inside the project, usually the calling test's ``__file__``.

    Returns:
      wheel: The built wheel.

    Raises:
      pytest.skip.Exception: No project above ``anchor`` builds ``distribution``.

    """
    project = project_root(distribution, anchor=Path(anchor))
    if project is None:
        pytest.skip(f"{distribution} builds from its own project, not this checkout")
    return _build(project)


def project_root(distribution: str, *, anchor: Path) -> Path | None:
    """Return the project above ``anchor``, if that project is ``distribution``.

    Args:
      distribution: The project name to look for.
      anchor: A file inside the project.

    Returns:
      root: The directory holding the nearest ``pyproject.toml`` above
        ``anchor``, or None when that project has another name.

    """
    for directory in anchor.resolve().parents:
        if (directory / "pyproject.toml").is_file():
            name = str(_table(_pyproject(directory), "project").get("name", ""))
            return directory if _normalized(name) == _normalized(distribution) else None
    return None


@cache
def _build(project: Path) -> Wheel:
    """Build ``project``'s wheel into a fresh directory."""
    out_dir = Path(tempfile.mkdtemp(prefix="wheel-"))
    subprocess.run(  # noqa: S603 -- Fixed uv argv over a project directory.
        ["uv", "build", "--wheel", "--out-dir", str(out_dir), str(project)],  # noqa: S607 -- uv from PATH, as every gate runs it.
        check=True,
        capture_output=True,
    )
    (path,) = out_dir.glob("*.whl")
    return Wheel(path=path, project=project)


def _normalized(name: str) -> str:
    """Return a distribution name as packaging compares it."""
    return re.sub(r"[-_.]+", "-", name).casefold()


# Standard library only, like ``resources``: every package ships this module, and
# some (typings) depend on nothing that could read TOML into types.
def _pyproject(project: Path) -> dict[str, object]:
    """Return ``project``'s ``pyproject.toml``."""
    # TOML is UTF-8 by spec; bytes skip the locale.
    return cast(
        dict[str, object],
        tomllib.loads((project / "pyproject.toml").read_bytes().decode()),
    )


def _table(document: dict[str, object], *keys: str) -> dict[str, object]:
    """Return the table at ``keys``, or an empty one where any is absent."""
    table: object = document
    for key in keys:
        table = (
            cast(dict[str, object], table).get(key) if isinstance(table, dict) else None
        )
    return cast(dict[str, object], table) if isinstance(table, dict) else {}
