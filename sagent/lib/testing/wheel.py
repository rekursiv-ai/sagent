"""Build a package's wheel and run code against it, outside the checkout.

``wheel_test.py`` ships beside this module to the root of every exported
package and uses it to test that package's own wheel. A test that needs the
wheel for something more uses it too::

    from sagent.lib.testing.wheel import built_wheel, project_root

    wheel = built_wheel(project_root(anchor=Path(__file__)))
    wheel.run("import mypkg")

Mark such a test ``cli_uv``: that puts it in the ``ci_smoke`` tier every
package's pre-push stage runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from email import parser
from fnmatch import fnmatch
from functools import cache
from pathlib import Path
from typing import Final, cast, override

import configparser
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

    def run(self, code: str) -> str:
        """Run ``code`` where only this wheel and its dependencies are installed.

        Args:
          code: Python source for ``python -c``.

        Returns:
          stdout: What ``code`` printed.

        """
        return subprocess.run(  # noqa: S603 -- Fixed uv argv; code is the test's own.
            self.argv(code),
            # Outside the checkout, so nothing resolves from the source tree.
            cwd=tempfile.gettempdir(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def argv(self, code: str) -> list[str]:
        """Return the command ``run`` executes.

        Args:
          code: Python source for ``python -c``.

        Returns:
          argv: The ``uv run`` command line.

        """
        pyproject = _pyproject(self.project)
        indexes = [
            url
            for index in cast(
                list[object],
                _table(pyproject, "tool", "uv").get("index", []),
            )
            if isinstance(index, dict)
            and isinstance(url := cast(dict[str, object], index).get("url"), str)
        ]
        return [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            # The project's own indexes, as its lock resolves: trackinizer pins torch
            # to the CPU index, and PyPI's linux torch is the CUDA build.
            *(part for url in indexes for part in ("--index", url)),
            *(("--index-strategy", "unsafe-best-match") if indexes else ()),
            "--with",
            str(self.path),
            # A sibling the package tracks by git may not be on PyPI yet.
            *(
                part
                for name, source in _table(pyproject, "tool", "uv", "sources").items()
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


def built_wheel(project: Path | None) -> Wheel:
    """Return ``project``'s wheel, built once per process.

    Args:
      project: The project root, as ``project_root`` returns it.

    Returns:
      wheel: The built wheel.

    Raises:
      pytest.skip.Exception: ``project`` publishes no wheel.

    """
    if project is None:
        pytest.skip("this checkout's project is not a published package")
    return _build(project)


def project_root(*, anchor: Path) -> Path | None:
    """Return the nearest project above ``anchor``, if it publishes a wheel.

    Args:
      anchor: A file inside the project, usually the calling test's ``__file__``.

    Returns:
      root: The directory holding the nearest ``pyproject.toml`` above
        ``anchor``; None when there is none, it declares no build system, or
        it is the monorepo, whose wheel nobody installs.

    """
    for directory in anchor.resolve().parents:
        if (directory / "pyproject.toml").is_file():
            pyproject = _pyproject(directory)
            published = (
                "build-system" in pyproject
                and _table(pyproject, "project").get("name") != _MONOREPO
            )
            return directory if published else None
    return None


# The monorepo builds a wheel of everything, which takes over a minute and is
# installed by no one; every package this module ships to has another name.
_MONOREPO: Final = "loop"


def top_level_modules(wheel: Wheel) -> tuple[str, ...]:
    """Return the importable top-level modules ``wheel`` ships, sorted.

    Stub trees (``*-stubs``, ``.pyi``) are for type checkers, not import, and
    ``*.dist-info`` and ``*.data`` are install metadata.

    Args:
      wheel: The wheel to read.

    Returns:
      modules: Top-level package and module names.

    """
    tops = {
        name.split("/")[0].removesuffix(".py")
        for name in wheel.names()
        if name.endswith(".py")
    }
    return tuple(
        sorted(
            top for top in tops if not top.endswith(("-stubs", ".dist-info", ".data"))
        ),
    )


def missing_artifacts(wheel: Wheel) -> tuple[str, ...]:
    """Return the wheel's declared ``artifacts`` patterns that match no file in it.

    Args:
      wheel: The wheel to read.

    Returns:
      missing: ``[tool.hatch.build.targets.wheel].artifacts`` entries, in order,
        that the build never made.

    """
    patterns = _table(
        _pyproject(wheel.project),
        "tool",
        "hatch",
        "build",
        "targets",
        "wheel",
    ).get("artifacts", [])
    names = wheel.names()
    return tuple(
        pattern
        for pattern in cast(list[str], patterns)
        if not any(fnmatch(name, pattern) for name in names)
    )


def declared_extras(wheel: Wheel) -> tuple[str, ...]:
    """Return the extras ``wheel``'s metadata declares, in order.

    Args:
      wheel: The wheel to read.

    Returns:
      extras: Its ``Provides-Extra`` names.

    """
    names = [n for n in wheel.names() if n.endswith(".dist-info/METADATA")]
    if not names:
        return ()
    metadata = parser.Parser().parsestr(wheel.read(names[0]), headersonly=True)
    return tuple(metadata.get_all("Provides-Extra") or ())


def console_scripts(wheel: Wheel) -> dict[str, str]:
    """Return ``wheel``'s console scripts, each name to its ``module:attr``.

    Args:
      wheel: The wheel to read.

    Returns:
      scripts: The ``[console_scripts]`` of its ``entry_points.txt``.

    """
    names = [n for n in wheel.names() if n.endswith(".dist-info/entry_points.txt")]
    if not names:
        return {}
    parser = _CaseSensitiveParser(delimiters=("=",), interpolation=None)
    parser.read_string(wheel.read(names[0]))
    if not parser.has_section("console_scripts"):
        return {}
    return dict(parser.items("console_scripts"))


class _CaseSensitiveParser(configparser.ConfigParser):
    """``ConfigParser`` that keeps script names' case, as the wheel does."""

    @override
    def optionxform(self, optionstr: str) -> str:
        return optionstr


@cache
def _build(project: Path) -> Wheel:
    """Build ``project``'s wheel into a fresh directory."""
    out_dir = Path(tempfile.mkdtemp(prefix="wheel-"))
    done = subprocess.run(  # noqa: S603 -- Fixed uv argv over a project directory.
        ["uv", "build", "--wheel", "--out-dir", str(out_dir), str(project)],  # noqa: S607 -- uv from PATH, as every gate runs it.
        check=False,
        capture_output=True,
        text=True,
    )
    # A CI log keeps only the exception, so it carries what the build said.
    if done.returncode:
        raise RuntimeError(
            f"uv build of {project} failed (exit {done.returncode}):\n{done.stderr}",
        )
    (path,) = out_dir.glob("*.whl")
    return Wheel(path=path, project=project)


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
    table = document
    for key in keys:
        child = table.get(key)
        if not isinstance(child, dict):
            return {}
        table = cast(dict[str, object], child)
    return table
