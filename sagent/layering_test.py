"""Tests that only the terminal frontend imports it, so ``Agent`` runs without it."""

from __future__ import annotations

from importlib.util import resolve_name
from pathlib import Path
from typing import Final

import ast

import pytest


_CWD: Final = Path(__file__).resolve().parent

_ROOT = _CWD
_FRONTEND = "sagent.tui"


def _imports(path: Path) -> set[str]:
    """Return every module ``path`` imports, by its full name."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                package = ".".join(("sagent", *path.relative_to(_ROOT).parts[:-1]))
                module = resolve_name("." * node.level + module, package)
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def _outside_the_frontend() -> list[Path]:
    """Return the package's modules but the frontend's and the entry point."""
    return [
        path
        for path in sorted(_ROOT.rglob("*.py"))
        if "tui" not in path.relative_to(_ROOT).parts[:1]
        and path.name != "__main__.py"
        and not path.name.endswith("_test.py")
    ]


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(path, id=str(path.relative_to(_ROOT)))
        for path in _outside_the_frontend()
    ],
)
def test_nothing_but_the_frontend_and_the_entry_point_imports_the_frontend(
    path: Path,
) -> None:
    assert not {
        name
        for name in _imports(path)
        if name == _FRONTEND or name.startswith(f"{_FRONTEND}.")
    }


@pytest.mark.parametrize(
    "source",
    [
        "import sagent.tui.console",
        "from sagent.tui.console import ConsolePrinter",
        "from sagent.tui import console",
        "from sagent import tui",
        "from ..tui import console",
        "from .. import tui",
    ],
)
def test_the_check_sees_an_import_of_the_frontend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    monkeypatch.setattr(__name__ + "._ROOT", tmp_path)
    module = tmp_path / "tools" / "leak.py"
    module.parent.mkdir()
    module.write_text(source + "\n")
    assert any(
        name == _FRONTEND or name.startswith(f"{_FRONTEND}.")
        for name in _imports(module)
    )


@pytest.mark.parametrize(
    "name",
    ["agent/agent.py", "tools/bash.py", "runtime/runtime.py"],
)
def test_the_check_covers_the_core(name: str) -> None:
    assert _ROOT / name in _outside_the_frontend()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
