"""Tests that only the terminal frontend imports it, so ``Agent`` runs without it."""

from __future__ import annotations

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
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and not node.level
        ):
            found.add(node.module)
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


def test_nothing_but_the_frontend_and_the_entry_point_imports_the_frontend() -> None:
    importers = [
        str(path.relative_to(_ROOT))
        for path in _outside_the_frontend()
        if any(
            name == _FRONTEND or name.startswith(f"{_FRONTEND}.")
            for name in _imports(path)
        )
    ]
    assert importers == []


def test_the_check_sees_an_import_of_the_frontend(tmp_path: Path) -> None:
    module = tmp_path / "leak.py"
    module.write_text("from sagent.tui.console import ConsolePrinter\n")
    assert _FRONTEND + ".console" in _imports(module)


@pytest.mark.parametrize(
    "name",
    ["agent/agent.py", "tools/bash.py", "runtime/runtime.py"],
)
def test_the_check_covers_the_core(name: str) -> None:
    assert _ROOT / name in _outside_the_frontend()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
