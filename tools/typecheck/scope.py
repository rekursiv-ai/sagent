"""Decide which files a gate checks.

The excludes are ``[tool.pyrefly] project-excludes`` in ``pyproject.toml``.
ty and pyrefly both check a path they are NAMED even when their own config
excludes it, so a gate that names files must drop excluded ones itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import re
import tomllib


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def excludes(root: Path) -> tuple[str, ...]:
    """Return the project's exclude globs, from ``root/pyproject.toml``.

    Args:
      root: Directory holding the ``pyproject.toml``.

    Returns:
      excludes: ``[tool.pyrefly] project-excludes``; empty without the file.

    """
    path = root / "pyproject.toml"
    if not path.is_file():
        return ()
    # Not ``from_plain``: every export ships this module, and not every export
    # depends on treekle.
    pyproject = cast(_Pyproject, tomllib.loads(path.read_text()))
    return tuple(
        map(
            str,
            pyproject.get("tool", {}).get("pyrefly", {}).get("project-excludes", ()),
        ),
    )


def kept(paths: Iterable[str], *, excludes: Sequence[str]) -> list[str]:
    """Return ``paths`` no glob in ``excludes`` matches, deduplicated, in order.

    Args:
      paths: Repository-relative paths.
      excludes: Globs as ``project-excludes`` spells them.

    Returns:
      kept: The surviving paths.

    """
    patterns = [_glob(pattern) for pattern in excludes]
    return [
        path
        for path in dict.fromkeys(paths)
        if not any(pattern.match(path) for pattern in patterns)
    ]


# Not `glob.translate`: it arrived in 3.13, and exported packages validate on 3.12.
# A trailing `/` names a directory and everything under it, as pyrefly reads it.
def _glob(pattern: str) -> re.Pattern[str]:
    if pattern.endswith("/"):
        pattern += "**"
    wildcards = {"**/": "(?:.*/)?", "**": ".*", "*": "[^/]*", "?": "[^/]"}
    return re.compile(
        "".join(
            wildcards.get(piece[0]) or re.escape(piece[0])
            for piece in re.finditer(r"\*\*/|\*\*|\*|\?|[^*?]+", pattern)
        )
        + r"\Z",
    )


_Pyrefly = TypedDict("_Pyrefly", {"project-excludes": NotRequired[Sequence[str]]})


class _Tool(TypedDict):
    pyrefly: NotRequired[_Pyrefly]


class _Pyproject(TypedDict):
    tool: NotRequired[_Tool]
