"""Shared utilities for ``.sagent/`` directory conventions.

Used by both ``agents_md`` (AGENTS.md discovery) and ``tools.skill``
(skill discovery). Keep this module dependency-light.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import re

import yaml


_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def walk_up(cwd: Path) -> list[Path]:
    """Return [root, ..., parent, cwd] -- ancestors root-first.

    Args:
      cwd: Starting directory.

    Returns:
      ancestors: List from filesystem root down to ``cwd``.

    """
    resolved = cwd.resolve()
    parts = list(resolved.parents)
    parts.reverse()
    parts.append(resolved)
    return parts


def parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """Strip YAML frontmatter, return (metadata_dict, body).

    On parse error or missing frontmatter, returns ``({}, raw)``.

    Args:
      raw: Raw file content potentially prefixed with YAML frontmatter.

    Returns:
      metadata: Parsed YAML frontmatter dict, or empty dict if absent/invalid.
      body: File body with frontmatter stripped.

    """
    m = _FRONTMATTER_RE.match(raw)
    if m is None:
        return {}, raw
    body = raw[m.end() :]
    try:
        parsed = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return {}, raw
    if not isinstance(parsed, dict):
        return {}, body
    return cast(dict[str, Any], parsed), body
