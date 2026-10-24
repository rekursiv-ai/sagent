"""Memory tool: save and forget persistent project memories.

A memory is ``<type>_<name>.md`` with frontmatter in the project's memory
directory, and one line of ``MEMORY.md`` indexes it. The tool keeps the two in
step, so the model never edits the index by hand. The index the model reads
sits in the project block of the system prompt, built once per context; a save
or a forget reaches the model as this tool's call and result instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import math
import re

from treekle import PlainTree, immutable

import yaml

from sagent.agent.state import get_tool_state
from sagent.lib.dotsagent import parse_frontmatter
from sagent.memory import ensure_memory_dir
from sagent.tools.core import load_tool_description, run_sync
from sagent.types.runtime import ToolResult


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


_TYPES: Final = ("user", "feedback", "project", "reference")


class Memory:
    """Tool: save or forget one persistent memory of this project."""

    name: str = "Memory"
    tool_id: str = "application/x-tool-memory"
    clearable_results: bool = True
    description: str = load_tool_description("Memory")
    directive_schema: Mapping[str, PlainTree] = immutable(
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["save", "forget"]},
                "name": {
                    "type": "string",
                    "description": "Short name; saving it again replaces it.",
                },
                "type": {
                    "type": "string",
                    "enum": list(_TYPES),
                    "description": "Required to save.",
                },
                "description": {
                    "type": "string",
                    "description": "One line for relevance matching; required to save.",
                },
                "content": {
                    "type": "string",
                    "description": "The memory itself; required to save.",
                },
            },
            "required": ["action", "name"],
            "additionalProperties": False,
        },
    )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return ``Memory <action> <name>`` for the pending call."""
        return f"Memory {args.get('action', '')} {args.get('name', '')}".strip()

    def prompt(self) -> str:
        """Return nothing; the guidance is the description, the index the project's."""
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Serialize every call: each one rewrites ``MEMORY.md``."""
        del args
        return "MEMORY.md"

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Save or forget the memory ``args`` names.

        Args:
          args: Directive with ``action`` and ``name``; a save also carries
            ``type``, ``description``, and ``content``.

        Returns:
          result: What was saved or forgotten, or why nothing was.

        """
        # The index the model reads was built for the project directory, so a save
        # after a ``cd`` still lands beside it.
        state = get_tool_state()
        directory = ensure_memory_dir(state.project_cwd or state.bash_cwd)
        return await run_sync(self._run, directory=directory, args=args)

    def _run(self, *, directory: Path, args: Mapping[str, object]) -> str | ToolResult:
        """Apply the action to ``directory``."""
        # Each memory is one index line and one frontmatter value, so a newline in
        # either splits it; a bracket closes the index link early.
        name = " ".join(str(args.get("name", "")).split())
        if "[" in name or "]" in name:
            return _error("name must not contain [ or ].")
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        if not slug:
            return _error("name must contain a letter or digit.")
        # Names that differ only in punctuation or case share a file.
        if (taken := _name_on_file(directory, slug)) not in (None, name):
            return _error(f"{name!r} files as {taken!r}; pick another name.")
        if args.get("action") == "forget":
            return _forget(directory, name=name, slug=slug)
        kind = args.get("type")
        if kind not in _TYPES:
            return _error(f"type must be one of {', '.join(_TYPES)}.")
        description = " ".join(str(args.get("description", "")).split())
        content = str(args.get("content", "")).strip()
        if not description or not content:
            return _error("a save needs a description and content.")
        for stale in _files(directory, slug):
            stale.unlink()
        file = f"{kind}_{slug}.md"
        meta = yaml.safe_dump(
            {"name": name, "description": description, "type": kind},
            sort_keys=False,
            allow_unicode=True,
            width=math.inf,
        )
        (directory / file).write_text(
            f"---\n{meta}---\n\n{content}\n",
            encoding="utf-8",
        )
        # Newest first: the model reads the index cut at its line cap from the end.
        _write_index(
            directory,
            [
                f"- [{name}]({file}) -- {description}",
                *_index_lines(directory, without=slug),
            ],
        )
        return f"Saved {file}."


def _forget(directory: Path, *, name: str, slug: str) -> str | ToolResult:
    """Delete the memory ``slug`` and its index line."""
    files = _files(directory, slug)
    if not files:
        return _error(f"No memory named {name!r}.")
    for file in files:
        file.unlink()
    _write_index(directory, _index_lines(directory, without=slug))
    return f"Forgot {', '.join(f.name for f in files)}."


# Named exactly, one candidate per type: a glob such as ``*_testing.md`` also matched
# ``feedback_unit_testing.md``, so saving or forgetting ``testing`` deleted it.
def _files(directory: Path, slug: str) -> list[Path]:
    """Return the memory files named ``slug``, of any type."""
    return [
        path for kind in _TYPES if (path := directory / f"{kind}_{slug}.md").exists()
    ]


def _name_on_file(directory: Path, slug: str) -> str | None:
    """Return the name the memory filed as ``slug`` was saved under, if any."""
    for path in _files(directory, slug):
        meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        name = meta.get("name")
        if isinstance(name, str):
            return name
    return None


def _index_lines(directory: Path, *, without: str) -> list[str]:
    """Return ``MEMORY.md``'s lines, less any pointing at the memory ``without``."""
    index = directory / "MEMORY.md"
    if not index.exists():
        return []
    pointer = re.compile(rf"\]\((?:{'|'.join(_TYPES)})_{re.escape(without)}\.md\)")
    return [
        line
        for line in index.read_text(encoding="utf-8").splitlines()
        if line.strip() and not pointer.search(line)
    ]


def _write_index(directory: Path, lines: list[str]) -> None:
    """Write ``MEMORY.md`` as ``lines``."""
    (directory / "MEMORY.md").write_text(
        "".join(f"{line}\n" for line in lines),
        encoding="utf-8",
    )


def _error(text: str) -> ToolResult:
    return ToolResult(call_id="", content=text, is_error=True)
