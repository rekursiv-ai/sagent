"""Memory: save and forget the project's persistent memories.

A memory is ``<type>_<slug>.md`` with frontmatter in the project's memory
directory, and one line of ``MEMORY.md`` indexes it. The tool keeps the two in
step, so the model never edits the index by hand. The index is this tool's
briefing, which the agent reads once a window into the system prompt, so a
save mid-window changes no cached prefix; the save reaches the model as the
call and its result instead.

Typical usage example:

  memory = Memory(directory=project_dir(Path.cwd()) / "memory")
  index = memory.briefing(context)
  result = await memory.call(context, {"action": "forget", "name": "testing"})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

import asyncio
import re

from treekle import ReadError, from_plain, json_dumps

import yaml

from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import parse, schema_of


if TYPE_CHECKING:
    from collections.abc import Mapping

    from treekle import PlainTree

    from sagent.tools.types import ToolContext


__all__ = [
    "Memory",
]


type Kind = Literal[
    "user",
    "feedback",
    "project",
    "reference",
]


@dataclass(slots=True, kw_only=True)
class Memory:
    """Save or forget one persistent memory of this project.

    Args:
      directory: Where the memories and ``MEMORY.md`` live; made on first use.
      max_index_lines: The most index lines the briefing carries.
      max_index_bytes: The most index bytes the briefing carries.

    """

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        action: Annotated[Literal["save", "forget"], "What to do."]
        name: Annotated[str, "Short name; saving it again replaces it."]
        type: Annotated[Kind | None, "Required to save."] = None
        description: Annotated[
            str,
            "One line for relevance matching; required to save.",
        ] = ""
        content: Annotated[str, "The memory itself; required to save."] = ""

    name = "Memory"
    description = (files(__package__) / "assets" / "memory.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)
    kinds: ClassVar[tuple[Kind, ...]] = (
        "user",
        "feedback",
        "project",
        "reference",
    )

    directory: Path
    max_index_lines: int = 200
    max_index_bytes: int = 25_000
    display: ToolDisplay = field(default=ToolDisplay())

    def briefing(self, context: ToolContext) -> str:
        """Return where the memories live, and the index as it is now.

        Args:
          context: The calling agent's state; unused.

        Returns:
          text: The ``MEMORY.md`` section of the window's system prompt.

        """
        del context
        return (
            f"## MEMORY.md\n\nThe memories live at `{self.directory}/`; Read one"
            " when it seems relevant.\n\n"
            + (self._read_index().strip() or "(no memories yet)")
        )

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``Memory <action> <name>``."""
        return (
            f"Memory {arguments.get('action', '')} {arguments.get('name', '')}".strip()
        )

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``MEMORY.md``: every call rewrites it."""
        del context, arguments
        return str(self.directory / "MEMORY.md")

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        """Save or forget the memory ``name``, and its index line.

        Args:
          context: The calling agent's state.
          arguments: What the model passed, as ``Memory.Arguments``.

        Returns:
          result: What was saved or forgotten, or why nothing was.

        """
        del context
        a = parse(arguments, Memory.Arguments)
        try:
            text = await asyncio.to_thread(self._apply, a)
        except ValueError as e:
            return UncategorizedToolResult(call_id="", content=str(e), is_error=True)
        return UncategorizedToolResult(call_id="", content=text)

    def _apply(self, a: Memory.Arguments) -> str:
        """Apply the action to the directory; raise ``ValueError`` for a bad call."""
        # Each memory is one index line and one frontmatter value, so a newline in
        # either splits it; a bracket closes the index link early.
        name = " ".join(a.name.split())
        if "[" in name or "]" in name:
            raise ValueError("name must not contain [ or ].")
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        if not slug:
            raise ValueError("name must contain a letter or digit.")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if a.action == "forget":
            found = self._files(slug)
            if not found:
                raise ValueError(f"No memory named {name!r}.")
            _check_owner(found, name, then="forget it under that name.")
            for path in found:
                path.unlink()
            self._write_index(self._index_lines(without=slug))
            return f"Forgot {', '.join(p.name for p in found)}."
        description = " ".join(a.description.split())
        content = a.content.strip()
        if a.type is None or not description or not content:
            raise ValueError("A save needs a type, a description, and content.")
        stale = self._files(slug)
        _check_owner(stale, name, then="save it under that name, or another.")
        for path in stale:
            path.unlink()
        file = f"{a.type}_{slug}.md"
        (self.directory / file).write_text(
            f"---\nname: {_scalar(name)}\ndescription: {_scalar(description)}\n"
            f"type: {a.type}\n---\n\n"
            f"{content}\n",
            encoding="utf-8",
        )
        # Newest first: the briefing cuts the index at its line cap.
        self._write_index(
            [f"- [{name}]({file}) -- {description}", *self._index_lines(without=slug)],
        )
        return f"Saved {file}."

    # Named exactly, one candidate per type: a glob such as ``*_testing.md`` also
    # matched ``feedback_unit_testing.md``, so forgetting ``testing`` deleted it.
    def _files(self, slug: str) -> list[Path]:
        """Return the memory files named ``slug``, of any type."""
        return [
            path
            for kind in self.kinds
            if (path := self.directory / f"{kind}_{slug}.md").exists()
        ]

    def _index_lines(self, *, without: str) -> list[str]:
        """Return ``MEMORY.md``'s lines, less any pointing at the memory ``without``."""
        index = self.directory / "MEMORY.md"
        if not index.exists():
            return []
        kinds = "|".join(self.kinds)
        pointer = re.compile(rf"\]\((?:{kinds})_{re.escape(without)}\.md\)")
        return [
            line
            for line in index.read_text(encoding="utf-8").splitlines()
            if line.strip() and not pointer.search(line)
        ]

    def _write_index(self, lines: list[str]) -> None:
        """Write ``MEMORY.md`` as ``lines``."""
        (self.directory / "MEMORY.md").write_text(
            "".join(f"{line}\n" for line in lines),
            encoding="utf-8",
        )

    def _read_index(self) -> str:
        """Return ``MEMORY.md``, cut at the line and byte caps; ``""`` for none."""
        try:
            raw = (self.directory / "MEMORY.md").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""
        cut: list[str] = []
        lines = raw.splitlines(keepends=True)
        if len(lines) > self.max_index_lines:
            lines = lines[: self.max_index_lines]
            cut.append(f"over {self.max_index_lines} lines")
        text = "".join(lines)
        data = text.encode()
        if len(data) > self.max_index_bytes:
            data = data[: self.max_index_bytes]
            text = data[: max(0, data.rfind(b"\n"))].decode(errors="replace") + "\n"
            cut.append(f"over {self.max_index_bytes} bytes")
        return text + (f"\n[MEMORY.md truncated: {'; '.join(cut)}]\n" if cut else "")


# Two names can share a slug, ``Foo Bar`` and ``foo-bar``; saving or forgetting
# one by the other's file would erase a memory the model never named.
def _check_owner(paths: list[Path], name: str, *, then: str) -> None:
    """Raise if a file of ``paths`` holds a memory saved under another name."""
    for path in paths:
        other = _stored_name(path)
        if other is not None and other != name:
            raise ValueError(f"{name!r} files as the memory {other!r}; {then}")


def _stored_name(path: Path) -> str | None:
    """Return the ``name`` a memory file's frontmatter gives; ``None`` for none."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = re.match(r"---\n(.*?)\n---\n", raw, re.DOTALL)
    if match is None:
        return None
    try:
        meta = from_plain(yaml.safe_load(match.group(1)), dict[str, object])
    except (yaml.YAMLError, ReadError):
        return None
    name = meta.get("name")
    return name if isinstance(name, str) else None


# A colon or a leading ``#`` changes what a plain YAML value says, and the
# frontmatter is read back as YAML; a JSON string is a YAML string too.
def _scalar(value: str) -> str:
    """Return ``value`` as a YAML scalar that reads back as itself."""
    try:
        plain = yaml.safe_load(f"k: {value}") == {"k": value}
    except yaml.YAMLError:
        plain = False
    return value if plain else json_dumps(value, ensure_ascii=False)
