"""Edit tool: exact-string replacement in existing files.

Unlike Write, Edit intentionally does not require a prior Read. The
``old_string`` exact match is the operation's safety gate, which keeps
sed-like replacements cheap when the caller already knows the target text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import asyncio
import difflib
import re

from sagent.lib.atomic_file import atomic_write_bytes
from sagent.lib.custom_json import JSON, bool_val, json_freeze
from sagent.tools.core import (
    file_lock_key,
    get_file_write_lock,
    get_tool_state,
    load_tool_description,
    resolve_tool_path,
)
from sagent.tools.lib.bash import Node, unwrap_cd_prefix
from sagent.types.runtime import ToolResult


# Simple ``s/OLD/NEW/[g]`` - no escaped delimiters, no alternate
# delimiter, no address ranges. If the sed script is more complex
# than this, we bail.
_SED_S_PATTERN = re.compile(r"^s/([^/\\]*)/([^/\\]*)/([gi]*)$")

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)((?:,\d+)?) \+(\d+)((?:,\d+)?) @@")


def make_diff(old: str, new: str, offset: int) -> str:
    """Produce a unified diff with absolute line numbers.

    Args:
      old: Original text fragment.
      new: Replacement text fragment.
      offset: Line offset (0-based) within the containing file at which
        ``old`` begins, added to each hunk header so the diff reflects
        absolute file line numbers rather than fragment-relative ones.

    Returns:
      diff: Unified-diff string with ``---``/``+++`` headers stripped.

    """
    lines = list(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            lineterm="",
            n=3,
        )
    )[2:]  # skip --- / +++ headers
    result: list[str] = []
    for line in lines:
        m = _HUNK_HEADER_RE.match(line)
        if m:
            old_start = int(m.group(1)) + offset
            new_start = int(m.group(3)) + offset
            result.append(f"@@ -{old_start}{m.group(2)} +{new_start}{m.group(4)} @@")
        else:
            result.append(line)
    return "\n".join(result)


_NUDGE: Final = "sed via Bash is a bad UX. Use the Edit tool."


class Edit:
    """Perform exact-string replacement in files.

    This tool deliberately permits unread files. The exact ``old_string`` match
    is treated as the safety check; stale-file checks apply when prior session
    state exists, but lack of a prior Read is not itself an error.
    """

    name: str = "Edit"
    tool_id: str = "application/x-tool-edit"
    clearable_results: bool = False
    description: str = load_tool_description("Edit")
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file."},
                "old_string": {"type": "string", "description": "Text to find."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences.",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }
    )

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Apply an exact-string replacement to a file.

        Args:
          args: Directive with ``file_path``, ``old_string``, ``new_string``,
              and optional ``replace_all``.

        Returns:
          result: Replacement confirmation with a unified diff, or an
              error when validation fails or no match is found.

        """
        file_path = resolve_tool_path(str(args.get("file_path", "")))
        old_string = str(args.get("old_string", ""))
        new_string = str(args.get("new_string", ""))
        replace_all = bool_val(args.get("replace_all"), False)
        # Serialize against any other mutating tool on the same file.
        async with get_file_write_lock(file_path):
            return await asyncio.to_thread(
                self._run,
                file_path,
                old_string,
                new_string,
                replace_all,
            )

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Serialize same-file Read/Edit/Write within a cohort.

        Args:
          args: Directive carrying ``file_path``.

        Returns:
          key: Canonical path, or ``None`` when no path was supplied.

        """
        path = resolve_tool_path(str(args.get("file_path", "")))
        return file_lock_key(path) if path else None

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label summarizing this edit.

        Args:
          args: Directive carrying ``file_path``.

        Returns:
          label: ``Edit <basename>`` line shown before invocation.

        """
        file_path = resolve_tool_path(str(args.get("file_path", "")))
        fname = Path(file_path).name if file_path else "?"
        return f"Edit {fname}"

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for Edit.

        Args:
          result: Completed ``ToolResult`` (ignored).

        Returns:
          receipt: Always ``None`` (no receipt line).

        """
        del result
        return None

    def prompt(self) -> str:
        """Return no supplemental system-prompt text for Edit.

        Returns:
          contribution: Empty string.

        """
        return ""

    def _validate(self, file_path: str, old_string: str) -> ToolResult | None:
        """Return an error ``ToolResult`` on invalid inputs, else ``None``."""
        if not old_string:
            # text.count("") returns len+1; text.replace("", …) inserts
            # between every char. Silently destroys the file - reject.
            return ToolResult(
                call_id="", content="old_string cannot be empty.", is_error=True
            )
        p = Path(file_path)
        if not p.exists():
            return ToolResult(
                call_id="", content=f"File not found: {file_path}", is_error=True
            )
        if p.is_dir():
            return ToolResult(
                call_id="",
                content=f"{file_path} is a directory, not a file.",
                is_error=True,
            )
        state = get_tool_state()
        if state.check_stale(file_path):
            return ToolResult(
                call_id="",
                content=(
                    "File has been modified since read, either by the user, a"
                    " linter, or another agent. Read it again before attempting"
                    " to edit it."
                ),
                is_error=True,
            )
        return None

    def _run(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool,
    ) -> ToolResult:
        """Run the validated edit synchronously and return the result."""
        err = self._validate(file_path, old_string)
        if err is not None:
            return err
        p = Path(file_path)
        state = get_tool_state()
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            size = p.stat().st_size
            return ToolResult(
                call_id="",
                content=(
                    f"[Non-UTF-8 file: {file_path} ({size} bytes)."
                    " Use Bash with an explicit decoder to inspect or rewrite.]"
                ),
                is_error=True,
            )
        count = text.count(old_string)
        if count == 0:
            return ToolResult(
                call_id="", content="old_string not found in file.", is_error=True
            )
        if count > 1 and not replace_all:
            return ToolResult(
                call_id="",
                content=(
                    f"old_string found {count} times."
                    f" Use replace_all=True or provide more context."
                ),
                is_error=True,
            )
        new_text = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )
        # Atomic write (tmp + rename). Forward the original file mode so
        # Edit doesn't silently flip a ``0o600`` file to ``0o644``.
        file_mode = p.stat().st_mode & 0o777
        atomic_write_bytes(p, new_text.encode("utf-8"), file_mode=file_mode)
        state.mark_written(file_path)
        replaced = count if replace_all else 1
        confirmation = f"Replaced {replaced} occurrence(s) in {file_path}"

        # Compute diff from text already in hand -- no extra file read.
        idx = text.find(old_string)
        offset = text[:idx].count("\n") if idx >= 0 else 0
        diff = make_diff(old_string, new_string, offset)

        return ToolResult(
            call_id="",
            content=confirmation,
            diff=diff,
            diff_file_path=file_path,
        )

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a hint if the command is a simple ``sed -i 's/X/Y/g' FILE``.

        Args:
          trees: Parsed bashlex command trees from the active Bash call.

        Returns:
          hint: Nudge string redirecting to the Edit tool, or ``None``.

        """
        unwrapped = unwrap_cd_prefix(trees)
        if unwrapped is None:
            return None
        cwd, cmd = unwrapped
        if cmd.exe != "sed" or cmd.env_prefix:
            return None
        return _match_sed(cwd, cmd.args)


def _match_sed(cwd: str | None, args: tuple[str, ...]) -> str | None:
    """Match a simple ``sed -i 's/OLD/NEW/[g]' FILE`` for an Edit nudge."""
    del cwd  # Hint is a fixed string; path resolution is the LLM's job.
    in_place = False
    script: str | None = None
    file: str | None = None
    i = 0
    while i < len(args):
        a = args[i]
        # ``-i`` (GNU) or ``-i BACKUP`` (BSD) - accept only the
        # no-backup form to avoid suggesting a lossy translation.
        if a in {"-i", "--in-place"}:
            in_place = True
            i += 1
            continue
        if a.startswith("-i") and len(a) > 2 and not a.startswith("--"):
            # ``-i.bak`` - backup suffix, don't try to translate.
            return None
        if a.startswith("-"):
            return None
        if script is None:
            script = a
        elif file is None:
            file = a
        else:
            return None
        i += 1
    if not in_place or script is None or file is None:
        return None
    m = _SED_S_PATTERN.match(script)
    if m is None:
        return None
    if "i" in m.group(3):
        # Case-insensitive substitution - Edit is exact-match only.
        return None
    return _NUDGE
