"""Edit tool: exact-string replacement in existing files.

Unlike Write, Edit intentionally does not require a prior Read. The
``old_string`` exact match is the operation's safety gate, which keeps
sed-like replacements cheap when the caller already knows the target text.

Tradeoff:
- Pro: avoids spending context on a full file read for simple, precise edits.
- Pro: still fails closed when ``old_string`` is absent or ambiguous unless
  ``replace_all`` is explicitly enabled.
- Con: stale-file detection is only meaningful when the file has previously
  been read or written in this session.
- Con: callers can perform blind exact replacements, so unclear edits should
  still use Read first for context.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import difflib
import re

from sagent.custom_types import (
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.atomic_file import atomic_write_bytes
from sagent.lib.json import JSON, bool_val, json_freeze
from sagent.lib.message import get_directive
from sagent.tools.core import (
    get_file_write_lock,
    get_tool_state,
    load_tool_description,
    resolve_tool_path,
    run_sync,
)
from sagent.tools.lib.bash import Node, unwrap_cd_prefix


# Simple ``s/OLD/NEW/[g]`` - no escaped delimiters, no alternate
# delimiter, no address ranges. If the sed script is more complex
# than this, we bail.
_SED_S_PATTERN = re.compile(r"^s/([^/\\]*)/([^/\\]*)/([gi]*)$")


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)((?:,\d+)?) \+(\d+)((?:,\d+)?) @@")


def make_diff(old: str, new: str, offset: int) -> str:
    """Produce a unified diff with absolute line numbers.

    Args:
      old: Original text.
      new: Replacement text.
      offset: Line offset applied to @@ hunk headers.

    Returns:
      diff: Unified diff string with adjusted line numbers.

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


_NUDGE = "sed via Bash is a bad UX. Use the Edit tool."


class Edit:
    """Perform exact-string replacement in files.

    This tool deliberately permits unread files. The exact ``old_string`` match
    is treated as the safety check; stale-file checks apply when prior session
    state exists, but lack of a prior Read is not itself an error.
    """

    name: str = "Edit"
    tool_id: str = "application/x-tool-edit"
    description: str = load_tool_description("Edit")
    supports_microcompaction: bool = True
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

    async def run(self, msg: Message) -> Message:
        """Apply an exact-string replacement to a file.

        Args:
          msg: Incoming tool-use message containing the directive.

        Returns:
          result: Tool result Message confirming the edit.

        """
        directive = get_directive(msg)
        file_path = resolve_tool_path(str(directive.get("file_path", "")))
        old_string = str(directive.get("old_string", ""))
        new_string = str(directive.get("new_string", ""))
        replace_all = bool_val(directive.get("replace_all"), False)
        # Serialize against any other mutating tool on the same file.
        # Different files → different locks → parallel subagents editing
        # disjoint files don't queue. Same file → atomic read-modify-write.
        async with get_file_write_lock(file_path):
            return await run_sync(
                self._run,
                parent_id=msg.id,
                file_path=file_path,
                old_string=old_string,
                new_string=new_string,
                replace_all=replace_all,
            )

    def summary(self, msg: Message) -> str:
        """Return a short label summarizing this edit.

        Args:
          msg: Incoming tool-use message.

        Returns:
          label: ``"Edit <filename>"``.

        """
        directive = get_directive(msg)
        file_path = resolve_tool_path(str(directive.get("file_path", "")))
        fname = Path(file_path).name if file_path else "?"
        return f"Edit {fname}"

    def summary_result(self, result: Message) -> str | None:
        del result
        return None

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          prompt: Always empty.

        """
        return ""

    def _validate(self, file_path: str, old_string: str) -> Message | None:
        """Return an error ``Message`` on invalid inputs, else ``None``."""
        if not old_string:
            # text.count("") returns len+1; text.replace("", …) inserts
            # between every char. Silently destroys the file - reject.
            return TextMessage("old_string cannot be empty.", "text/x-error")
        p = Path(file_path)
        if not p.exists():
            return TextMessage(f"File not found: {file_path}", "text/x-error")
        if p.is_dir():
            return TextMessage(
                f"{file_path} is a directory, not a file.", "text/x-error"
            )
        state = get_tool_state()
        if state.check_stale(file_path):
            return TextMessage(
                (
                    "File has been modified since read, either by the user, a"
                    " linter, or another agent. Read it again before attempting"
                    " to edit it."
                ),
                "text/x-error",
            )
        return None

    def _run(
        self,
        *,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str | Message:
        err = self._validate(file_path, old_string)
        if err is not None:
            return err
        p = Path(file_path)
        state = get_tool_state()
        text = p.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            return TextMessage("old_string not found in file.", "text/x-error")
        if count > 1 and not replace_all:
            return TextMessage(
                (
                    f"old_string found {count} times."
                    f" Use replace_all=True or provide more context."
                ),
                "text/x-error",
            )
        new_text = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )
        # Atomic write (tmp + rename). Without this a concurrent Read
        # on the same file could observe a torn state between
        # ``write_text``'s truncate and its rewrite. The tmp+rename
        # creates a fresh inode, which by default picks up umask
        # permissions; forward the original file mode so Edit doesn't
        # silently flip a ``0o600`` file to ``0o644``.
        file_mode = p.stat().st_mode & 0o777
        atomic_write_bytes(p, new_text.encode("utf-8"), file_mode=file_mode)
        state.mark_written(file_path)
        replaced = count if replace_all else 1
        confirmation = f"Replaced {replaced} occurrence(s) in {file_path}"

        # Compute diff from text already in hand -- no extra file read.
        idx = text.find(old_string)
        offset = text[:idx].count("\n") if idx >= 0 else 0
        diff = make_diff(old_string, new_string, offset)

        return MultipartMessage(
            (
                TextMessage(confirmation, "text/plain"),
                TextMessage(diff, "text/x-diff"),
            ),
            "multipart/x-tool-result",
        )

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a hint if the command is a simple ``sed -i 's/X/Y/g' FILE``.

        Accepts ``cd PATH && CMD`` compounds via ``unwrap_cd_prefix``.
        Only matches trivial ``s/OLD/NEW/[g]`` with ``-i``; anything
        more complex (address ranges, multiple commands, alternate
        delimiters, escaped slashes) bails.

        Args:
          trees: Parsed shell AST nodes.

        Returns:
          hint: Suggested Edit invocation, or ``None`` if no match.

        """
        unwrapped = unwrap_cd_prefix(trees)
        if unwrapped is None:
            return None
        cwd, cmd = unwrapped
        if cmd.exe != "sed" or cmd.env_prefix:
            return None
        return _match_sed(cwd, cmd.args)


def _match_sed(cwd: str | None, args: tuple[str, ...]) -> str | None:
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
