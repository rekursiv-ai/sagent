"""Path-sandboxed wrappers around ``sagent.tools.Write`` and ``sagent.tools.Edit``.

The statistician role is permitted to edit experiment scripts and
diagnostic notebooks under ``tuningfork/experiments/`` only. Without
runtime enforcement, the role's prompt-level "do not edit production
code" instruction is advisory — a misjudgment costs us a production
edit. These wrappers convert the boundary into a structural property
of the tool dispatch: any attempt to write or edit a path outside the
configured sandbox root resolves immediately as a ``ToolResult`` with
``is_error=True``, without ever opening the file.

The wrappers inherit metadata (``name``, ``tool_id``,
``directive_schema``, ``summary``, etc.) from the underlying tool, so
the model sees an ordinary ``Write`` / ``Edit`` tool — no special
prompt needed. Only the validation hook in ``run`` differs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import override

from sagent.tools.core import resolve_tool_path
from sagent.tools.edit import Edit
from sagent.tools.write import Write
from sagent.types.runtime import ToolResult


def _is_within(child: Path, parent: Path) -> bool:
    """True if ``child`` is ``parent`` or any descendant.

    Resolves both ends (follows symlinks, normalizes ``..``) so a
    sneaky ``../../etc/passwd`` argument cannot escape via path
    manipulation. ``parent`` is resolved at config time by the
    caller; ``child`` at dispatch time.
    """
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _reject(message: str) -> ToolResult:
    return ToolResult(call_id="", content=message, is_error=True)


class SandboxedWrite(Write):
    """``Write`` restricted to a configured root directory.

    Args:
        sandbox_root: Directory tree the tool is allowed to write
            inside. Must be an absolute path. Resolved once at
            construction.

    """

    def __init__(self, *, sandbox_root: Path | str) -> None:
        super().__init__()
        root = Path(sandbox_root)
        if not root.is_absolute():
            raise ValueError(f"sandbox_root must be absolute, got {sandbox_root!r}")
        self._sandbox_root = root.resolve()

    @override
    async def run(self, args: Mapping[str, object]) -> ToolResult:
        raw_path = str(args.get("file_path", ""))
        if not raw_path:
            return _reject("'file_path' is required.")
        resolved_str = resolve_tool_path(raw_path)
        target = Path(resolved_str).resolve()
        if not _is_within(target, self._sandbox_root):
            return _reject(
                f"Refused: {target} is outside the statistician sandbox "
                f"({self._sandbox_root}). Statistician may only write under "
                f"this root. Flag production-code changes to @swe via "
                f"AgentSend instead."
            )
        return await super().run(args)


class SandboxedEdit(Edit):
    """``Edit`` restricted to a configured root directory. See ``SandboxedWrite``."""

    def __init__(self, *, sandbox_root: Path | str) -> None:
        super().__init__()
        root = Path(sandbox_root)
        if not root.is_absolute():
            raise ValueError(f"sandbox_root must be absolute, got {sandbox_root!r}")
        self._sandbox_root = root.resolve()

    @override
    async def run(self, args: Mapping[str, object]) -> ToolResult:
        raw_path = str(args.get("file_path", ""))
        if not raw_path:
            return _reject("'file_path' is required.")
        resolved_str = resolve_tool_path(raw_path)
        target = Path(resolved_str).resolve()
        if not _is_within(target, self._sandbox_root):
            return _reject(
                f"Refused: {target} is outside the statistician sandbox "
                f"({self._sandbox_root}). Statistician may only edit under "
                f"this root. Flag production-code changes to @swe via "
                f"AgentSend instead."
            )
        return await super().run(args)
