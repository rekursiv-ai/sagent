"""Tool framework: decorator, state, sandbox.

This module contains the infrastructure for building tools.
For built-in tool implementations, see ``tools/`` siblings.

Usage::

    from sagent.tools.core import tool, get_tool_state

    @tool(name="MyTool")
    def my_tool(query: str) -> str:
        ...
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast, get_type_hints, overload

import asyncio
import contextvars
import dataclasses
import difflib
import inspect
import itertools
import logging
import re
import time
import typing

import yaml

from sagent.custom_types import (
    Message,
    TextMessage,
    TokenCount,
)
from sagent.lib.asyncio_collections import Deque
from sagent.lib.json import JSON, int_val, json_freeze
from sagent.lib.message import get_directive
from sagent.tools.lib.bash import BashParseCache


if TYPE_CHECKING:
    from sagent.agent import Agent
    from sagent.custom_types import ModelResponse
    from sagent.tools.background_task import BackgroundTaskEntry


logger = logging.getLogger(__name__)


# -- Recipe-based asset loader ----------------------------------------

_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_NOW_PLACEHOLDER = "{{NOW}}"
_DEFAULT_RECIPE = "sagent"
_RE_INCLUDE = re.compile(r"\{\{include:\s*(.+?)\}\}")
_recipe_path_override: Path | None = None
_recipe_cache: dict[str, object] | None = None


def resolve_recipe(name_or_path: str) -> Path:
    """Resolve a recipe spec to a yaml path.

    Accepts either a bare name (looked up in ``assets/<name>.yaml``) or
    a filesystem path (anything containing ``/`` or ending in
    ``.yaml``/``.yml``). The path is expanded and resolved.
    """
    looks_like_path = "/" in name_or_path or name_or_path.endswith((".yaml", ".yml"))
    base = (
        Path(name_or_path) if looks_like_path else _ASSETS_DIR / f"{name_or_path}.yaml"
    )
    return base.expanduser().resolve()


def set_recipe(name_or_path: str) -> None:
    """Switch the active recipe; clears the load cache."""
    global _recipe_path_override, _recipe_cache  # noqa: PLW0603 -- process-level state
    _recipe_path_override = resolve_recipe(name_or_path)
    _recipe_cache = None


def _load_recipe() -> dict[str, object]:
    """Load and cache the active recipe.yaml."""
    global _recipe_cache  # noqa: PLW0603 -- module-level cache
    if _recipe_cache is not None:
        return _recipe_cache
    recipe_path = _recipe_path_override or (_ASSETS_DIR / f"{_DEFAULT_RECIPE}.yaml")
    loaded = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    _recipe_cache = cast(dict[str, object], loaded) if isinstance(loaded, dict) else {}
    return _recipe_cache


def read_asset(path: str | Path) -> str:
    """Read an asset file, expanding ``{{include: path}}`` directives.

    Args:
      path: Relative path within the assets directory, or absolute path.

    Returns:
      text: File contents with include directives recursively expanded.

    """
    p = _ASSETS_DIR / path if isinstance(path, str) else path
    text = p.read_text(encoding="utf-8")

    def _replace(m: re.Match[str]) -> str:
        return read_asset(m.group(1).strip())

    return _RE_INCLUDE.sub(_replace, text)


def recipe_dict(key: str) -> dict[str, str]:
    """Get a typed dict section from the recipe.

    Args:
      key: Top-level recipe key to look up.

    Returns:
      section: String-to-string mapping, empty if key is absent or not a dict.

    """
    raw = _load_recipe().get(key, {})
    if not isinstance(raw, dict):
        return {}
    d = cast(dict[str, object], raw)
    return {str(k): str(v) for k, v in d.items()}


def recipe_list(section: str, key: str) -> list[str]:
    """Get a typed list from a recipe section.

    Args:
      section: Top-level recipe key containing a dict.
      key: Key within that dict whose value is a list.

    Returns:
      items: List of stringified items, empty if absent or malformed.

    """
    raw = _load_recipe().get(section, {})
    if not isinstance(raw, dict):
        return []
    d = cast(dict[str, object], raw)
    items = d.get(key, [])
    if not isinstance(items, list):
        return []
    return [str(x) for x in cast(list[object], items)]


_MISSING_TOOL_DESCRIPTION = "Tool description unavailable; use the JSON schema exactly."


def load_tool_description(name: str) -> str:
    """Load a tool description from the active recipe.

    Tool descriptions are optional prompt fragments, so missing files soft-fail
    to a generic agent-visible description. Other recipe assets stay strict at
    their call sites. Paths are explicit in recipe.yaml -- no fallback search.
    Substitutes ``{{NOW}}`` with the current
    ``Month YYYY``.

    Args:
      name: Tool name (case-insensitive lookup).

    Returns:
      description: Rendered tool description, or a generic fallback on miss.

    """
    tool_descs = recipe_dict("tool_descriptions")
    by_lower = {k.lower(): v for k, v in tool_descs.items()}
    key = name.lower()
    if key not in by_lower:
        logger.error(
            "Tool %r not in recipe %s", name, _recipe_path_override or _DEFAULT_RECIPE
        )
        return ""
    try:
        text = read_asset(str(by_lower[key])).rstrip()
    except FileNotFoundError:
        logger.exception(
            "Missing tool description asset for tool %r in recipe %s: %s",
            name,
            _recipe_path_override or _DEFAULT_RECIPE,
            by_lower[key],
        )
        return _MISSING_TOOL_DESCRIPTION
    if _NOW_PLACEHOLDER in text:
        text = text.replace(_NOW_PLACEHOLDER, time.strftime("%B %Y"))
    return text


# 100K tokens × ~4 chars/token.
TOOL_RESULT_MAX_CHARS = 400_000

# -- Python type → JSON Schema mapping --------------------------------

_TYPE_MAP: dict[type[object], str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _type_to_schema(t: type | object) -> dict[str, object]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    if isinstance(t, type) and t in _TYPE_MAP:
        return {"type": _TYPE_MAP[t]}
    origin = typing.get_origin(t)
    if origin is list:
        args = typing.get_args(t) or (str,)
        return {
            "type": "array",
            "items": _type_to_schema(args[0]),
        }
    if origin is dict:
        return {"type": "object"}
    return {"type": "string"}


def _build_schema(
    fn: Callable[..., object],
    hints: dict[str, object],
) -> dict[str, object]:
    """Build JSON Schema from function signature and type hints."""
    sig = inspect.signature(fn)
    properties: dict[str, object] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        t = hints.get(name, str)
        description = None
        origin = typing.get_origin(t)
        if origin is typing.Annotated:
            args = typing.get_args(t)
            t = args[0]
            for arg in args[1:]:
                if isinstance(arg, str):
                    description = arg
                    break
        prop = _type_to_schema(t)
        if description:
            prop["description"] = description
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)
    schema: dict[str, object] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def truncate(text: str, max_chars: int) -> str:
    """Truncate text, appending a notice if truncated.

    Args:
      text: Input text.
      max_chars: Maximum character count before truncation.

    Returns:
      result: Original text, or prefix with a truncation notice appended.

    """
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars] + f"\n... (truncated, {len(text) - max_chars} chars omitted)"
    )


def to_msg(result: str | Message) -> Message:
    """Normalize a tool return value to a ``Message``.

    Args:
      result: Plain string or existing Message.

    Returns:
      msg: A ``Message`` with ``text/plain`` descriptor if input was a string.

    """
    if isinstance(result, str):
        return TextMessage(result, "text/plain")
    return result


# -- Optional directive-parse helpers ----------------------------------


def opt_int(directive: Mapping[str, object], key: str) -> int | None:
    """Coerce ``directive[key]`` to int, or None if absent.

    Collapses the recurring 2-line ``_raw`` boilerplate at tool
    entrypoints where an arg is optional.

    Args:
      directive: Parsed tool directive mapping.
      key: Key to look up.

    Returns:
      value: Integer value, or None if the key is absent.

    """
    v = directive.get(key)
    return None if v is None else int_val(v, 0)


def opt_str(directive: Mapping[str, object], key: str) -> str | None:
    """Coerce ``directive[key]`` to str, or None if absent/empty.

    Args:
      directive: Parsed tool directive mapping.
      key: Key to look up.

    Returns:
      value: String value, or None if the key is absent or empty.

    """
    v = directive.get(key)
    if v is None:
        return None
    return str(v) or None


def resolve_tool_path(path: str) -> str:
    """Resolve a tool-supplied filesystem path against the agent shell cwd.

    Bash maintains ``ToolState.bash_cwd`` across ``cd`` commands. File tools
    should interpret relative paths the same way as Bash/List/Grep/Glob so a
    model does not see one cwd for shell commands and another for file edits.
    Empty paths are left empty so required-argument validation and tool-specific
    errors still surface cleanly.
    """
    if not path:
        return ""
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    return str(Path(get_tool_state().bash_cwd) / p)


# -- Helper for sync tool dispatch --------------------------------------


async def run_sync(
    fn: Callable[..., str | Message],
    *,
    parent_id: int = -1,
    **kwargs: object,
) -> Message:
    """Run a sync function in a thread, truncating text/plain output.

    ``fn`` may return a plain ``str`` or a ``Message``. Exceptions
    propagate to the caller (``agent.py::_run_tool`` catches them at
    the dispatch boundary).

    Usage in a tool's ``run``::

        async def run(self, msg: Message) -> Message:
            directive = get_directive(msg)
            return await run_sync(self._execute, parent_id=msg.id, **kwargs)

    Args:
      fn: Synchronous callable returning a string or Message.
      parent_id: Parent message ID for threading.
      **kwargs: Forwarded to ``fn``.

    Returns:
      msg: Result Message, truncated if text/plain.

    """
    result = await asyncio.to_thread(fn, **kwargs)
    m = to_msg(result)
    if m.descriptor == "text/plain":
        return TextMessage(
            truncate(cast(str, m.content), TOOL_RESULT_MAX_CHARS),
            "text/plain",
            parent_id=parent_id,
        )
    if parent_id != -1 and m.parent_id == -1:
        return dataclasses.replace(m, parent_id=parent_id)
    return m


# -- @tool decorator ---------------------------------------------------


class _ToolImpl:
    """Wraps a function as a Tool implementation."""

    __slots__ = (
        "_fn",
        "_is_async",
        "_max_result_chars",
        "description",
        "directive_schema",
        "emit_tool_summary",
        "name",
        "supports_microcompaction",
        "tool_id",
    )

    def __init__(
        self,
        fn: Callable[..., object],
        *,
        name: str | None = None,
        description: str | None = None,
        schema: JSON | None = None,
        max_result_chars: int = TOOL_RESULT_MAX_CHARS,
        supports_microcompaction: bool = False,
    ) -> None:
        self._fn = fn
        self._is_async = inspect.iscoroutinefunction(fn)
        self._max_result_chars = max_result_chars
        self.name = name or getattr(fn, "__name__", "tool")
        self.tool_id = f"application/x-tool-{self.name.lower()}"
        self.description = description or fn.__doc__ or ""
        hints = get_type_hints(fn, include_extras=True)
        hints.pop("return", None)
        self.directive_schema = schema or json_freeze(_build_schema(fn, hints))
        self.supports_microcompaction = supports_microcompaction
        self.emit_tool_summary = False

    def summary(self, msg: Message) -> str:
        """Return a short label for this tool invocation.

        Args:
          msg: Incoming tool-use message (unused; interface conformance).

        Returns:
          label: The tool name.

        """
        del msg
        return self.name

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool.

        Returns:
          prompt: Always empty for decorator-based tools.

        """
        return ""

    def summary_result(self, result: Message) -> str | None:
        """Return no receipt for decorator-based tools by default."""
        del result
        return None

    async def run(self, msg: Message) -> Message:
        """Invoke the wrapped function, propagating exceptions to the caller.

        Args:
          msg: Incoming tool-use message containing the directive.

        Returns:
          result: Tool result Message, truncated if text/plain.

        """
        kwargs = dict(get_directive(msg))
        if self._is_async:
            raw = cast(
                str | Message,
                await cast(Callable[..., Awaitable[object]], self._fn)(**kwargs),
            )
        else:
            raw = cast(str | Message, await asyncio.to_thread(self._fn, **kwargs))
        m = to_msg(raw)
        if m.descriptor == "text/plain":
            return TextMessage(
                truncate(cast(str, m.content), self._max_result_chars),
                "text/plain",
                parent_id=msg.id,
            )
        if m.parent_id == -1:
            return dataclasses.replace(m, parent_id=msg.id)
        return m


@overload
def tool(
    fn: Callable[..., object],
    /,
    *,
    name: str | None = ...,
    description: str | None = ...,
    schema: JSON | None = ...,
    max_result_chars: int = ...,
    supports_microcompaction: bool = ...,
) -> _ToolImpl: ...


@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
    schema: JSON | None = ...,
    max_result_chars: int = ...,
    supports_microcompaction: bool = ...,
) -> Callable[[Callable[..., object]], _ToolImpl]: ...


def tool(
    fn: Callable[..., object] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: JSON | None = None,
    max_result_chars: int = TOOL_RESULT_MAX_CHARS,
    supports_microcompaction: bool = False,
) -> _ToolImpl | Callable[[Callable[..., object]], _ToolImpl]:
    """Decorator to create a Tool from a function.

    Args:
      fn: The function to wrap (when used without arguments).
      name: Override the tool name (defaults to function name).
      description: Override description (defaults to docstring).
      schema: Override JSON schema (defaults to auto-generated).
      max_result_chars: Truncate results beyond this length.
      supports_microcompaction: Whether old results of this tool are
        eligible for microcompaction (clearing on cache-cold). Default False.

    Returns:
      tool_impl: A Tool implementation wrapping fn.

    """
    if fn is not None:
        return _ToolImpl(
            fn,
            name=name,
            description=description,
            schema=schema,
            max_result_chars=max_result_chars,
            supports_microcompaction=supports_microcompaction,
        )
    return lambda f: _ToolImpl(
        f,
        name=name,
        description=description,
        schema=schema,
        max_result_chars=max_result_chars,
        supports_microcompaction=supports_microcompaction,
    )


# -- Per-agent tool state ----------------------------------------------


class ReadCacheEntry(NamedTuple):
    """Cached read parameters and mtime for dedup and staleness checks."""

    offset: int
    limit: int
    last_lines: int
    mtime: float


class ToolState:
    """Per-agent state for tools (file tracking, cwd).

    Each Agent creates one and sets it as the active context
    before dispatching tools. Tools access it via
    ``get_tool_state()``.
    """

    def __init__(self) -> None:
        # Ordered: resolved_path → original_path. Serves as both
        # the "has been read" set and recency-ordered file list.
        self._read_order: dict[str, str] = {}
        # (offset, limit, last_lines, mtime) of the last read. All
        # four participate in the dedup key: a Read with different
        # last_lines returns different content, so cache hits must
        # match last_lines too.
        self.read_cache: dict[str, ReadCacheEntry] = {}
        # resolved_path → content at last read (for change diffs).
        self._content_cache: dict[str, str] = {}
        self.start_cwd: str = str(Path.cwd())
        self.bash_cwd: str = self.start_cwd
        # Extra directories from ``--add-dir`` whose AGENTS.md files get
        # walked alongside cwd's. Defaults off because it can blow the
        # prompt budget.
        self.additional_dirs: list[str] = []
        # Deferred-compaction signal: the Compact tool sets this to
        # ``custom_instructions``; the Agent drains it after the tool
        # dispatch batch and runs compaction.
        self.compact_requested: str | None = None
        # Deferred full-clear signal: the Clear tool sets this to a
        # ``reason`` string; the Agent drains it between model requests and
        # wipes history + file caches.
        self.clear_requested: str | None = None
        # Deferred recompact signal: the Recompact tool sets this to
        # ``custom_instructions``; the Agent drains it between model requests,
        # reloads the pre-compact transcript, and re-runs the
        # compactor with the new guidance.
        self.recompact_requested: str | None = None
        # Per-request session stats written by the Agent, read by the
        # Diagnostics tool. Stays empty until the first model request completes.
        self.stats: dict[str, float | int] = {}
        # Per-request bashlex parse cache: command string → trees (or None).
        # Populated by the agent's pre-dispatch concurrency check; read
        # by the Bash tool's matcher dispatch. Eliminates the
        # second parse of the same command within one Bash invocation.
        # Cleared between model requests by the Agent.
        self.bash_parse_cache: BashParseCache = {}
        # Skills invoked this session (by name). Tracked so
        # post-compaction can re-inject their SKILL.md bodies.
        self.invoked_skills: set[str] = set()
        # Subagent recursion depth. 0 for root Agent; incremented by
        # ``Agent.run`` on entry when a parent ``ToolState`` is
        # visible in the ambient context. Used by AgentSpawn's depth
        # enforcement.
        self.depth: int = 0

    @property
    def recent_files(self) -> list[str]:
        """Recently-read files (original paths, oldest first)."""
        return list(self._read_order.values())

    def reset_file_tracking(self) -> None:
        """Clear read/content/recency caches.

        Called by the Agent on AgentSelf(clear) so the cleared session
        starts with no recollection of previously-read or -edited
        files. Intentionally does NOT reset ``bash_cwd`` (the shell
        cwd is independent of conversation state) or
        ``additional_dirs`` (CLI-supplied, not session-state).
        """
        self.read_cache.clear()
        self._read_order.clear()
        self._content_cache.clear()

    def mark_read(
        self,
        path: str,
        offset: int = 0,
        limit: int = 0,
        last_lines: int = 0,
        content: str = "",
        mtime: float | None = None,
    ) -> None:
        """Record that a file has been read.

        Args:
          path: File path (resolved internally).
          offset: Starting line offset of the read.
          limit: Maximum lines read.
          last_lines: EOF-anchored line count.
          content: Full file text for content-based staleness checks.
          mtime: Pre-read mtime. If None, the file is stat'd now.
              Pass the mtime captured *before* reading bytes to avoid
              races with concurrent writers.

        """
        resolved = str(Path(path).resolve())
        self._read_order.pop(resolved, None)
        self._read_order[resolved] = path
        logger.debug("mark_read: %s", resolved)
        if mtime is None:
            try:
                mtime = Path(path).stat().st_mtime
            except OSError:
                mtime = 0.0
        self.read_cache[resolved] = ReadCacheEntry(offset, limit, last_lines, mtime)
        if content:
            self._content_cache[resolved] = content

    def mark_written(self, path: str) -> None:
        """Re-stamp after a successful write.

        Clears offset/limit/last_lines to break read dedup (forces
        re-fetch on next Read) and updates mtime.

        Args:
          path: File path (resolved internally).

        """
        resolved = str(Path(path).resolve())
        try:
            mtime = Path(path).stat().st_mtime
        except OSError:
            mtime = 0.0
        self.read_cache[resolved] = ReadCacheEntry(0, 0, 0, mtime)
        # Update content cache with what's now on disk.
        try:
            self._content_cache[resolved] = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            self._content_cache.pop(resolved, None)

    def check_unchanged(
        self,
        path: str,
        offset: int,
        limit: int,
        last_lines: int = 0,
    ) -> bool:
        """Return True if file unchanged since last read.

        Args:
          path: File path to check.
          offset: Line offset of the read to compare.
          limit: Line limit of the read to compare.
          last_lines: EOF-anchored line count to compare.

        Returns:
          unchanged: True if mtime and read parameters match the cache.

        """
        resolved = str(Path(path).resolve())
        cached = self.read_cache.get(resolved)
        if cached is None:
            return False
        prev_offset, prev_limit, prev_last_lines, prev_mtime = cached
        if (
            prev_offset != offset
            or prev_limit != limit
            or prev_last_lines != last_lines
        ):
            return False
        try:
            current_mtime = Path(path).stat().st_mtime
        except OSError:
            return False
        return current_mtime == prev_mtime

    def check_stale(self, path: str) -> bool:
        """Return True if cached belief of file content no longer matches disk.

        Fast path: mtime matches → not stale. Fallback: mtime differs
        but disk content equals cached content → not stale (mtime bumped
        without a real change: cloud sync, antivirus, idempotent
        reformatter, etc.).

        Args:
          path: File path to check.

        Returns:
          stale: True if file content on disk differs from cached belief.

        """
        resolved = str(Path(path).resolve())
        cached = self.read_cache.get(resolved)
        if cached is None:
            return False
        *_, prev_mtime = cached
        try:
            current_mtime = Path(path).stat().st_mtime
        except OSError:
            return False
        if current_mtime == prev_mtime:
            return False
        cached_content = self._content_cache.get(resolved)
        if cached_content is None:
            # No cached content (binary / PDF / image). Conservatively stale.
            return True
        try:
            current_content = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return True
        if current_content != cached_content:
            return True
        # Content-equality match: mtime bumped but bytes identical.
        # Refresh the cached mtime so subsequent ``check_stale`` hits
        # the O(1) mtime-match fast path instead of re-reading and
        # re-comparing the full file on every call.
        prev_offset, prev_limit, prev_last_lines, _ = cached
        self.read_cache[resolved] = ReadCacheEntry(
            prev_offset, prev_limit, prev_last_lines, current_mtime
        )
        return False

    def consume_changed_files(self) -> dict[str, str]:
        """Pop files modified since last read, returning diffs.

        Side-effecting by design: after a file's change is reported
        once, the cached mtime is bumped so the next call won't
        re-report it. The ``consume_`` prefix signals that the state
        transition happens on every call.

        Returns:
          changes: Map from original path to a unified diff
              snippet, for files whose disk mtime exceeds the
              cached mtime.

        """
        changes: dict[str, str] = {}
        for resolved, orig_path in self._read_order.items():
            cached = self.read_cache.get(resolved)
            if cached is None:
                continue
            prev_offset, prev_limit, prev_last_lines, prev_mtime = cached
            try:
                current_mtime = Path(orig_path).stat().st_mtime
            except OSError:
                continue
            if current_mtime == prev_mtime:
                continue
            old = self._content_cache.get(resolved, "")
            try:
                new = Path(orig_path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            diff = "".join(
                difflib.unified_diff(
                    old.splitlines(keepends=True),
                    new.splitlines(keepends=True),
                    fromfile=orig_path,
                    tofile=orig_path,
                    n=3,
                ),
            )
            if diff:
                changes[orig_path] = diff
            # Refresh mtime so we don't re-report next request, but
            # preserve the (offset, limit, last_lines) shape so a subsequent
            # Read with the same parameters still hits dedup.
            self.read_cache[resolved] = ReadCacheEntry(
                prev_offset,
                prev_limit,
                prev_last_lines,
                current_mtime,
            )
            self._content_cache[resolved] = new
        return changes

    def has_been_read(self, path: str) -> bool:
        """Check if a file has been read in this session.

        Args:
          path: File path to check.

        Returns:
          was_read: True if the file has been read.

        """
        return str(Path(path).resolve()) in self._read_order

    def enforce_read(self, file_path: str) -> str | None:
        """Return error if file not yet read, else None.

        Args:
          file_path: File path to check.

        Returns:
          error: Error message string, or None if the file was read.

        """
        if not self.has_been_read(file_path):
            return f"File not yet read: {file_path}. Read it first."
        return None


# Context variable for per-agent tool state.
tool_state_var: contextvars.ContextVar[ToolState] = contextvars.ContextVar("tool_state")

# Fallback state for use outside an Agent context.
_default_state = ToolState()


def get_tool_state() -> ToolState:
    """Get the current agent's tool state.

    Returns:
      state: The active ToolState, or a module-level default.

    """
    return tool_state_var.get(_default_state)


@contextmanager
def tool_state_context(state: ToolState) -> Generator[None]:
    """Install ``state`` as the active tool state for the duration of the block.

    Args:
      state: ToolState to make active.

    """
    token = tool_state_var.set(state)
    try:
        yield
    finally:
        tool_state_var.reset(token)


# -- Agent-tree ContextVars for AgentSpawn -----------------------------
#
# These three ContextVars carry cross-agent state down the spawn tree
# so AgentSpawn can look up its parent, enforce ``max_depth``, and
# aggregate cost into one shared ledger at the root.
#
# Placed in ``tools/core.py`` rather than ``agent.py`` because
# ``tools/agent_spawn.py`` must import them without dragging the full
# ``Agent`` class through a circular import. The ``Agent`` annotation is
# guarded by ``if TYPE_CHECKING`` above.

# The Agent instance currently executing. Set by ``Agent.run``
# on entry and reset on exit. Tools invoked mid-request read this to
# reach their invoking Agent (e.g. to inherit system/tools/model).
current_agent_var: contextvars.ContextVar[Agent | None] = contextvars.ContextVar(
    "current_agent", default=None
)

# Effective max_depth cap for the current subtree. Set by AgentSpawn so
# grandchild factory invocations see the same cap as their ancestor,
# regardless of which factory instance they use.
max_depth_var: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "max_depth", default=None
)

# Shared cost ledger across a tree of Agents. The root Agent creates
# one on entry if none exists; descendants read and mutate the same
# instance (dict mutation in asyncio.gather children propagates
# because tasks *share* the same CostLedger object even though each
# gets its own copy of the ContextVar binding).
cost_ledger_var: contextvars.ContextVar[CostLedger | None] = contextvars.ContextVar(
    "cost_ledger", default=None
)

# Hierarchical agent path. Root is ""; first child is "0", its
# second child is "0_1", etc. Set by AgentSpawn before launching
# the child so the child and its descendants can build labels like
# ``Agent_0_1_3``.
agent_path_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agent_path", default=""
)

# Per-agent child counter. ``Agent.run`` installs a fresh
# ``itertools.count()`` on entry; concurrent siblings spawned by
# ``asyncio.gather`` share the same object (atomic under CPython)
# so each gets a unique index.
agent_counter_var: contextvars.ContextVar[itertools.count[int]] = (
    contextvars.ContextVar("agent_counter")
)

# Human-readable label for the currently-executing agent. Root uses
# ``Agent.name`` (e.g. "Agent"); children get labels like "Agent_0",
# "Agent_0_1". Set by ``Agent.run`` (root) and ``AgentSpawn.run``
# (children). Read by ``AgentSend`` to tag the sender.
agent_label_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agent_label", default=""
)


class AgentLike(Protocol):
    inbox: Deque[Message]
    # Foreground spawned tasks (model calls, tool batches, subagent runs).
    # Cancelled by ``/break`` and ``/abort``.
    tasks: dict[int, asyncio.Task[None]]
    # Long-running tasks: visible (user-scheduled bg tools) and hidden
    # (REPL pump, daemons). ``BackgroundTaskEntry.hidden`` distinguishes.
    # Survives ``/break`` and bare ``/abort``; ``/abort all`` cancels
    # the visible ones.
    background_tasks: dict[str, BackgroundTaskEntry]


# Process-wide registry of live agents, keyed by label. Agents
# register on ``run()`` entry and deregister in ``finally``. Used
# by ``AgentSend`` to route messages by label. Safe under sagent's
# single-event-loop concurrency model; not thread-safe.
agent_registry: dict[str, AgentLike] = {}


@dataclass(slots=True, kw_only=True)
class CostLedger:
    """Aggregated cost/token stats across an Agent subtree.

    Root ``Agent.run`` creates one and sets it as the active
    ContextVar. Each subsequent model response (across the root and
    every descendant subagent) accumulates into this same object by
    in-place mutation - so parallel siblings spawned via
    ``asyncio.gather`` share one authoritative ledger despite
    ContextVar copy-on-spawn semantics.
    """

    total_cost_usd: float = 0.0
    tokens: TokenCount = field(default_factory=TokenCount)
    calls_by_model: dict[str, int] = field(default_factory=dict)

    def accumulate(self, response: ModelResponse, model_id: str) -> None:
        """Fold one ``ModelResponse`` into the running totals.

        Args:
          response: Model response with cost and token counts.
          model_id: Model identifier for per-model call tracking.

        """
        self.total_cost_usd += response.total_cost
        self.tokens = self.tokens + response.tokens
        self.calls_by_model[model_id] = self.calls_by_model.get(model_id, 0) + 1


# Convenience functions that delegate to current state.
def mark_read(
    path: str,
    offset: int = 0,
    limit: int = 0,
    last_lines: int = 0,
    content: str = "",
    mtime: float | None = None,
) -> None:
    """Record that a file has been read.

    Args:
      path: File path (resolved internally).
      offset: Starting line offset of the read.
      limit: Maximum lines read.
      last_lines: EOF-anchored line count.
      content: Full file text for content-based staleness checks.
      mtime: Pre-read mtime, or None to stat now.

    """
    get_tool_state().mark_read(
        path,
        offset,
        limit,
        last_lines=last_lines,
        content=content,
        mtime=mtime,
    )


def has_been_read(path: str) -> bool:
    """Check if a file has been read.

    Args:
      path: File path to check.

    Returns:
      was_read: True if the file has been read in this session.

    """
    return get_tool_state().has_been_read(path)


# -- Per-file write serialization --------------------------------------

# Process-wide registry of asyncio.Lock keyed by resolved file path.
# Mutating tools (Edit, Write, etc.) serialize their read-modify-write
# critical sections through this registry so concurrent coroutines -
# e.g. parallel subagents dispatching Edit on the same file - can't
# interleave between the staleness check and the write. Non-mutating
# Reads don't participate; parallel Reads are always safe.
#
# Same path → same lock → serialized across all mutating tools.
# Different paths → different locks → parallel.
#
# asyncio.Lock is safe because sagent runs on a single event loop; the
# lock yields the coroutine (doesn't block the loop) so other work
# continues while a mutation is in flight in the thread pool.
#
# Dict growth is unbounded across a session but each entry is tiny
# (~200 bytes); hundreds of distinct files edited per session is
# bounded-memory trivia. No cleanup needed.
_file_write_locks: dict[str, asyncio.Lock] = {}


def get_file_write_lock(path: str) -> asyncio.Lock:
    """Return the shared write lock for ``path``.

    Args:
      path: File path (resolved internally).

    Returns:
      lock: Shared asyncio.Lock for the resolved path.

    """
    # Avoid ``setdefault`` here - it eagerly constructs a new Lock on
    # every call even when the key already exists, and throws it
    # away. Edit/Write hit this on every mutation; cheap to skip.
    resolved = str(Path(path).resolve())
    lock = _file_write_locks.get(resolved)
    if lock is None:
        lock = asyncio.Lock()
        _file_write_locks[resolved] = lock
    return lock


# -- Changed-file context provider ------------------------------------


def changed_files_context(max_diff_lines: int = 500) -> str:
    """Context provider: detect externally modified files.

    Checks all cached files against disk mtime. For each
    changed file, generates a system-reminder with a unified
    diff snippet (capped at ``max_diff_lines``).

    Wire this into Agent via ``context_providers``::

        Agent(context_providers=[changed_files_context])

    Args:
      max_diff_lines: Per-file truncation threshold for the diff
          snippet.

    Returns:
      context: System reminder text, or empty string if
          no files changed.

    """
    changes = get_tool_state().consume_changed_files()
    if not changes:
        return ""
    parts: list[str] = []
    for path, raw_diff in changes.items():
        diff_lines = raw_diff.splitlines(keepends=True)
        snippet = (
            "".join(diff_lines[:max_diff_lines]) + "\n... (truncated)"
            if len(diff_lines) > max_diff_lines
            else raw_diff
        )
        parts.append(
            f"Note: {path} was modified, either by the user, a"
            f" linter, or another agent. This change was intentional,"
            f" so make sure to take it into account as you proceed"
            f" (ie. don't revert it unless the user asks you to)."
            f" Don't tell the user this, since they are already"
            f" aware. Here are the relevant changes (shown with"
            f" line numbers):\n{snippet}",
        )
    return "<system-reminder>\n" + "\n".join(parts) + "\n</system-reminder>"
