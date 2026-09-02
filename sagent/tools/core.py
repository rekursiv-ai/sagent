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

from collections.abc import Awaitable, Callable, Iterable, Mapping
from pathlib import Path
from typing import Final, cast, get_type_hints, overload

import asyncio
import dataclasses
import functools
import inspect
import locale
import logging
import re
import threading
import time
import typing

import yaml

from sagent.agent.state import (
    approx_tokens,
    current_agent_var,
    get_tool_state,
)
from sagent.lib.custom_json import JSON, IntCodec, json_freeze
from sagent.types.runtime import ToolResult


# Only names this module DEFINES. Re-exporting what ``agent.state`` owns
# gave every such symbol two import paths; reach into the owner instead.
__all__ = (
    "bound_by_tokens",
    "changed_files_context",
    "file_lock_key",
    "get_file_write_lock",
    "has_been_read",
    "load_tool_description",
    "locked_file_write",
    "mark_read",
    "opt_int",
    "opt_str",
    "provider_not_allowed_result",
    "read_asset",
    "recipe_dict",
    "recipe_list",
    "resolve_recipe",
    "resolve_tool_path",
    "result_token_budget",
    "run_sync",
    "set_recipe",
    "to_result",
    "tool",
    "truncate_to_budget",
)

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_NOW_PLACEHOLDER: Final = "{{NOW}}"
_DEFAULT_RECIPE: Final = "sagent"
_RE_INCLUDE = re.compile(r"\{\{include:\s*(.+?)\}\}")
_recipe_path_override: Path | None = None
_recipe_cache: dict[str, object] | None = None


def resolve_recipe(name_or_path: str) -> Path:
    """Resolve a recipe spec to a yaml path.

    Accepts either a bare name (looked up in ``assets/<name>.yaml``) or
    a filesystem path (anything containing ``/`` or ending in
    ``.yaml``/``.yml``). The path is expanded and resolved.

    Args:
      name_or_path: Recipe name or filesystem path.

    Returns:
      path: Expanded, resolved absolute path to the recipe yaml.

    """
    looks_like_path = "/" in name_or_path or name_or_path.endswith((".yaml", ".yml"))
    base = (
        Path(name_or_path) if looks_like_path else _ASSETS_DIR / f"{name_or_path}.yaml"
    )
    return base.expanduser().resolve()


def set_recipe(name_or_path: str) -> None:
    """Switch the active recipe; clears the load cache.

    Args:
      name_or_path: Recipe name or filesystem path.

    """
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

    Cycles and runaway nesting are clamped: a per-call visited-set
    detects self-include and a depth cap (see ``_MAX_ASSET_DEPTH``)
    bounds total include chain length. Both render an inline marker
    instead of recursing -- mirrors the contract in
    ``agents_md._process``.

    Args:
      path: Relative path within the assets directory, or absolute path.

    Returns:
      text: File contents with include directives recursively expanded.

    """
    return _read_asset(path, visited=set(), depth=0)


_MAX_ASSET_DEPTH = 8  # config-globals: ignore -- include-recursion depth dial


def _read_asset(path: str | Path, *, visited: set[str], depth: int) -> str:
    """Recursive worker for :func:`read_asset` with cycle/depth guards."""
    p = _ASSETS_DIR / path if isinstance(path, str) else path
    try:
        resolved = p.resolve()
    except OSError:
        return f"[include: unreadable path {p}]"
    key = str(resolved)
    if key in visited:
        return f"[include: cycle on {p}]"
    if depth >= _MAX_ASSET_DEPTH:
        return f"[include: depth cap reached at {p}]"
    visited.add(key)
    text = resolved.read_text(encoding="utf-8")

    def _replace(m: re.Match[str]) -> str:
        return _read_asset(m.group(1).strip(), visited=visited, depth=depth + 1)

    try:
        return _RE_INCLUDE.sub(_replace, text)
    finally:
        # Discard on the way out: ``visited`` tracks the ANCESTOR chain, so
        # a cycle is a file including itself. Keeping every file ever seen
        # would report the second of two sibling includes as a cycle.
        visited.discard(key)


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


_MISSING_TOOL_DESCRIPTION: Final = (
    "Tool description unavailable; use the JSON schema exactly."
)


def load_tool_description(name: str) -> str:
    """Load a tool description from the active recipe.

    Tool descriptions are optional prompt fragments, so missing files soft-fail
    to a generic agent-visible description. Other recipe assets stay strict at
    their call sites. Paths are explicit in recipe.yaml -- no fallback search.
    Substitutes ``{{NOW}}`` with the current locale-formatted weekday, date,
    and 6-hour bucket (``Weekday, locale-date, 12am - 6am``). Coarse buckets
    keep cross-process prompt caches warm: at most four invalidations per day.

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
        return _MISSING_TOOL_DESCRIPTION
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
        text = text.replace(_NOW_PLACEHOLDER, _now_bucket())
    return text


def _now_bucket(now: time.struct_time | None = None) -> str:
    """Locale-formatted weekday, date, and 6-hour bucket range.

    Honors ``$LC_TIME``. Buckets at 00/06/12/18 keep cross-process prompt
    caches warm: at most four invalidations per day.
    """
    _ensure_locale_time()
    t = now if now is not None else time.localtime()
    lo = (t.tm_hour // 6) * 6

    def fmt(h: int) -> str:
        b = time.struct_time(
            (
                t.tm_year,
                t.tm_mon,
                t.tm_mday,
                h % 24,
                0,
                0,
                t.tm_wday,
                t.tm_yday,
                t.tm_isdst,
            )
        )
        return re.sub(r":00(?=\D|$)", "", time.strftime("%X", b), count=1)

    return f"{time.strftime('%a, %x', t)}, {fmt(lo)} - {fmt(lo + 6)}"


@functools.cache
def _ensure_locale_time() -> None:
    try:
        locale.setlocale(locale.LC_TIME, "")
    except locale.Error:
        locale.setlocale(locale.LC_TIME, "C")


def result_token_budget(*, fallback: int = 50_000) -> int:
    """Tokens one tool result may occupy, from the ACTIVE model.

    Args:
      fallback: Budget when no agent is in context (standalone tool use,
          tests). Sized near the smallest window a real model declares.

    Returns:
      budget: Token ceiling for one result.

    """
    agent = current_agent_var.get(None)
    ceiling = agent.max_result_tokens if agent is not None else 0
    return ceiling if ceiling > 0 else fallback


def bound_by_tokens(units: Iterable[str], *, budget: int) -> tuple[str, int]:
    """Join ``units`` while they fit ``budget``; report how many were kept.

    The single place every text tool stops. Tools pass their own rendered
    units -- a numbered line, a match row, a path -- and this counts what
    the provider will count, so no tool converts a budget into a unit of
    its own. Read bounded itself in LINES via an assumed 80 chars/line,
    which is 40x wrong on JSONL: session ``190b6baec7ed`` shipped an 11.1M
    character result and could not recover.

    Emitting at least one unit is deliberate: a result whose first unit
    alone busts the budget is still more useful than an empty body, and
    the caller's resume note names where to continue.

    The count is RETURNED rather than folded into a note here: a caller
    that windowed its input already withheld units this function never
    saw, so only the caller can phrase the remainder correctly.

    Args:
      units: Rendered pieces, in output order. Each is emitted whole --
          a sliced unit (half a line, a truncated path) reads as content
          rather than as a boundary.
      budget: Token ceiling from :func:`result_token_budget`.

    Returns:
      body: The kept units, joined.
      kept: How many units ``body`` holds.

    """
    out: list[str] = []
    used = 0
    for unit in units:
        cost = approx_tokens(unit)
        if out and used + cost > budget:
            break
        if not out and cost > budget:
            # The FIRST unit alone busts the budget, so dropping units
            # cannot help -- a minified blob is ONE line. Emitting it
            # whole is how an 11.1M-character result reached the wire in
            # session ``190b6baec7ed``; cutting inside the unit is worse
            # than a clean boundary and far better than a result the
            # provider rejects outright.
            return _slice_to_budget(unit, budget=budget), 1
        used += cost
        out.append(unit)
    # Per-unit costs are a LOWER bound on the joined cost: tokenization
    # does not distribute over concatenation (a subword can span a unit
    # boundary), and the no-agent fallback ratio discards a fraction per
    # unit to floor division -- measured 7% under on 5.5k grep rows. So
    # re-count the REAL body and shrink from the tail until it fits: what
    # is returned is what was measured, not a sum of parts.
    body = "".join(out)
    while len(out) > 1 and approx_tokens(body) > budget:
        del out[-max(1, len(out) // 16) :]
        body = "".join(out)
    # One unit left and still over: same unsplittable case as above,
    # reached when the joined re-count exceeds what the per-unit sum said.
    if len(out) == 1 and approx_tokens(body) > budget:
        return _slice_to_budget(body, budget=budget), 1
    return body, len(out)


def _slice_to_budget(unit: str, *, budget: int) -> str:
    """Cut one unsplittable unit down to ``budget``, saying that it cut.

    Characters are the only handle left once a unit cannot be split, so
    the cut is made there and then VERIFIED in tokens: a chars-per-token
    estimate is exactly the guess this module exists to remove, and a
    minified line is far denser than any average.
    """

    def rendered(chars: int) -> str:
        return (
            f"{unit[:chars]}\n"
            f"... (truncated mid-line, {len(unit) - chars:,} chars omitted)"
        )

    ratio = max(1, len(unit) // max(1, approx_tokens(unit)))
    cut = max(1, budget * ratio)
    # Measure the RENDERED result, notice included: the notice is part of
    # what ships, so sizing only the prefix leaves the reply one notice
    # over budget -- the same append-after-counting mistake this module
    # exists to prevent.
    while cut > 1 and approx_tokens(rendered(cut)) > budget:
        cut //= 2
    return rendered(cut)


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


def truncate_to_budget(text: str) -> str:
    """Bound ``text`` to the active model's per-result token budget.

    The framework's backstop for tools that do not bound themselves. A
    tool that CAN paginate should call :func:`bound_by_tokens` instead and
    offer an ``offset``: this cuts mid-line and the remainder is
    unreachable, so it says so rather than trailing off silently.

    Args:
      text: Rendered tool output.

    Returns:
      result: ``text``, or a prefix plus a truncation notice.

    """
    budget = result_token_budget()
    if approx_tokens(text) <= budget:
        return text
    lines = text.splitlines(keepends=True)
    body, kept = bound_by_tokens(lines, budget=budget)
    if kept < len(lines):
        return f"{body}\n... (truncated, {len(lines) - kept} lines omitted)"
    # Every line was kept yet the whole still busts the budget: the body
    # is one (or few) very long line, which ``bound_by_tokens`` cannot
    # split without slicing a unit. Cut by characters here -- this is the
    # backstop, and an over-budget result is worse than a mid-line cut
    # that says so.
    ratio = max(1, len(text) // max(1, approx_tokens(text)))
    cut = max(1, budget * ratio)
    return f"{text[:cut]}\n... (truncated, {len(text) - cut} chars omitted)"


def to_result(result: str | ToolResult) -> ToolResult:
    """Normalize a tool return value to a ``ToolResult``.

    Strings wrap to ``ToolResult(call_id="", content=result)`` so the
    runtime can stamp the call_id from the originating ``ToolCall``.

    Args:
      result: Plain string or existing ToolResult.

    Returns:
      result: A ``ToolResult`` with the content set.

    """
    if isinstance(result, str):
        return ToolResult(call_id="", content=result)
    return result


def provider_not_allowed_result(
    name: str,
    allow: tuple[str, ...],
    parent_provider: str | None,
) -> ToolResult:
    """Build the shared ``provider not in allow list`` rejection.

    Used by AgentSpawn and AgentSelf when an LLM-requested provider
    falls outside the host's ``--allow-providers`` master knob. The
    parent's own provider is always implicitly allowed; callers gate
    on that *before* calling this helper.

    Args:
      name: The provider name the LLM tried to use.
      allow: Currently-allowed provider names.
      parent_provider: The parent agent's provider name, surfaced as
        the most likely fallback for the LLM to retry with. ``None``
        when no parent context exists (e.g. cold-start tests).

    Returns:
      result: A ``ToolResult`` with ``is_error=True`` and a
      retry-hint suggesting either the parent provider or one of the
      allowed names.

    """
    hint = (
        f" The parent agent is on {parent_provider!r}."
        if parent_provider is not None
        else ""
    )
    return ToolResult(
        call_id="",
        content=(
            f"Provider {name!r} is not in the allowed list"
            f" {list(allow)}.{hint} Pass ``--allow-providers`` at CLI"
            " startup to widen the set, or choose one of the allowed"
            " names."
        ),
        is_error=True,
    )


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
    return None if v is None else IntCodec.coerce(v, 0)


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

    Args:
      path: Tool-supplied filesystem path (relative or absolute).

    Returns:
      resolved: Absolute path string, or empty when ``path`` is empty.

    """
    if not path:
        return ""
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    return str(Path(get_tool_state().bash_cwd) / p)


async def run_sync(
    fn: Callable[..., str | ToolResult],
    **kwargs: object,
) -> ToolResult:
    """Run a sync function in a thread, returning a normalized ToolResult.

    ``fn`` may return a plain ``str`` (auto-wrapped into a
    ``ToolResult`` with empty ``call_id``) or a ``ToolResult``
    directly. Plain text content is truncated at
    the active model's token budget. Exceptions propagate to the caller;
    the ``_AgentTool`` wrapper or the runtime converts them to
    ``ToolResult(is_error=True)`` at the dispatch boundary.

    Usage in a tool's ``run``::

        async def run(self, args: Mapping[str, object]) -> ToolResult:
            return await run_sync(self._execute, **kwargs)

    Args:
      fn: Synchronous callable returning a string or ToolResult.
      **kwargs: Forwarded to ``fn``.

    Returns:
      result: Normalized ToolResult, content truncated if too large.

    """
    raw = await asyncio.to_thread(fn, **kwargs)
    result = to_result(raw)
    bounded = truncate_to_budget(result.content)
    if bounded != result.content:
        return dataclasses.replace(result, content=bounded)
    return result


class _ToolImpl:
    """Wraps a function as a Tool implementation."""

    __slots__ = (
        "_fn",
        "_is_async",
        "clearable_results",
        "description",
        "directive_schema",
        "name",
        "tool_id",
    )

    def __init__(
        self,
        fn: Callable[..., object],
        *,
        name: str | None = None,
        description: str | None = None,
        schema: JSON | None = None,
        clearable_results: bool = False,
    ) -> None:
        self._fn = fn
        self._is_async = inspect.iscoroutinefunction(fn)
        self.name = name or getattr(fn, "__name__", "tool")
        self.tool_id = f"application/x-tool-{self.name.lower()}"
        self.description = description or fn.__doc__ or ""
        hints = get_type_hints(fn, include_extras=True)
        hints.pop("return", None)
        # ``is None``, not truthiness: an explicit empty schema is a
        # legitimate override (a no-argument tool) and must survive.
        self.directive_schema = (
            json_freeze(_build_schema(fn, hints)) if schema is None else schema
        )
        self.clearable_results = clearable_results

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label for this tool invocation."""
        del args
        return str(self.name)

    def prompt(self) -> str:
        """Return supplemental prompt text for this tool."""
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run decorator-based tools in parallel (no serialization)."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Invoke the wrapped function and return a ToolResult.

        The wrapped function may return ``str`` (wrapped to
        ``ToolResult(content=...)``) or ``ToolResult`` directly. Text
        content over the model's token budget is truncated. Exceptions
        propagate; the ``_AgentTool`` wrapper at the dispatch boundary
        converts them to ``is_error=True``.

        Args:
          args: Parsed directive forwarded as keyword arguments.

        Returns:
          result: Normalized ``ToolResult`` with truncated content.

        """
        kwargs = dict(args)
        if self._is_async:
            raw = cast(
                str | ToolResult,
                await cast(Callable[..., Awaitable[object]], self._fn)(**kwargs),
            )
        else:
            raw = cast(
                str | ToolResult,
                await asyncio.to_thread(self._fn, **kwargs),
            )
        result = to_result(raw)
        bounded = truncate_to_budget(result.content)
        if bounded != result.content:
            return dataclasses.replace(result, content=bounded)
        return result


@overload
def tool(
    fn: Callable[..., object],
    /,
    *,
    name: str | None = ...,
    description: str | None = ...,
    schema: JSON | None = ...,
    clearable_results: bool = ...,
) -> _ToolImpl: ...


@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
    schema: JSON | None = ...,
    clearable_results: bool = ...,
) -> Callable[[Callable[..., object]], _ToolImpl]: ...


def tool(
    fn: Callable[..., object] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    schema: JSON | None = None,
    clearable_results: bool = False,
) -> _ToolImpl | Callable[[Callable[..., object]], _ToolImpl]:
    """Decorator to create a Tool from a function.

    Args:
      fn: The function to wrap (when used without arguments).
      name: Override the tool name (defaults to function name).
      description: Override description (defaults to docstring).
      schema: Override JSON schema (defaults to auto-generated).
      clearable_results: Whether provider context management may clear results.

    Returns:
      tool_impl: A Tool implementation wrapping fn.

    """
    if fn is not None:
        return _ToolImpl(
            fn,
            name=name,
            description=description,
            schema=schema,
            clearable_results=clearable_results,
        )
    return lambda f: _ToolImpl(
        f,
        name=name,
        description=description,
        schema=schema,
        clearable_results=clearable_results,
    )


# Convenience functions that delegate to current state.
def mark_read(
    path: str,
    offset: int = 0,
    limit: int = 0,
    *,
    last_lines: int = 0,
    content: str | None = None,
    mtime: float | None = None,
) -> None:
    """Record that a file has been read.

    Args:
      path: File path (resolved internally).
      offset: Starting line offset of the read.
      limit: Maximum lines read.
      last_lines: EOF-anchored line count.
      content: Full file text for content-based staleness checks, or ``None``
          when there is no text to cache. An empty string is cached as a real
          value (an empty file), distinct from ``None`` (no content provided).
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


# Process-wide registry of threading.Lock keyed by resolved file path.
# Mutating tools (Edit, Write, etc.) serialize their read-modify-write
# critical sections through this registry so concurrent coroutines -
# e.g. parallel subagents dispatching Edit on the same file - can't
# interleave between the staleness check and the write. Non-mutating
# Reads don't participate; parallel Reads are always safe.
#
# Same path → same lock → serialized across all mutating tools.
# Different paths → different locks → parallel.
#
# threading.Lock, not asyncio.Lock: the exclusion is process-wide, and an
# asyncio.Lock is only ever exclusive within one loop. Scoping this per
# loop does not make it loop-safe, it deletes the guarantee -- two agents
# on two loops would both enter. The lock is acquired inside the worker
# thread that performs the mutation (see ``locked_file_write``), so it
# blocks a pool thread and never an event loop.
#
# Dict growth is unbounded across a session but each entry is tiny
# (~200 bytes); hundreds of distinct files edited per session is
# bounded-memory trivia. No cleanup needed.
_file_write_locks: dict[str, threading.Lock] = {}
_file_write_locks_guard = threading.Lock()


def file_lock_key(path: str) -> str:
    """Return the canonical serialization key for a file path.

    Both the per-path write lock and the runtime's same-file cohort
    grouping key off this, so a grouped call and the lock it acquires
    always agree on identity (symlinks/``..`` normalized via
    ``resolve``).

    Args:
      path: File path (already cwd-resolved by ``resolve_tool_path``).

    Returns:
      key: Canonical absolute path string.

    """
    return str(Path(path).resolve())


def get_file_write_lock(path: str) -> threading.Lock:
    """Return the shared write lock for ``path``.

    Shared per path, process-wide: same path is the same lock on every
    loop and every thread, which is what serializes mutations.

    Acquire it via :func:`locked_file_write` rather than directly --
    holding a sync lock across an ``await`` blocks the event loop that
    must deliver the worker's completion.

    Args:
      path: File path (resolved internally).

    Returns:
      lock: Shared lock for the resolved path.

    """
    # Avoid ``setdefault`` here - it eagerly constructs a new Lock on
    # every call even when the key already exists, and throws it
    # away. Edit/Write hit this on every mutation; cheap to skip.
    resolved = file_lock_key(path)
    with _file_write_locks_guard:
        lock = _file_write_locks.get(resolved)
        if lock is None:
            lock = threading.Lock()
            _file_write_locks[resolved] = lock
        return lock


async def locked_file_write[T](path: str, mutate: Callable[[], T]) -> T:
    """Run ``mutate`` in a worker thread, holding ``path``'s write lock.

    The lock is taken inside the worker, so a blocked mutation parks a
    pool thread and the event loop keeps running. Taking it on the loop
    instead would deadlock: the loop would be blocked and so could never
    deliver the completion that releases it.

    Args:
      path: File path being mutated.
      mutate: Synchronous read-modify-write to run under the lock.

    Returns:
      result: Whatever ``mutate`` returned.

    """
    lock = get_file_write_lock(path)

    def _locked() -> T:
        with lock:
            return mutate()

    return await asyncio.to_thread(_locked)


def changed_files_context() -> str:
    """Context provider: detect externally modified files.

    Checks all cached files against disk mtime. For each changed file,
    generates a system-reminder with the full unified diff.

    Wire this into Agent via ``context_providers``::

        Agent(context_providers=[changed_files_context])

    Returns:
      context: System reminder text, or empty string if
          no files changed.

    """
    changes = get_tool_state().consume_changed_files()
    if not changes:
        return ""
    parts: list[str] = []
    for path, snippet in changes.items():
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
