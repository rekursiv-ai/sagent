"""Agent-runtime state owned by the agent layer.

This module holds the cross-tool primitives the agent layer needs at
module-load time:

- ``ToolState`` / ``ReadCacheEntry`` -- per-agent tool bookkeeping
  (file tracking, cwd, bash parse cache, skill recency, depth).
- ``AgentLike`` -- the minimal Protocol both the real ``Agent`` and
  ``FakeAgent`` satisfy.
- ``agent_registry`` -- process-wide ``label -> AgentLike`` map.
- ``current_agent_var`` / ``tool_state_var`` / ``cost_root_var`` /
  ``max_depth_var`` / ``agent_path_var`` / ``agent_counter_var`` /
  ``agent_label_var`` -- ContextVars threaded through the spawn tree.

These live here -- not under ``tools/core`` -- because ``agent.agent``
imports them at module-load time and pulling ``tools/`` in that
early triggers the ``providers → agent → tools → providers`` import
cycle. ``tools.core`` re-exports them so tool authors keep the
familiar import surface.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

import contextvars
import dataclasses
import difflib
import itertools
import logging

from sagent.agent.runtime import AgentRuntime


if TYPE_CHECKING:
    from sagent.agent.background import BackgroundTaskEntry
    from sagent.agent.cost_tracker import CostTracker
    from sagent.tools.lib.bash import BashParseCache

logger = logging.getLogger(__name__)


class ReadCacheEntry(NamedTuple):
    """Cached read parameters and mtime for dedup and staleness checks."""

    offset: int
    """Starting line offset of the cached read."""

    limit: int
    """Line limit of the cached read."""

    last_lines: int
    """EOF-anchored line count of the cached read."""

    mtime: float
    """File mtime captured at the time of the read."""


@dataclasses.dataclass(slots=True, kw_only=True)
class ToolState:
    """Per-agent state for tools (file tracking, cwd).

    Each Agent creates one and sets it as the active context before
    dispatching tools. Tools access it via ``get_tool_state()``.
    """

    read_cache: dict[str, ReadCacheEntry] = dataclasses.field(default_factory=dict)
    """Resolved-path → ``ReadCacheEntry`` dedup + staleness cache."""

    start_cwd: str = dataclasses.field(default_factory=lambda: str(Path.cwd()))
    """Working directory when the agent started."""

    bash_cwd: str = ""
    """Current bash working directory (mutated by ``cd``). Defaults to
    ``start_cwd`` when left empty at construction."""

    additional_dirs: list[str] = dataclasses.field(default_factory=list)
    """Extra dirs whose ``AGENTS.md`` files are walked."""

    stats: dict[str, float | int] = dataclasses.field(default_factory=dict)
    """Per-request session stats written by the Agent."""

    bash_parse_cache: BashParseCache = dataclasses.field(default_factory=dict)
    """Per-request bashlex parse cache."""

    invoked_skills: set[str] = dataclasses.field(default_factory=set)
    """Names of skills exercised this session."""

    depth: int = 0
    """Subagent recursion depth."""

    _read_order: dict[str, str] = dataclasses.field(default_factory=dict, repr=False)
    """Resolved-path → original-path. Doubles as the "has been read" set
    and the recency-ordered file list."""

    _content_cache: dict[str, str] = dataclasses.field(default_factory=dict, repr=False)
    """Resolved-path → content captured at last read; powers change diffs."""

    def __post_init__(self) -> None:
        if not self.bash_cwd:
            self.bash_cwd = self.start_cwd

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
            return True
        try:
            current_content = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return True
        if current_content != cached_content:
            return True
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
    """Return the current agent's tool state.

    Returns:
      state: The active ToolState, or a module-level default.

    """
    return tool_state_var.get(_default_state)


@contextmanager
def tool_state_context(state: ToolState) -> Generator[None]:
    """Install ``state`` as the active tool state for the duration of the block.

    Args:
      state: ToolState to make active.

    Yields:
      control: yields once; the previous state is restored on exit.

    """
    token = tool_state_var.set(state)
    try:
        yield
    finally:
        tool_state_var.reset(token)


#
# These ContextVars carry cross-agent state down the spawn tree so
# AgentSpawn can look up its parent, enforce ``max_depth``, and
# aggregate cost into one shared ledger at the root.

current_agent_var: contextvars.ContextVar[AgentLike | None] = contextvars.ContextVar(
    "current_agent", default=None
)

max_depth_var: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "max_depth", default=None
)

cost_root_var: contextvars.ContextVar[CostTracker | None] = contextvars.ContextVar(
    "cost_root", default=None
)

agent_path_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agent_path", default=""
)

agent_counter_var: contextvars.ContextVar[itertools.count[int]] = (
    contextvars.ContextVar("agent_counter")
)

agent_label_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agent_label", default=""
)


class AgentLike(Protocol):
    """Minimal agent surface for tools that route messages between agents."""

    runtime: AgentRuntime
    """The agent's ``AgentRuntime``; exposes ``inbox``, ``cohort``,
    ``model_call``, ``compact_task``, and ``detached`` for routing
    and liveness inspection."""

    @property
    def background(self) -> Mapping[str, BackgroundTaskEntry]:
        """Return a read view of all backgrounded jobs (merged).

        Implementations may merge explicit-bg + cohort-detached +
        persistent + hidden-infra entries.

        Returns:
          jobs: ``queue_id -> BackgroundTaskEntry`` view of live tasks.

        """
        ...

    def cancel_background(self, job_id: str) -> None:
        """Remove ``job_id`` from the explicit-bg registry, if present.

        Cohort-detached tasks live on ``runtime.detached`` and clean
        themselves up; tools use this to retire explicit-bg entries.

        Args:
          job_id: Queue id of the registered task to remove.

        """
        ...

    def register_background(self, job_id: str, entry: BackgroundTaskEntry) -> None:
        """Add ``entry`` to the explicit-bg registry under ``job_id``.

        Spawning tools and the REPL pump use this to surface their
        long-lived tasks in ``background`` and survive ``halt``.

        Args:
          job_id: Queue id used as the registry key.
          entry: Background-task record to store.

        """
        ...


# Process-wide registry of live agents, keyed by label.
agent_registry: dict[str, AgentLike] = {}


def unique_registry_label(base: str) -> str:
    """Pick the first free ``{base}`` / ``{base}_1`` / ... key.

    Used by ``Agent._install_contextvars`` so two agents with the same
    default name (the orchestrator builds many ``Agent`` instances with
    ``name="Agent"``) don't overwrite each other's ``agent_registry``
    entry and break ``AgentSend`` routing.

    Args:
      base: Preferred registry key for this agent.

    Returns:
      label: ``base`` when unused, else the smallest free numeric suffix.

    """
    if base not in agent_registry:
        return base
    for i in itertools.count(1):
        candidate = f"{base}_{i}"
        if candidate not in agent_registry:
            return candidate
    raise AssertionError("unreachable")  # itertools.count is infinite
