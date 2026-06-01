"""Shared test utilities for sagent.

- ``FakeAgent`` / ``with_fake_agent``: stand-in agent satisfying
  ``AgentLike`` for unit-testing tools that touch
  ``current_agent_var``.
- ``MockModelCaps``: base capability flags for test model mocks.

Usage::

    from sagent.testing import with_fake_agent

    async def test_my_tool() -> None:
        with with_fake_agent() as agent:
            agent.tool_state.bash_cwd = "/tmp"
            result = await MyTool().run({"path": "."})
        assert not result.is_error
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

import itertools
import time

from sagent.agent import runtime as agent_runtime
from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.cost_tracker import CostTracker
from sagent.lib import token_count
from sagent.tools.core import (
    ToolState,
    current_agent_var,
    tool_state_var,
)
from sagent.types.model import ModelRequest, Pricing
from sagent.types.runtime import (
    AssistantMessage,
    Halt,
    Kill,
    ModelContextEvent,
    Quit,
    RuntimeEvent,
)


class MockModelCaps:
    """Base capability flags and helpers for **provider** ``Model`` mocks.

    Subclasses build mocks that satisfy the rich provider
    ``types.model.Model`` Protocol (``stream(request, ...) ->
    ModelResponse``). This base supplies the static capability flags
    (``supports_*``, ``max_*``) and the token-estimation helpers every
    provider mock needs; subclasses only have to add ``model_id``,
    ``max_request_tokens``, ``buffer``, and ``stream``. The method
    bodies here are the trivial deterministic defaults a unit test
    almost always wants; override per test as needed.

    Distinct from ``_NullModel`` in this module, which satisfies the
    leaner **runtime** ``agent.runtime.Model`` Protocol that
    ``AgentRuntime(model=...)`` consumes. The agent layer's
    ``_AgentModel`` (``agent/agent.py:1645``) bridges the rich provider
    surface to the lean runtime surface; tests pick whichever side they
    actually exercise.
    """

    max_response_tokens: int = 8_192
    supports_streaming: bool = True
    supports_thinking: bool = False
    supports_effort: bool = False
    supports_cache_control: bool = False
    valid_service_tiers: tuple[str, ...] = ()
    valid_latency_modes: tuple[str, ...] = ()
    supports_context_management: bool = False
    supports_persistent_retry: bool = False
    supports_account_auth: bool = False
    max_image_dim: int = 8000
    max_image_bytes: int = 5 * 1024 * 1024

    @property
    def pricing(self) -> Pricing:
        return Pricing()

    def approx_text_tokens(self, text: str) -> int:
        return len(text) // 4

    def approx_image_tokens(self, data: bytes) -> int:
        del data
        return 256

    def approx_request_tokens(self, request: ModelRequest) -> int:
        return token_count.approx_request_tokens(request, self)

    async def actual_text_tokens(self, text: str) -> int:
        return self.approx_text_tokens(text)

    async def actual_image_tokens(self, data: bytes) -> int:
        return self.approx_image_tokens(data)

    async def actual_request_tokens(self, request: ModelRequest) -> int:
        return self.approx_request_tokens(request)

    def is_context_overflow(self, error: Exception) -> bool:
        del error
        return False

    def is_retryable_provider_error(self, error: Exception) -> bool:
        del error
        return False


class _NullModel:
    """No-op runtime ``Model`` for tests that never hit model dispatch.

    Satisfies the lean ``agent.runtime.Model`` Protocol (the one
    ``AgentRuntime(model=...)`` accepts), not the rich provider
    ``types.model.Model``. Returns an empty ``AssistantMessage`` so a
    runtime that does end up calling ``stream`` makes forward progress
    instead of raising.
    """

    async def stream(
        self,
        history: list[ModelContextEvent],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        del history, on_text, on_thinking
        return AssistantMessage(text="")


def _new_runtime() -> agent_runtime.AgentRuntime:
    """Build a fresh ``AgentRuntime`` wired to a null model."""
    return agent_runtime.AgentRuntime(model=_NullModel())


@dataclass(slots=True, kw_only=True)
class FakeAgent:
    """Stand-in ``AgentLike`` for unit tests."""

    tool_state: ToolState = field(default_factory=ToolState)
    """Per-agent tool state (read cache, bash_cwd, etc.)."""

    runtime: agent_runtime.AgentRuntime = field(default_factory=_new_runtime)
    """Real ``AgentRuntime`` with a null model; its observers list
    captures every published event."""

    _bg: dict[str, BackgroundTaskEntry] = field(default_factory=dict)
    """Background task registry (mirrors ``Agent._bg``)."""

    _tool_registry: dict[str, tuple[str, float]] = field(default_factory=dict)
    """Detached runtime tool metadata (mirrors ``Agent._tool_registry``)."""

    _job_counter: Iterator[int] = field(default_factory=lambda: itertools.count(1))
    """Human job id counter (mirrors ``Agent._job_counter``)."""

    _job_ids_by_call_id: dict[str, str] = field(default_factory=dict)
    """Provider call id to human job id mapping."""

    _call_ids_by_job_id: dict[str, str] = field(default_factory=dict)
    """Human job id to provider call id mapping."""

    cost_tracker: CostTracker = field(default_factory=CostTracker)
    """Token + cost ledger (mirrors ``Agent.cost_tracker``)."""

    events: list[RuntimeEvent] = field(default_factory=list)
    """Every published event lands here for assertion."""

    def __post_init__(self) -> None:
        self.runtime.observers.append(self.events.append)

    @property
    def background(self) -> Mapping[str, BackgroundTaskEntry]:
        """Read view of explicit and detached background entries.

        Mirrors ``Agent.background`` (``agent/agent.py:570-586``) -- the
        merged view is rebuilt per access; the returned
        ``BackgroundTaskEntry`` for a detached call is a fresh value
        each call and is **not** identity-stable across reads. Tests
        that need to track the same entry across multiple ``background``
        reads should pin by ``queue_id`` / ``call_id``, not by ``is``.
        """
        merged: dict[str, BackgroundTaskEntry] = {}
        for call_id, task in self.runtime.detached.items():
            name, started = self._tool_registry.get(call_id, ("?", time.time()))
            job_id = self.job_id_for_call(call_id)
            merged[job_id] = BackgroundTaskEntry(
                task=task,
                tool_name=name,
                queue_id=job_id,
                call_id=call_id,
                started=started,
                kind="detached",
            )
        merged.update(self._bg)
        return merged

    def cancel_background(self, job_id: str) -> None:
        """Cancel and forget a visible background job, if present."""
        job = self._bg.pop(job_id, None)
        if job is None:
            job = self.background.get(job_id)
        if job is None:
            return
        if job.kind == "detached":
            call_id = job.call_id or job.queue_id
            _ = self.runtime.discard_detached(call_id)
            self._forget_job_id(call_id)
        if not job.task.done():
            job.task.cancel()

    def register_background(self, job_id: str, entry: BackgroundTaskEntry) -> None:
        """Add ``entry`` to the explicit-bg registry under ``job_id``."""
        self._bg[job_id] = entry

    def halt(self) -> None:
        """Queue a ``Halt`` runtime event (mirrors ``Agent.halt``).

        Pushes to ``runtime.inbox`` rather than publishing directly so
        the observer list is reserved for events the runtime itself
        publishes; tests that assert "runtime emitted Halt" can then
        distinguish runtime-sourced halts from this stub call.
        """
        self.runtime.inbox.push_back(Halt())

    def kill_tool(self, qid: str) -> None:
        """Cancel one outstanding tool task by human job id or call id.

        ``qid`` may be either id; ``cancel_background`` resolves either
        form via its ``_bg`` / ``background`` lookups, while the queued
        ``Kill`` carries the resolved provider ``call_id`` so the
        runtime's tool dispatch handler matches its registry key.
        Mirrors ``Agent.kill_tool`` (``agent/agent.py:887-896``).
        """
        call_id = self._call_id_for_job(qid)
        self.cancel_background(qid)
        self.runtime.inbox.push_back(Kill(call_id=call_id))

    def kill_all_tools(self) -> None:
        """Cancel every visible explicit background tool job.

        Filter (``kind == "tool" and not job.hidden``) matches the real
        Agent's ``_cancel_all_background`` (``agent/agent.py:1329-1333``),
        which ``Agent.kill_all_tools`` (``agent/agent.py:898-901``)
        delegates to. Detached and persistent-subagent jobs survive --
        ``shutdown(force=True)`` is the broader sweep.
        """
        for job_id, job in tuple(self._bg.items()):
            if job.kind == "tool" and not job.hidden:
                self.cancel_background(job_id)
        self.runtime.inbox.push_back(Kill())

    def shutdown(self, *, force: bool = False) -> None:
        """Stub for ``AgentLike.shutdown``; queue a quit event."""
        if force:
            self.kill_all_tools()
        self.runtime.inbox.push_back(Quit())

    def events_of[T: RuntimeEvent](self, cls: type[T]) -> list[T]:
        """Return all captured events that are instances of ``cls``."""
        return [e for e in self.events if isinstance(e, cls)]

    def job_id_for_call(self, call_id: str) -> str:
        """Return the stable human job id for a provider call id, minting on miss.

        Asymmetric with ``_call_id_for_job`` by design: a provider call
        id always wants a stable display id, so the lookup mints one on
        miss; a display id without a known call id falls back to itself
        (the runtime's id space is the same shape as the human one).
        Mirrors ``Agent.job_id_for_call``.
        """
        job_id = self._job_ids_by_call_id.get(call_id)
        if job_id is not None:
            return job_id
        job_id = f"job-{next(self._job_counter)}"
        self._job_ids_by_call_id[call_id] = job_id
        self._call_ids_by_job_id[job_id] = call_id
        return job_id

    def _call_id_for_job(self, job_or_call_id: str) -> str:
        """Resolve a human job id to its provider call id when known."""
        return self._call_ids_by_job_id.get(job_or_call_id, job_or_call_id)

    def _forget_job_id(self, call_id: str) -> None:
        """Forget a completed or cancelled provider call's human job id."""
        job_id = self._job_ids_by_call_id.pop(call_id, None)
        if job_id is not None:
            self._call_ids_by_job_id.pop(job_id, None)


@contextmanager
def with_fake_agent(
    *,
    tool_state: ToolState | None = None,
    agent: FakeAgent | None = None,
) -> Generator[FakeAgent]:
    """Install a ``FakeAgent`` in ``current_agent_var`` for the block.

    Pairs ``FakeAgent`` with ``tool_state_context`` so tools that call
    ``get_tool_state()`` see the fake's tool state, and tools that
    call ``current_agent_var.get()`` see the fake agent.

    Args:
      tool_state: Optional pre-built ``ToolState``. Ignored when
          ``agent`` is supplied (the supplied agent's state wins). When
          both are omitted, a fresh empty ``ToolState`` is used.
      agent: Optional pre-built ``FakeAgent``. Use this to inject a
          custom runtime/model or to preserve fake state across nested
          ``with_fake_agent`` blocks. When omitted, a fresh
          ``FakeAgent`` is constructed (from ``tool_state`` if given).

    Yields:
      agent: The active ``FakeAgent`` for the block.

    """
    if agent is None:
        agent = FakeAgent(tool_state=tool_state) if tool_state else FakeAgent()
    agent_token = current_agent_var.set(agent)
    state_token = tool_state_var.set(agent.tool_state)
    try:
        yield agent
    finally:
        tool_state_var.reset(state_token)
        current_agent_var.reset(agent_token)


__all__ = ["FakeAgent", "MockModelCaps", "with_fake_agent"]
