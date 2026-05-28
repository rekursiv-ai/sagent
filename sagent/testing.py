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

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

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
    """Base capability flags for test model mocks.

    Provides the Model protocol's property/method stubs so individual
    test files only need to add response logic. Does NOT satisfy the
    full Model protocol alone — concrete mocks must add ``model_id``,
    ``max_request_tokens``, ``buffer``, and ``stream``.
    """

    max_response_tokens: int = 8_192
    supports_streaming: bool = True
    supports_thinking: bool = False
    supports_effort: bool = False
    supports_cache_control: bool = False
    valid_service_tiers: tuple[str, ...] = ()
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
    """No-op ``Model`` for tests that never need a real model call."""

    async def stream(
        self,
        history: list[ModelContextEvent],
        system: str,
        tools: list[agent_runtime.Tool],
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None],
    ) -> AssistantMessage:
        del history, system, tools, on_text, on_thinking
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

    cost_tracker: CostTracker = field(default_factory=CostTracker)
    """Token + cost ledger (mirrors ``Agent.cost_tracker``)."""

    events: list[RuntimeEvent] = field(default_factory=list)
    """Every published event lands here for assertion."""

    def __post_init__(self) -> None:
        self.runtime.observers.append(self.events.append)

    @property
    def background(self) -> Mapping[str, BackgroundTaskEntry]:
        """Read view of explicit and detached background entries."""
        merged: dict[str, BackgroundTaskEntry] = {}
        for qid, task in self.runtime.detached.items():
            name, started = self._tool_registry.get(qid, ("?", time.time()))
            merged[qid] = BackgroundTaskEntry(
                task=task,
                tool_name=name,
                queue_id=qid,
                started=started,
                kind="detached",
            )
        merged.update(self._bg)
        return merged

    def cancel_background(self, job_id: str) -> None:
        """Remove ``job_id`` from the explicit-bg registry, if present."""
        self._bg.pop(job_id, None)

    def register_background(self, job_id: str, entry: BackgroundTaskEntry) -> None:
        """Add ``entry`` to the explicit-bg registry under ``job_id``."""
        self._bg[job_id] = entry

    def halt(self) -> None:
        """Stub for ``AgentLike.halt``; published as a ``Halt`` runtime event."""
        self.runtime.publish(Halt())

    def kill_all_tools(self) -> None:
        """Cancel every visible explicit background tool job."""
        for job_id, job in tuple(self._bg.items()):
            if job.kind == "tool" and not job.hidden:
                self._cancel_background(job_id)
        self.runtime.inbox.push_back(Kill())

    def _cancel_background(self, job_id: str) -> None:
        """Cancel and forget one explicit background job."""
        job = self._bg.pop(job_id, None)
        if job is not None and not job.task.done():
            job.task.cancel()

    def shutdown(self, *, force: bool = False) -> None:
        """Stub for ``AgentLike.shutdown``; queue a quit event."""
        if force:
            self.kill_all_tools()
        self.runtime.inbox.push_back(Quit())

    def events_of[T: RuntimeEvent](self, cls: type[T]) -> list[T]:
        """Return all captured events that are instances of ``cls``."""
        return [e for e in self.events if isinstance(e, cls)]


@contextmanager
def with_fake_agent(*, tool_state: ToolState | None = None) -> Generator[FakeAgent]:
    """Install a ``FakeAgent`` in ``current_agent_var`` for the block.

    Pairs ``FakeAgent`` with ``tool_state_context`` so tools that call
    ``get_tool_state()`` see the fake's tool state, and tools that
    call ``current_agent_var.get()`` see the fake agent.

    Args:
      tool_state: Optional pre-built ``ToolState``. When omitted, a
          fresh empty ``ToolState`` is used.

    Yields:
      agent: The active ``FakeAgent`` for the block.

    """
    fake = FakeAgent(tool_state=tool_state) if tool_state else FakeAgent()
    agent_token = current_agent_var.set(fake)
    state_token = tool_state_var.set(fake.tool_state)
    try:
        yield fake
    finally:
        tool_state_var.reset(state_token)
        current_agent_var.reset(agent_token)


__all__ = ["FakeAgent", "MockModelCaps", "with_fake_agent"]
