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
from types import MappingProxyType

import itertools
import time

from sagent.agent import runtime as agent_runtime
from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.cost_tracker import CostTracker
from sagent.agent.state import (
    ToolState,
    current_agent_var,
    tool_state_var,
)
from sagent.lib import token_count
from sagent.types.capability import (
    ModelCapability,
    ModelLimits,
    ModelSettings,
    ServiceTier,
    ThinkingEffort,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenCost,
    TokenCount,
    TokenPrice,
)
from sagent.types.model import (
    ModelRequest,
    UsageSnapshot,
)
from sagent.types.runtime import (
    AssistantMessage,
    Halt,
    Kill,
    ModelContextEvent,
    Quit,
    RuntimeEvent,
)


__all__ = ["FakeAgent", "MockModelCaps", "with_fake_agent"]


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
    ``_AgentModel`` bridges the rich provider surface to the lean runtime
    surface; tests pick whichever side they actually exercise.
    """

    model_id: str = "mock-model"
    max_request_tokens: int = 200_000
    max_response_tokens: int = 8_192
    supports_persistent_retry: bool = False
    supports_thinking: bool = False
    valid_efforts: tuple[ThinkingEffort, ...] = ()
    service_tiers: tuple[ServiceTier, ...] = ()

    @property
    def capability(self) -> ModelCapability:
        """Derive the capability row from this mock's configured limits."""
        thinking: frozenset[ThinkingEffort] = frozenset({"none", *self.valid_efforts})
        return ModelCapability(
            model_id=self.model_id,
            context=MappingProxyType(
                {
                    "": ModelLimits(
                        max_request_tokens=self.max_request_tokens,
                        max_response_tokens=self.max_response_tokens,
                        max_request_bytes=32 * 1024 * 1024,
                        max_image_edge_px=8000,
                        max_image_bytes=5 * 1024 * 1024,
                    )
                }
            ),
            prices=PriceCatalog(
                {PriceCatalogProduct(): TokenPrice()}
                | {
                    PriceCatalogProduct(service_tier=t): TokenPrice()
                    for t in self.service_tiers
                }
            ),
            thinking_effort=thinking,
            thinking_budget=(
                frozenset({"none", "auto", "fixed"})
                if self.supports_thinking
                else frozenset({"none"})
            ),
            thinking_output=(
                frozenset({"none", "text"})
                if self.supports_thinking
                else frozenset({"none"})
            ),
            service_tier={"auto", *self.service_tiers},
            retries_internally=self.supports_persistent_retry,
        )

    _settings: ModelSettings | None = None
    """Materialized on first read; see :attr:`settings`."""

    @property
    def settings(self) -> ModelSettings:
        """The mutable selection this mock carries.

        Materialized once and cached: settings are state the caller writes
        through, so returning a fresh object per read silently discarded
        every selection made against the mock. Lazy because the narrowest
        selection depends on ``capability``, which subclasses override.
        """
        if self._settings is None:
            self._settings = ModelSettings.narrowest(self.capability)
        return self._settings

    @property
    def limits(self) -> ModelLimits:
        """Ceilings of the selected context tag."""
        return self.settings.limits

    @property
    def tagged_model_id(self) -> str:
        """Display id carrying its context tag."""
        return f"{self.model_id}{self.settings.context}"

    def spend(self, tokens: TokenCount) -> TokenCost:
        """Price ``tokens`` at the tier these settings select."""
        prompt = tokens.request + tokens.cache_write + tokens.cache_read
        return (
            self.capability.prices[
                PriceCatalogProduct(
                    service_tier=self.settings.service_tier,
                    min_request_tokens=prompt,
                )
            ]
            * tokens
        )

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

    def usage_snapshot(self) -> UsageSnapshot | None:
        return None

    async def close(self) -> None:
        """No-op teardown -- the mock holds no resources.

        ``close`` is a required ``Model`` contract member; this mock
        satisfies it by returning immediately.
        """
        return


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
        publish: Callable[[RuntimeEvent], None],
    ) -> AssistantMessage:
        del history, publish
        return AssistantMessage(text="")


def _new_runtime() -> agent_runtime.AgentRuntime:
    """Build a fresh ``AgentRuntime`` wired to a null model."""
    return agent_runtime.AgentRuntime(model=_NullModel())


@dataclass(slots=True, kw_only=True)
class FakeAgent:
    """Stand-in ``AgentLike`` for unit tests."""

    tool_state: ToolState = field(default_factory=ToolState)
    """Per-agent tool state (read cache, bash_cwd, etc.)."""

    max_request_bytes: int = 32 * 1024 * 1024
    """Active model's request byte ceiling (mirrors ``Agent.max_request_bytes``)."""

    max_result_tokens: int = 50_000
    """Per-result token ceiling (mirrors ``Agent.max_result_tokens``).

    Defaults to what a 200k-token model derives (``window // 4``), so
    tools under test bound themselves the way they would in production.
    """

    chars_per_token: int = 4
    """Divisor backing :meth:`approx_text_tokens`."""

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

    def approx_text_tokens(self, text: str) -> int:
        """Ratio-based stand-in for the active model's tokenizer."""
        return len(text) // self.chars_per_token

    @property
    def background(self) -> Mapping[str, BackgroundTaskEntry]:
        """Read view of explicit and detached background entries.

        Mirrors ``Agent.background`` -- the
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
        Mirrors ``Agent.kill_tool``.
        """
        call_id = self._call_id_for_job(qid)
        self.cancel_background(qid)
        self.runtime.inbox.push_back(Kill(call_id=call_id))

    def kill_all_tools(self) -> None:
        """Cancel every visible explicit background tool job.

        Filter (``kind == "tool" and not job.hidden``) matches the real
        Agent's ``_cancel_all_background``, which ``Agent.kill_all_tools``
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
