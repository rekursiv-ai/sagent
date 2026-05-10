"""Smoke tests for the v3 Agent surface.

Covers the public API and the cross-cutting invariants the design doc
commits to: ``_start_foreground`` cancel-and-claim, ``run`` yielding
events for one turn, ``cancel`` interrupting the foreground task,
``_next_op`` draining at the top of each iteration, and ``shutdown``
exiting ``serve_forever``. Tool-level / provider-level behavior lives in
the corresponding ``tools/*_test.py`` and ``providers/*_test.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import asyncio
import contextlib

import pytest

from sagent.agent import Agent, PendingOp
from sagent.custom_types import (
    ErrorEvent,
    Event,
    Model,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    Pricing,
    TextMessage,
    TokenCount,
)
from sagent.lib.descriptors import QUIT_SENTINEL


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _model_response(
    text: str = "ok",
    *,
    stop_reason: str = "model_finished",
) -> ModelResponse:
    return ModelResponse(
        content=MultipartMessage(
            (TextMessage(text, "text/plain"),),
            "multipart/x-model-message",
        ),
        tokens=TokenCount(input_tokens=10, output_tokens=5),
        stop_reason=stop_reason,
    )


class _MockModel:
    """Minimal Model stub for tests not exercising provider behavior."""

    model_id: str = "mock"
    max_request_tokens: int = 1_000_000
    max_response_tokens: int = 8_000
    supports_thinking: bool = False
    supports_effort: bool = False
    supports_cache_control: bool = False
    supports_streaming: bool = True
    supports_persistent_retry: bool = False
    supports_account_auth: bool = False
    supports_context_management: bool = False
    max_image_dim: int = 0
    max_image_bytes: int = 0
    pricing: Pricing = Pricing()

    def __init__(self) -> None:
        self.responses: list[ModelResponse] = []

    def estimate_text_token_count(self, text: str) -> int:
        return len(text) // 4

    def estimate_image_token_count(self, data: bytes) -> int:
        del data
        return 0

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        del request
        if self.responses:
            return self.responses.pop(0)
        return _model_response()

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        del request, on_thinking
        resp = self.responses.pop(0) if self.responses else _model_response()
        if on_text is not None and isinstance(resp.content, MultipartMessage):
            for part in resp.content.content:
                if part.descriptor == "text/plain":
                    on_text(str(part.content))
        return resp

    def is_context_overflow(self, error: Exception) -> bool:
        del error
        return False


def _build_agent(model: object) -> Agent:
    return Agent(
        model=cast(Model, model),
        system="",
        tools=[],
        compactor=None,
    )


class TestLifecycle:
    async def test_shutdown_exits_serve_forever(self) -> None:
        agent = _build_agent(_MockModel())
        task = asyncio.create_task(agent.serve_forever())
        await asyncio.sleep(0)
        agent.shutdown()
        await asyncio.wait_for(task, timeout=1.0)

    async def test_quit_sentinel_exits_serve_forever(self) -> None:
        agent = _build_agent(_MockModel())
        task = asyncio.create_task(agent.serve_forever())
        await asyncio.sleep(0)
        _ = agent.inbox.put(TextMessage("", QUIT_SENTINEL))
        await asyncio.wait_for(task, timeout=1.0)


class TestForeground:
    async def test_cancel_clears_work(self) -> None:
        agent = _build_agent(_MockModel())
        started = asyncio.Event()

        async def long() -> None:
            started.set()
            await asyncio.sleep(10)

        bg = asyncio.create_task(agent._start_foreground(long()))
        await started.wait()
        assert agent.work is not None
        agent.cancel()
        # Drain bg; expect cancellation. pytest-anyio may swallow the
        # CancelledError at the harness boundary, so just confirm the
        # task ended and ``work`` was cleared.
        with contextlib.suppress(asyncio.CancelledError):
            await bg
        assert bg.done()
        assert agent.work is None


class TestRun:
    async def test_single_turn_yields_events_in_order(self) -> None:
        model = _MockModel()
        model.responses.append(_model_response("hello"))
        agent = _build_agent(model)
        events: list[Event] = [
            ev
            async for ev in agent.run(
                TextMessage("hi", "text/x-user-message"),
            )
        ]
        kinds = [type(e).__name__ for e in events]
        assert kinds[0] == "UserBarEvent"
        assert kinds[-1] == "TurnCompleteEvent"

    async def test_history_grows_after_run(self) -> None:
        model = _MockModel()
        model.responses.append(_model_response("hi back"))
        agent = _build_agent(model)
        async for _ev in agent.run(TextMessage("hi", "text/x-user-message")):
            pass
        descriptors = [m.descriptor for m in agent.history]
        assert descriptors == ["text/x-user-message", "multipart/x-model-message"]


class TestPendingOp:
    async def test_clear_op_wipes_history(self) -> None:
        model = _MockModel()
        agent = _build_agent(model)
        agent.history.append(TextMessage("seed", "text/x-user-message"))
        agent._next_op = PendingOp(kind="clear")
        async for _ev in agent.run(TextMessage("trigger", "text/x-user-message")):
            pass
        assert agent.history == []


class TestCancellation:
    async def test_run_yields_interrupted_on_cancel(self) -> None:
        model = _MockModel()
        hang_event = asyncio.Event()
        called = asyncio.Event()

        async def _hang(*args: object, **kwargs: object) -> ModelResponse:
            del args, kwargs
            called.set()
            await hang_event.wait()
            return _model_response()

        # Mock model swap; ty complains about implicit shadowing.
        model.stream = _hang  # ty: ignore[invalid-assignment]
        model.buffer = _hang  # ty: ignore[invalid-assignment]
        agent = _build_agent(model)
        events: list[Event] = []

        async def consume() -> None:
            # Async list comprehension would lose items on mid-iter cancel.
            async for ev in agent.run(
                TextMessage("hi", "text/x-user-message"),
            ):
                events.append(ev)  # noqa: PERF401

        consumer = asyncio.create_task(consume())
        await called.wait()
        agent.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await consumer
        kinds = [type(e).__name__ for e in events]
        assert "InterruptedEvent" in kinds


class TestPublish:
    def test_observer_exception_swallowed_and_logged(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        agent = _build_agent(_MockModel())
        events: list[Event] = []

        def good(ev: Event) -> None:
            events.append(ev)

        def bad(ev: Event) -> None:
            del ev
            raise RuntimeError("kaboom")

        agent.observers.extend([bad, good])
        with caplog.at_level("WARNING", logger="sagent.agent.agent"):
            agent.publish(ErrorEvent(text="hi"))
        # ``good`` still saw the event (fan-out continued past ``bad``).
        assert len(events) == 1
        # Warning is one-liner; no ERROR-level traceback dump.
        records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("RuntimeError" in r.getMessage() for r in records)
        assert not any(r.levelname == "ERROR" for r in caplog.records)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
