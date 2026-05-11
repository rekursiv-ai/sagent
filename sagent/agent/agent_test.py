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
from pathlib import Path
from typing import cast, override

import asyncio
import contextlib

import pytest

from sagent.agent import Agent, PendingOp
from sagent.agent.agent import _MAX_UNSAVED_EVENTS
from sagent.agent.cohort import Cohort, CohortMember
from sagent.custom_types import (
    Compactor,
    ErrorEvent,
    Event,
    Message,
    Model,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    Pricing,
    TextMessage,
    TokenCount,
    Tool,
)
from sagent.lib.descriptors import QUIT_SENTINEL
from sagent.lib.json import json_freeze
from sagent.lib.message import tool_call_message


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

    def is_retryable_provider_error(self, error: Exception) -> bool:
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
        agent.inbox.send(TextMessage("", QUIT_SENTINEL), source="quit")
        await asyncio.wait_for(task, timeout=1.0)

    async def test_serve_forever_publishes_error_and_continues(self) -> None:
        class _CrashingModel(_MockModel):
            @override
            async def stream(
                self,
                request: ModelRequest,
                on_text: Callable[[str], None] | None = None,
                on_thinking: Callable[[str], None] | None = None,
            ) -> ModelResponse:
                del request, on_text, on_thinking
                raise RuntimeError("provider exploded")

        agent = _build_agent(_CrashingModel())
        events: list[Event] = []
        error_seen = asyncio.Event()

        def observe(event: Event) -> None:
            events.append(event)
            if isinstance(event, ErrorEvent):
                error_seen.set()

        agent.observers.append(observe)
        task = asyncio.create_task(agent.serve_forever())
        done_wait = asyncio.create_task(error_seen.wait())
        try:
            agent.inbox.send(
                TextMessage("hi", "text/x-user-message"),
                source="user",
            )
            done, _ = await asyncio.wait(
                {task, done_wait},
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            assert done_wait in done
            assert not task.done()
            agent.inbox.send(TextMessage("", QUIT_SENTINEL), source="quit")
            await asyncio.wait_for(task, timeout=1.0)
        finally:
            done_wait.cancel()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        errs = [event for event in events if isinstance(event, ErrorEvent)]
        assert len(errs) == 1
        assert "turn failed" in errs[0].text
        assert "RuntimeError" in errs[0].text


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


class TestHaltSemantics:
    """``halt()`` arms the round body to requeue items + block_until_user."""

    @pytest.mark.real_sleep
    async def test_halt_during_model_call_requeues_and_blocks(self) -> None:
        model = _MockModel()
        hang = asyncio.Event()
        entered = asyncio.Event()

        async def _hang(*args: object, **kwargs: object) -> ModelResponse:
            del args, kwargs
            entered.set()
            await hang.wait()
            return _model_response()

        model.stream = _hang  # ty: ignore[invalid-assignment]
        model.buffer = _hang  # ty: ignore[invalid-assignment]
        agent = _build_agent(model)
        loop_task = asyncio.create_task(agent.serve_forever())
        agent.inbox.send(
            TextMessage("fix bug", "text/x-user-message"),
            source="user",
        )
        await entered.wait()
        agent.halt()
        # Wait for the round body to finish unwinding the cancellation.
        for _ in range(100):
            await asyncio.sleep(0.01)
            if agent.work is None:
                break
        # After halt, the round body has requeued the user item at the
        # front and armed ``block_until_user``. A peer-source arrival
        # should accumulate without waking the drain.
        agent.inbox.send(
            TextMessage("peer ping", "text/x-user-message"),
            source="Agent_X",
        )
        await asyncio.sleep(0.02)
        sources = [item.source for item in agent.inbox]
        assert sources[0] == "user"
        assert sources[1] == "Agent_X"
        # Cleanup.
        agent.shutdown(force=True)
        hang.set()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(loop_task, timeout=1.0)


class TestKillTool:
    """``kill_tool`` / ``kill_all_tools`` cancel cohort + bg tasks by qid."""

    @pytest.mark.real_sleep
    async def test_kill_tool_finds_member_in_active_cohort(self) -> None:
        agent = _build_agent(_MockModel())
        hang = asyncio.Event()

        async def _hang() -> Message:
            await hang.wait()
            return TextMessage("never", "text/plain")

        def _noop_promote(_: CohortMember) -> None:
            return

        emissions: list[list[Message]] = []
        cohort = Cohort(
            on_emit=emissions.append,
            on_promote_to_bg=_noop_promote,
        )
        agent._active_cohorts.append(cohort)
        member = CohortMember(
            tool_use_id="qid_1",
            tool_name="Bash",
            task=asyncio.create_task(_hang()),
        )
        cohort.add_member(member)
        try:
            assert agent.kill_tool("qid_1") is True
            for _ in range(20):
                await asyncio.sleep(0.01)
                if member.task.done():
                    break
            assert member.task.cancelled() or member.task.done()
        finally:
            hang.set()
            with contextlib.suppress(asyncio.CancelledError):
                await member.task

    @pytest.mark.real_sleep
    async def test_kill_all_tools_returns_count(self) -> None:
        agent = _build_agent(_MockModel())
        hang = asyncio.Event()

        async def _hang() -> Message:
            await hang.wait()
            return TextMessage("never", "text/plain")

        def _noop_emit(_: list[Message]) -> None:
            return

        def _noop_promote(_: CohortMember) -> None:
            return

        cohort = Cohort(
            on_emit=_noop_emit,
            on_promote_to_bg=_noop_promote,
        )
        agent._active_cohorts.append(cohort)
        members = [
            CohortMember(
                tool_use_id=f"qid_{i}",
                tool_name="Bash",
                task=asyncio.create_task(_hang()),
            )
            for i in range(2)
        ]
        for m in members:
            cohort.add_member(m)
        try:
            count = agent.kill_all_tools()
            assert count == 2
        finally:
            hang.set()
            for m in members:
                with contextlib.suppress(asyncio.CancelledError):
                    await m.task

    def test_kill_tool_returns_false_on_unknown_qid(self) -> None:
        agent = _build_agent(_MockModel())
        assert agent.kill_tool("nope") is False


class TestCohortEmitsToInbox:
    """Cohort completion sends a ``multipart/x-tool-batch-result`` to inbox."""

    async def test_cohort_bundle_drains_into_history(self) -> None:
        # Two-response sequence: first emits a tool call; second is final.
        tc = tool_call_message("t1", "Echo", json_freeze({"text": "hi"}))
        model = _MockModel()
        model.responses = [
            ModelResponse(
                content=MultipartMessage(
                    (TextMessage("", "text/plain"), tc),
                    "multipart/x-model-message",
                ),
                stop_reason="model_tool_use",
                tokens=TokenCount(input_tokens=10, output_tokens=5),
            ),
            _model_response("done"),
        ]

        class _EchoTool:
            name = "Echo"
            tool_id = "application/x-tool-echo"
            description = ""
            directive_schema: object = json_freeze(
                {"type": "object", "properties": {"text": {"type": "string"}}},
            )
            supports_microcompaction = False

            def summary(self, msg: Message) -> str:
                del msg
                return "Echo"

            def summary_result(self, result: Message) -> str | None:
                del result
                return None

            def prompt(self) -> str:
                return ""

            async def run(self, msg: Message) -> Message:
                del msg
                return TextMessage("echo-result", "text/plain")

        agent = Agent(
            model=cast(Model, model),
            tools=[cast(Tool, _EchoTool())],
        )
        events: list[Event] = [
            ev async for ev in agent.run(TextMessage("go", "text/x-user-message"))
        ]
        # Round 2 should have unpacked the cohort bundle into history with
        # the underlying multipart/x-tool-result paired with the tool_use.
        # Either shape is acceptable: a synthetic empty user-msg may have
        # been added for the second round (only-tool-results case) or
        # omitted. Both satisfy the API contract.
        kinds = [m.descriptor for m in agent.history]
        assert kinds in (
            [
                "text/x-user-message",
                "multipart/x-model-message",
                "multipart/x-tool-result",
                "text/x-user-message",
                "multipart/x-model-message",
            ],
            [
                "text/x-user-message",
                "multipart/x-model-message",
                "multipart/x-tool-result",
                "multipart/x-model-message",
            ],
        )
        names = [type(e).__name__ for e in events]
        assert "ToolResultEvent" in names
        assert names[-1] == "TurnCompleteEvent"

    async def test_post_drain_inbox_items_get_picked_up(self) -> None:
        # Regression: between ``serve_forever``'s ``await drain()`` and
        # ``_round_body`` actually running, a cohort's done-callback can
        # fire (the ``await asyncio.create_task(...)`` yields). The
        # bundle lands in the inbox AFTER drain captured ``items`` --
        # if the round body doesn't ``drain_nowait`` at the top, the
        # bundle stays parked and the next model call hits an unpaired
        # ``tool_use`` -> 400 from Anthropic.
        #
        # Directly exercise the race: prime history with an assistant
        # tool_use, place a matching tool-result bundle in the inbox,
        # and call ``_round_body(items=[])``. The model will be invoked
        # iff merge unpacked the bundle into history.
        model = _MockModel()
        model.responses = [_model_response("done")]
        agent = _build_agent(model)

        tc = tool_call_message("t1", "Echo", json_freeze({}))
        agent.history.append(
            MultipartMessage(
                (TextMessage("", "text/plain"), tc),
                "multipart/x-model-message",
            )
        )
        tool_result = MultipartMessage(
            (
                TextMessage("t1", "text/x-queue-id"),
                TextMessage("echo-result", "text/plain"),
            ),
            "multipart/x-tool-result",
        )
        bundle = MultipartMessage(
            (tool_result,),
            "multipart/x-tool-batch-result",
        )
        agent.inbox.send(bundle, source="tools")

        # Round body with items=[]: the fix must drain_nowait and pull
        # the bundle in. Without it, the bundle stays in the inbox and
        # the model is called with a dangling tool_use.
        await agent._round_body([])

        descriptors = [m.descriptor for m in agent.history]
        # Tool_use assistant message must be immediately followed by the
        # tool_result so the Anthropic adapter pairs them in one user
        # message; the model response lands at the end.
        assert descriptors[0] == "multipart/x-model-message"
        assert descriptors[1] == "multipart/x-tool-result"
        assert descriptors[-1] == "multipart/x-model-message"

    @pytest.mark.real_sleep
    async def test_post_drain_cohort_emit_is_picked_up(self) -> None:
        # Regression: a cohort done-callback that fires AFTER serve_forever's
        # drain returned but BEFORE the round body starts must not strand
        # its tool-result bundle in the inbox -- otherwise the next model
        # call sees a dangling tool_use and Anthropic rejects with 400.
        tc = tool_call_message("t1", "Slow", json_freeze({}))
        model = _MockModel()
        model.responses = [
            ModelResponse(
                content=MultipartMessage(
                    (TextMessage("", "text/plain"), tc),
                    "multipart/x-model-message",
                ),
                stop_reason="model_tool_use",
                tokens=TokenCount(input_tokens=10, output_tokens=5),
            ),
            _model_response("done"),
        ]

        gate = asyncio.Event()

        class _SlowTool:
            name = "Slow"
            tool_id = "application/x-tool-slow"
            description = ""
            directive_schema: object = json_freeze(
                {"type": "object", "properties": {}},
            )
            supports_microcompaction = False

            def summary(self, msg: Message) -> str:
                del msg
                return "Slow"

            def summary_result(self, result: Message) -> str | None:
                del result
                return None

            def prompt(self) -> str:
                return ""

            async def run(self, msg: Message) -> Message:
                del msg
                await gate.wait()
                return TextMessage("slow-result", "text/plain")

        agent = Agent(
            model=cast(Model, model),
            tools=[cast(Tool, _SlowTool())],
        )
        loop_task = asyncio.create_task(agent.serve_forever())
        agent.inbox.send(
            TextMessage("go", "text/x-user-message"),
            source="user",
        )
        # Wait for round 1 to spawn the cohort and return.
        for _ in range(200):
            await asyncio.sleep(0.005)
            if agent._active_cohorts and agent.work is None:
                break
        assert agent._active_cohorts, "cohort never spawned"
        # Release the slow tool so its task completes. The done-callback
        # will fire, the cohort emits, and its bundle goes into the inbox
        # while serve_forever is mid-`await drain()`.
        gate.set()
        # Wait for the second round to consume the bundle and the model
        # to return "done" -- without the post-drain drain_nowait fix,
        # this hangs / errors because the bundle is parked in the inbox.
        for _ in range(200):
            await asyncio.sleep(0.005)
            if any(m.descriptor == "multipart/x-tool-result" for m in agent.history):
                break
        # History must contain the tool_result immediately after the
        # assistant message bearing the tool_use.
        descriptors = [m.descriptor for m in agent.history]
        assert "multipart/x-tool-result" in descriptors
        tu_idx = descriptors.index("multipart/x-model-message")
        assert descriptors[tu_idx + 1] == "multipart/x-tool-result"
        agent.shutdown(force=True)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(loop_task, timeout=1.0)


class TestErrorEventPublication:
    """save/compaction/recompact failures must publish ErrorEvent."""

    def test_save_failure_publishes_error_event(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        agent = _build_agent(_MockModel())
        agent.session_dir = tmp_path
        agent._event_log = [{"ts": 0.0} for _ in range(_MAX_UNSAVED_EVENTS + 1)]
        events: list[Event] = []
        agent.observers.append(events.append)

        def boom(*, clear: bool = False) -> None:
            del clear
            raise OSError("disk full")

        monkeypatch.setattr(agent, "save_session", boom)
        agent.log_event("trigger")

        errs = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(errs) == 1
        assert "event log save failed" in errs[0].text
        assert "OSError" in errs[0].text

    async def test_compaction_failure_publishes_error_event(self) -> None:
        agent = _build_agent(_MockModel())
        events: list[Event] = []
        agent.observers.append(events.append)

        class _BoomCompactor:
            async def should_compact(
                self,
                input_tokens: int,
                max_request_tokens: int,
                max_response_tokens: int = 0,
            ) -> bool:
                del input_tokens, max_request_tokens, max_response_tokens
                return True

            async def compact(self, **kwargs: object) -> list[Message]:
                del kwargs
                raise RuntimeError("compactor exploded")

        agent.compactor = cast(Compactor, _BoomCompactor())
        ok = await agent._do_compact("")

        assert ok is False
        errs = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(errs) == 1
        assert "compaction failed" in errs[0].text
        assert "RuntimeError" in errs[0].text

    async def test_recompact_load_failure_publishes_error_event(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        agent = _build_agent(_MockModel())
        agent.session_dir = tmp_path
        agent.compactor = cast(Compactor, object())
        agent.compaction_state.compact_count = 1
        (tmp_path / "pre_compact_0.jsonl").write_text("{}\n", encoding="utf-8")
        events: list[Event] = []
        agent.observers.append(events.append)

        def boom(data: object) -> Message:
            del data
            raise KeyError("descriptor")

        monkeypatch.setattr(
            "sagent.agent.agent.load_message",
            boom,
        )
        await agent._do_recompact("")

        errs = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(errs) == 1
        assert "recompact transcript load failed" in errs[0].text
        assert "KeyError" in errs[0].text


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
