"""Unit tests for ``Cohort``."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import cast

import asyncio

import pytest

from sagent.agent.cohort import Cohort, CohortMember
from sagent.custom_types import (
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.descriptors import MultipartDescriptor


pytestmark = [pytest.mark.anyio, pytest.mark.real_sleep]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _real_result(use_id: str, content: str) -> Message:
    return MultipartMessage(
        (
            TextMessage(use_id, "text/x-queue-id"),
            TextMessage(content, "text/plain"),
        ),
        cast(MultipartDescriptor, "multipart/x-tool-result"),
    )


def _result_text(msg: Message) -> str:
    assert isinstance(msg, MultipartMessage)
    for part in msg.content:
        if part.descriptor in ("text/plain", "text/x-error") and isinstance(
            part,
            TextMessage,
        ):
            return part.content
    return ""


def _result_qid(msg: Message) -> str:
    assert isinstance(msg, MultipartMessage)
    for part in msg.content:
        if part.descriptor == "text/x-queue-id" and isinstance(part, TextMessage):
            return part.content
    return ""


def _has_error(msg: Message) -> bool:
    assert isinstance(msg, MultipartMessage)
    return any(p.descriptor == "text/x-error" for p in msg.content)


async def _completes_with(use_id: str, content: str) -> Message:
    return _real_result(use_id, content)


async def _hangs() -> Message:
    await asyncio.Event().wait()
    return _real_result("never", "never")


async def _raises() -> Message:
    raise RuntimeError("boom")


def _make_cohort() -> tuple[Cohort, list[list[Message]], list[CohortMember]]:
    emissions: list[list[Message]] = []
    promotions: list[CohortMember] = []
    cohort = Cohort(
        on_emit=emissions.append,
        on_promote_to_bg=promotions.append,
    )
    return cohort, emissions, promotions


def _make_member(
    use_id: str,
    name: str,
    coro_factory: Callable[[], Coroutine[object, object, Message]],
) -> CohortMember:
    return CohortMember(
        tool_use_id=use_id,
        tool_name=name,
        task=asyncio.create_task(coro_factory()),
    )


class TestNaturalEmission:
    async def test_emits_when_all_members_done(self) -> None:
        cohort, emissions, promotions = _make_cohort()
        cohort.add_member(
            _make_member("t1", "Read", lambda: _completes_with("t1", "A"))
        )
        cohort.add_member(
            _make_member("t2", "Read", lambda: _completes_with("t2", "B"))
        )
        # Yield enough for done callbacks to fire.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(emissions) == 1
        assert [_result_text(m) for m in emissions[0]] == ["A", "B"]
        assert promotions == []
        assert cohort.emitted

    async def test_no_emission_until_all_done(self) -> None:
        cohort, emissions, _promotions = _make_cohort()
        gate = asyncio.Event()

        async def _wait_and_return() -> Message:
            await gate.wait()
            return _real_result("slow", "done")

        cohort.add_member(
            _make_member("fast", "Read", lambda: _completes_with("fast", "F"))
        )
        cohort.add_member(_make_member("slow", "Read", _wait_and_return))
        await asyncio.sleep(0)
        assert emissions == []
        gate.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(emissions) == 1

    async def test_failed_member_becomes_error_result(self) -> None:
        cohort, emissions, _promotions = _make_cohort()
        cohort.add_member(
            _make_member("t1", "Read", lambda: _completes_with("t1", "ok"))
        )
        cohort.add_member(_make_member("t2", "Bash", _raises))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(emissions) == 1
        results = emissions[0]
        assert _result_qid(results[1]) == "t2"
        assert _has_error(results[1])
        assert "RuntimeError" in _result_text(results[1])


class TestForceClose:
    async def test_unfinished_member_promoted_with_placeholder(self) -> None:
        cohort, emissions, promotions = _make_cohort()
        done_member = _make_member("t1", "Read", lambda: _completes_with("t1", "real"))
        await asyncio.sleep(0)  # let t1 complete naturally before adding to cohort
        # Re-spawn -- the prior add wouldn't have happened, so just add fresh.
        cohort, emissions, promotions = _make_cohort()
        cohort.add_member(
            _make_member("t1", "Read", lambda: _completes_with("t1", "real"))
        )
        running_member = _make_member("t2", "Bash", _hangs)
        cohort.add_member(running_member)
        await asyncio.sleep(0)  # let t1 complete; t2 still hangs
        cohort.force_close()
        assert len(emissions) == 1
        # First member: real result; second: placeholder + promotion.
        results = emissions[0]
        assert _result_text(results[0]) == "real"
        assert _result_text(results[1]) == "[Running in background: Bash]"
        assert len(promotions) == 1
        assert promotions[0].tool_use_id == "t2"
        _ = running_member  # silence unused
        _ = done_member

    async def test_idempotent(self) -> None:
        cohort, emissions, _promotions = _make_cohort()
        cohort.add_member(
            _make_member("t1", "Read", lambda: _completes_with("t1", "ok"))
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # Natural emission already fired; force_close should be a no-op.
        cohort.force_close()
        assert len(emissions) == 1

    async def test_after_natural_emit_is_noop(self) -> None:
        cohort, emissions, promotions = _make_cohort()
        cohort.add_member(
            _make_member("t1", "Read", lambda: _completes_with("t1", "a"))
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(emissions) == 1
        cohort.force_close()
        assert len(emissions) == 1
        assert promotions == []

    async def test_all_unfinished_emits_all_placeholders(self) -> None:
        cohort, emissions, promotions = _make_cohort()
        cohort.add_member(_make_member("t1", "Bash", _hangs))
        cohort.add_member(_make_member("t2", "WebFetch", _hangs))
        await asyncio.sleep(0)
        cohort.force_close()
        assert len(emissions) == 1
        results = emissions[0]
        assert _result_text(results[0]) == "[Running in background: Bash]"
        assert _result_text(results[1]) == "[Running in background: WebFetch]"
        assert {p.tool_use_id for p in promotions} == {"t1", "t2"}


class TestMemberCancellation:
    async def test_member_cancelled_yields_cancelled_result(self) -> None:
        cohort, emissions, _promotions = _make_cohort()
        m1 = _make_member("t1", "Read", lambda: _completes_with("t1", "ok"))
        m2 = _make_member("t2", "Bash", _hangs)
        cohort.add_member(m1)
        cohort.add_member(m2)
        await asyncio.sleep(0)  # let m1 complete
        _ = m2.task.cancel()  # /kill behavior
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(emissions) == 1
        results = emissions[0]
        assert _result_text(results[0]) == "ok"
        assert _result_text(results[1]) == "[Cancelled by user]"
        assert _has_error(results[1])


class TestAddMemberAfterEmit:
    async def test_raises(self) -> None:
        cohort, _emissions, _promotions = _make_cohort()
        cohort.add_member(
            _make_member("t1", "Read", lambda: _completes_with("t1", "a"))
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # Cohort is emitted now.
        late = _make_member("t2", "Read", lambda: _completes_with("t2", "b"))
        with pytest.raises(RuntimeError, match="already emitted"):
            cohort.add_member(late)
        _ = late.task.cancel()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
