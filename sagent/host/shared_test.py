"""Tests for daemon gate routing, waits, transport failures, and cleanup."""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import partial
from typing import TYPE_CHECKING, Literal, cast
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import asyncio
import socket

from wesearch.fetch.gate import Admission, FileGate, Gate, GatePolicy, GateState

import pytest

from sagent.host import shared
from sagent.host.shared import DaemonGate, ServedGate, _gate_answer, host_gate
from sagent.host.wire import GateAnswer, GateCall, Ping, encoded
from sagent.lib.userdirs import state_dir


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from pathlib import Path


_POLICY = GatePolicy(cooldown_sec=5.0)
_LONG = GatePolicy(max_wait_sec=60.0)


def test_admission_and_states_use_the_shared_host_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refused = Admission(wait_sec=600.0, admitted=False, reason="held")
    admitting = Mock(side_effect=[refused, Admission()])
    states = {"work": GateState(held_until=600.0)}
    listing = Mock(return_value=states)
    files = Mock(spec=FileGate, admit=admitting, states=listing)
    factory = Mock(return_value=files)
    monkeypatch.setattr(shared, "FileGate", factory)
    first, replacement = (
        DaemonGate(tmp_path / "daemon.sock"),
        DaemonGate(
            tmp_path / "replacement.sock",
        ),
    )

    assert first.admit("work", _POLICY) is refused
    assert replacement.admit("home", _POLICY) == Admission()
    assert first.states() is replacement.states() is states
    assert factory.call_args_list == [call(), call()]
    assert admitting.call_args_list == [call("work", _POLICY), call("home", _POLICY)]
    assert listing.call_args_list == [call(), call()]


@pytest.mark.parametrize("daemon", [False, True])
def test_changes_reach_the_daemon_or_fall_back_to_the_host_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    daemon: bool,
) -> None:
    address = tmp_path / "daemon.sock"
    gate = DaemonGate(address)
    files = Mock(spec=FileGate)
    gate._files = cast(FileGate, files)
    asking = Mock(return_value=GateAnswer() if daemon else None)
    monkeypatch.setattr(shared, "_asked", asking)

    gate.trip("k", _POLICY, challenge_url="https://site.example/")
    gate.hold("k", until=600.0)
    gate.release_probe("k")
    gate.reset("k")
    gate.reset("k", forget=True)
    gate.reset("k", clear_hold=True)

    assert asking.call_args_list == [
        call(
            address,
            GateCall(
                op="trip",
                key="k",
                policy=_POLICY,
                challenge_url="https://site.example/",
            ),
        ),
        call(address, GateCall(op="hold", key="k", until=600.0)),
        call(address, GateCall(op="release_probe", key="k")),
        call(address, GateCall(op="reset", key="k")),
        call(address, GateCall(op="reset", key="k", forget=True)),
        call(address, GateCall(op="reset", key="k", clear_hold=True)),
    ]
    assert files.mock_calls == (
        []
        if daemon
        else [
            ("trip", ("k", _POLICY), {"challenge_url": "https://site.example/"}),
            ("hold", ("k",), {"until": 600.0}),
            ("release_probe", ("k",), {}),
            ("reset", ("k",), {"forget": False, "clear_hold": False}),
            ("reset", ("k",), {"forget": True, "clear_hold": False}),
            ("reset", ("k",), {"forget": False, "clear_hold": True}),
        ]
    )


async def _noted_wait(
    ready: asyncio.Event,
    wait: Callable[[Callable[[], bool]], Awaitable[bool]],
    predicate: Callable[[], bool],
) -> bool:
    ready.set()
    return await wait(predicate)


@asynccontextmanager
async def _waiting(
    served: ServedGate,
) -> AsyncGenerator[asyncio.Task[GateAnswer]]:
    ready = asyncio.Event()
    observing = AsyncMock(
        side_effect=partial(_noted_wait, ready, served._changed.wait_for),
    )
    with patch.object(served._changed, "wait_for", observing):
        async with asyncio.TaskGroup() as tasks:
            waiting = tasks.create_task(
                served.answer(GateCall(op="admitted", key="k", policy=_LONG)),
            )
            try:
                async with asyncio.timeout(5):
                    await ready.wait()
                    yield waiting
            finally:
                waiting.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gate_call", "expected_change", "result"),
    [
        (
            GateCall(op="reset", key="k", forget=True, clear_hold=True),
            ("reset", ("k",), {"forget": True, "clear_hold": True}),
            Admission(),
        ),
        (
            GateCall(op="hold", key="k", until=600.0),
            ("hold", ("k",), {"until": 600.0}),
            Admission(wait_sec=600.0, admitted=False, reason="held"),
        ),
        (
            GateCall(
                op="trip",
                key="k",
                policy=_POLICY,
                challenge_url="https://site.example/",
            ),
            ("trip", ("k", _POLICY), {"challenge_url": "https://site.example/"}),
            Admission(wait_sec=100.0, admitted=False, reason="blocked"),
        ),
        (
            GateCall(op="release_probe", key="k"),
            ("release_probe", ("k",), dict[str, object]()),
            Admission(),
        ),
    ],
    ids=["reset", "hold", "trip", "release-probe"],
)
async def test_each_change_wakes_its_waiters_and_forwards_the_complete_request(
    gate_call: GateCall,
    expected_change: tuple[str, tuple[object, ...], dict[str, object]],
    result: Admission,
) -> None:
    admitting = Mock(
        side_effect=[
            Admission(wait_sec=30.0, admitted=gate_call.op == "hold"),
            result,
        ],
    )
    files = Mock(spec=Gate, admit=admitting)
    served = ServedGate(gate=cast(Gate, files))
    async with _waiting(served) as waiting:
        assert not waiting.done()
        assert await served.answer(gate_call) == GateAnswer()
        assert (await waiting).admission == result
    assert files.mock_calls == [
        ("admit", ("k", _LONG), {}),
        expected_change,
        ("admit", ("k", _LONG), {}),
    ]
    assert served._versions == {"k": 1}


@pytest.mark.asyncio
async def test_a_change_to_another_key_leaves_a_waiter_waiting() -> None:
    served = ServedGate()
    assert await served._changed_within("absent", [0], 0.0) is False
    waiting = asyncio.create_task(served._changed_within("k", [0], 60.0))
    try:
        await asyncio.sleep(0)
        async with served._changed:
            served._versions["other"] = 1
            served._changed.notify_all()
        await asyncio.sleep(0)
        assert not waiting.done()
        async with served._changed:
            served._versions["k"] = 1
            served._changed.notify_all()
        async with asyncio.timeout(5):
            assert await waiting is True
        assert await served._changed_within("k", [1], 0.0) is False
        assert await served._changed_within("k", [0], 60.0) is True
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


def _admit_with_a_concurrent_change(
    served: ServedGate,
    admitting: Mock,
    key: str,
    policy: GatePolicy,
) -> Admission:
    del policy
    served._versions[key] = 1
    admitting.side_effect = None
    admitting.return_value = Admission()
    return Admission(wait_sec=30.0)


@pytest.mark.asyncio
async def test_a_change_made_during_admission_is_not_missed() -> None:
    admitting = Mock()
    served = ServedGate(gate=cast(Gate, Mock(admit=admitting)))
    admitting.side_effect = partial(_admit_with_a_concurrent_change, served, admitting)
    with patch.object(
        served._changed,
        "wait",
        AsyncMock(side_effect=AssertionError("A change during admission was missed.")),
    ):
        assert (
            await served.answer(GateCall(op="admitted", key="k", policy=_LONG))
            == GateAnswer()
        )
    assert admitting.call_args_list == [call("k", _LONG), call("k", _LONG)]


def test_the_hosts_gate_names_the_users_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    address = tmp_path / "daemon.sock"
    choosing = Mock(return_value=address)
    monkeypatch.setattr(shared, "socket_address", choosing)
    assert host_gate()._address == address
    choosing.assert_called_once_with(
        state_dir() / "rekursiv-ai" / "sagent" / "serve" / "daemon.sock",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["connect", "drain", "read", "eof", "answer"])
async def test_async_admission_closes_its_connection_and_falls_back_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    gate = DaemonGate(tmp_path / "daemon.sock")
    admission = Admission(wait_sec=600.0, admitted=False, reason="held")
    fallback = AsyncMock(return_value=admission)
    gate._files = cast(FileGate, Mock(admitted=fallback))
    reader = asyncio.StreamReader()
    if outcome == "answer":
        reader.feed_data(encoded(GateAnswer(admission=admission)))
    elif outcome == "read":
        reader.set_exception(ConnectionResetError("read failed"))
    else:
        reader.feed_eof()
    writing, closing = Mock(), Mock()
    draining = AsyncMock(side_effect=BrokenPipeError() if outcome == "drain" else None)
    writer = cast(
        asyncio.StreamWriter,
        Mock(write=writing, drain=draining, close=closing),
    )
    connecting = AsyncMock(return_value=(reader, writer))
    if outcome == "connect":
        connecting.side_effect = FileNotFoundError()
    monkeypatch.setattr(asyncio, "open_unix_connection", connecting)

    assert await gate.admitted("k", _LONG) == admission

    connecting.assert_awaited_once_with(gate._address)
    assert writing.call_args_list == (
        []
        if outcome == "connect"
        else [
            call(encoded(GateCall(op="admitted", key="k", policy=_LONG))),
        ]
    )
    assert draining.await_count == (0 if outcome == "connect" else 1)
    assert closing.call_args_list == ([] if outcome == "connect" else [call()])
    assert fallback.await_args_list == (
        [] if outcome == "answer" else [call("k", _LONG)]
    )


@pytest.mark.asyncio
async def test_a_waiter_that_hangs_up_closes_without_falling_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = asyncio.StreamReader()
    writing, closing = Mock(), Mock()
    writer = cast(
        asyncio.StreamWriter,
        Mock(write=writing, drain=AsyncMock(), close=closing),
    )
    gate = DaemonGate(tmp_path / "daemon.sock")
    fallback = AsyncMock()
    gate._files = cast(FileGate, Mock(admitted=fallback))
    monkeypatch.setattr(
        asyncio,
        "open_unix_connection",
        AsyncMock(return_value=(reader, writer)),
    )
    waiting = asyncio.create_task(gate.admitted("k", _LONG))
    try:
        await asyncio.sleep(0)
        writing.assert_called_once_with(
            encoded(GateCall(op="admitted", key="k", policy=_LONG)),
        )
        assert not waiting.done()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        closing.assert_called_once_with()
        fallback.assert_not_awaited()
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.asyncio
async def test_an_answer_of_the_wrong_kind_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = DaemonGate(tmp_path / "liar.sock")
    monkeypatch.setattr(shared, "_asked", Mock(return_value=Ping()))
    with pytest.raises(TypeError, match="answered Ping"):
        gate.hold("k", until=1.0)
    reader = asyncio.StreamReader()
    reader.feed_data(encoded(Ping()))
    closing = Mock()
    writer = cast(
        asyncio.StreamWriter,
        Mock(write=Mock(), drain=AsyncMock(), close=closing),
    )
    monkeypatch.setattr(
        asyncio,
        "open_unix_connection",
        AsyncMock(return_value=(reader, writer)),
    )
    with pytest.raises(TypeError, match="answered Ping"):
        await gate.admitted("k", _LONG)
    closing.assert_called_once_with()


def test_a_sync_request_reads_a_fragmented_reply_and_closes_the_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = GateCall(op="hold", key="k", until=17.0)
    answer = GateAnswer()
    line = encoded(answer)
    receiving = Mock(side_effect=[line[:5], line[5:]])
    connection = Mock(recv=receiving)
    entering, exiting = Mock(return_value=connection), Mock(return_value=False)
    factory = Mock(return_value=MagicMock(__enter__=entering, __exit__=exiting))
    monkeypatch.setattr(socket, "socket", factory)
    address = tmp_path / "daemon.sock"

    assert shared._asked(address, request) == answer

    factory.assert_called_once_with(socket.AF_UNIX, socket.SOCK_STREAM)
    assert connection.mock_calls == [
        ("settimeout", (30.0,), {}),
        ("connect", (str(address),), {}),
        ("sendall", (encoded(request),), {}),
        ("recv", (65_536,), {}),
        ("recv", (65_536,), {}),
    ]
    exiting.assert_called_once_with(None, None, None)


@pytest.mark.parametrize("outcome", ["connect", "sendall", "recv", "eof"])
def test_an_unanswered_or_broken_sync_connection_returns_none_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    connection = Mock(recv=Mock(return_value=b""))
    if outcome != "eof":
        connection.configure_mock(**{f"{outcome}.side_effect": ConnectionResetError()})
    exiting = Mock(return_value=False)
    monkeypatch.setattr(
        socket,
        "socket",
        Mock(
            return_value=MagicMock(
                __enter__=Mock(return_value=connection),
                __exit__=exiting,
            ),
        ),
    )
    assert shared._asked(tmp_path / "daemon.sock", GateCall(op="hold")) is None
    exiting.assert_called_once_with(None, None, None)


@pytest.mark.parametrize("op", ["admitted", "trip"])
@pytest.mark.asyncio
async def test_a_gate_call_naming_no_policy_is_refused(
    op: Literal["admitted", "trip"],
) -> None:
    with pytest.raises(ValueError, match=f"^A gate's {op} needs a policy\\.$"):
        await ServedGate().answer(GateCall(op=op, key="k"))


def test_a_wait_asked_for_without_waiting_is_refused() -> None:
    with pytest.raises(ValueError, match="admitted call waits"):
        _gate_answer(GateCall(op="admitted", key="k"), FileGate())


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
