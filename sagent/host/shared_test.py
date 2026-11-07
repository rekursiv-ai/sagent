"""Tests for the rate gate the daemon serves every sagent on the host."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal, override

import asyncio
import socket
import threading
import time

from wesearch.fetch.gate import Admission, FileGate, GatePolicy, GateState

import pytest

from sagent.host.server import Server
from sagent.host.shared import DaemonGate, ServedGate, _gate_answer, host_gate
from sagent.host.strings import ServeDirs
from sagent.host.testing import LocalHost
from sagent.host.wire import GateCall, Ping, encoded
from sagent.lib.posix.sockets import socket_address
from sagent.lib.userdirs import state_dir


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


_POLICY = GatePolicy(cooldown_sec=5.0)
_LONG = GatePolicy(max_wait_sec=60.0)
_COOLING = GatePolicy(cooldown_sec=30.0, max_wait_sec=60.0)


class _Asked(FileGate):
    """The host's files, noting each time a waiter asks them."""

    def __init__(self) -> None:
        super().__init__()
        self.asked = 0
        self._each = threading.Condition()

    @override
    def admit(self, key: str, policy: GatePolicy) -> Admission:
        admission = super().admit(key, policy)
        with self._each:
            self.asked += 1
            self._each.notify_all()
        return admission

    async def until_asked(self, count: int) -> None:
        """Wait until the gate has been asked ``count`` times in all."""
        await asyncio.to_thread(self._until_asked, count)

    def _until_asked(self, count: int) -> None:
        with self._each:
            assert self._each.wait_for(lambda: self.asked >= count, timeout=5)


@asynccontextmanager
async def _daemon(tmp_path: Path) -> AsyncGenerator[tuple[Server, Path]]:
    """Serve a daemon in this process, on a socket under ``tmp_path``."""
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    address = tmp_path / "daemon.sock"
    server = Server(
        dirs,
        listen=[],
        local_socket=address,
        start_host=LocalHost(tools={}),
    )
    async with server:
        yield server, address


@pytest.mark.asyncio
async def test_a_hold_through_the_daemon_holds_every_caller_on_the_host(
    tmp_path: Path,
) -> None:
    async with _daemon(tmp_path) as (_, address):
        gate = DaemonGate(address)
        until = time.time() + 600.0
        await asyncio.to_thread(gate.hold, "openai-sub@work", until=until)
        refused = await asyncio.to_thread(gate.admit, "openai-sub@work", _POLICY)
        other = await asyncio.to_thread(gate.admit, "openai-sub@home", _POLICY)

    assert (refused.admitted, refused.wait_sec > 590.0) == (False, True)
    assert refused.reason.startswith("cooling down for 600s")
    assert other == Admission()
    # A process with no daemon reads the same state from the host's files.
    assert FileGate().states()["openai-sub@work"].held_until == until


@pytest.mark.asyncio
async def test_each_gate_call_reaches_the_daemons_gate(tmp_path: Path) -> None:
    async with _daemon(tmp_path) as (_, address):
        gate = DaemonGate(address)
        await asyncio.to_thread(
            gate.trip,
            "site::",
            _POLICY,
            challenge_url="https://site.example/",
        )
        tripped = await asyncio.to_thread(gate.states)
        await asyncio.to_thread(gate.release_probe, "site::")
        await asyncio.to_thread(gate.reset, "site::", forget=True)
        forgotten = await asyncio.to_thread(gate.states)

    assert tripped["site::"].failures == 1
    assert tripped["site::"].challenge_url == "https://site.example/"
    assert forgotten == {}


@pytest.mark.asyncio
async def test_a_waiter_sees_a_hold_set_mid_wait_without_sleeping_it_out() -> None:
    """The daemon wakes every waiter on a key when another caller's 429 holds it."""
    asked = _Asked()
    served = ServedGate(gate=asked)
    paced = GatePolicy(interval_sec=30.0, max_wait_sec=60.0)
    first = await served.answer(GateCall(op="admitted", key="k", policy=paced))
    assert first.admission == Admission()
    # The pace parks the next caller for 30s.
    waiting = asyncio.create_task(
        served.answer(GateCall(op="admitted", key="k", policy=paced)),
    )
    await asyncio.wait_for(asked.until_asked(2), timeout=5)
    await served.answer(GateCall(op="hold", key="k", until=time.time() + 600.0))
    async with asyncio.timeout(5):
        refused = (await waiting).admission

    assert refused.admitted is False
    assert refused.wait_sec > 590.0


@pytest.mark.asyncio
async def test_a_hold_lifted_mid_wait_admits_at_once() -> None:
    """Another caller's probe succeeded: every waiter goes at once."""
    asked = _Asked()
    served = ServedGate(gate=asked)
    await served.answer(GateCall(op="trip", key="k", policy=_COOLING))
    waiting = asyncio.create_task(
        served.answer(GateCall(op="admitted", key="k", policy=_COOLING)),
    )
    await asyncio.wait_for(asked.until_asked(1), timeout=5)
    await served.answer(GateCall(op="reset", key="k"))
    async with asyncio.timeout(5):
        assert (await waiting).admission == Admission()


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


@pytest.mark.asyncio
async def test_a_waiter_that_hangs_up_stops_the_daemons_wait(tmp_path: Path) -> None:
    """Gone, it asks the gate no more, though its hold ends."""
    async with _daemon(tmp_path) as (server, address):
        asked = _Asked()
        server._gate.gate = asked
        await asyncio.to_thread(asked.hold, "k", until=time.time() + 0.3)
        waiting = asyncio.create_task(DaemonGate(address).admitted("k", _LONG))
        await asyncio.wait_for(asked.until_asked(1), timeout=5)
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        # Still waiting, the daemon would ask again as the hold ends.
        await asyncio.sleep(0.6)
        assert asked.asked == 1


@pytest.mark.parametrize("change", ["reset", "hold", "trip", "release_probe"])
@pytest.mark.asyncio
async def test_each_change_through_the_daemon_wakes_its_waiters(
    tmp_path: Path,
    change: str,
) -> None:
    """Made by the daemon, not the files beside it: only the daemon wakes a waiter."""
    async with _daemon(tmp_path) as (server, address):
        asked = _Asked()
        server._gate.gate = asked
        if change == "release_probe":
            # A lapsed lock's probe, taken: the waiter waits on the probe's lease.
            asked.trip("k", GatePolicy(cooldown_sec=0.0))
            assert asked.admit("k", _COOLING).admitted
        else:
            asked.trip("k", _COOLING)
        gate = DaemonGate(address)
        waiting = asyncio.create_task(gate.admitted("k", _COOLING))
        await asyncio.wait_for(asked.until_asked(asked.asked + 1), timeout=5)
        match change:
            case "reset":
                await asyncio.to_thread(gate.reset, "k", forget=True)
            case "hold":
                await asyncio.to_thread(gate.hold, "k", until=time.time() + 600.0)
            case "trip":
                longer = GatePolicy(cooldown_sec=100.0)
                await asyncio.to_thread(gate.trip, "k", longer)
            case _:
                await asyncio.to_thread(gate.release_probe, "k")
        async with asyncio.timeout(5):
            admission = await waiting

    # Woken, the waiter asked again at once: freed, or refused a lock past its wait.
    assert admission.admitted is (change in ("reset", "release_probe"))
    if change == "reset":
        assert "k" not in FileGate().states()


@pytest.mark.asyncio
async def test_the_hosts_gate_is_the_daemon_at_the_users_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every sagent of a user reaches the one daemon its state directory names."""
    serve = state_dir() / "rekursiv-ai" / "sagent" / "serve"
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=serve)
    async with Server(
        dirs,
        listen=[],
        local_socket=socket_address(serve / "daemon.sock"),
        start_host=LocalHost(tools={}),
    ) as server:
        asked = _Asked()
        monkeypatch.setattr(server._gate, "gate", asked)
        assert (await host_gate().admitted("k", _LONG)).admitted
    assert asked.asked == 1


@pytest.mark.asyncio
async def test_a_wait_in_the_daemon_is_for_the_key_named(tmp_path: Path) -> None:
    async with _daemon(tmp_path) as (_, address):
        gate = DaemonGate(address)
        await asyncio.to_thread(gate.hold, "k", until=time.time() + 600.0)
        refused = await gate.admitted("k", _LONG)
        free = await gate.admitted("other", _LONG)

    assert (refused.admitted, refused.reason[:16]) == (False, "cooling down for")
    assert free == Admission()


@pytest.mark.asyncio
async def test_with_no_daemon_the_gate_answers_from_the_hosts_files(
    tmp_path: Path,
) -> None:
    """A daemon gone, as mid-exec, leaves the same state: the files it kept."""
    gate = DaemonGate(tmp_path / "nobody.sock")
    await asyncio.to_thread(gate.hold, "k", until=5_000_000_000.0)
    refused = await gate.admitted("k", _LONG)
    assert refused.admitted is False
    assert await asyncio.to_thread(gate.states) == FileGate().states()
    assert (await asyncio.to_thread(gate.admit, "free", _LONG)).admitted
    await asyncio.to_thread(gate.trip, "t", _LONG, challenge_url="https://t.example/")
    assert FileGate().states()["t"].challenge_url == "https://t.example/"
    await asyncio.to_thread(gate.release_probe, "t")
    assert FileGate().states()["t"].probe_until == 0.0
    await asyncio.to_thread(gate.reset, "t", forget=True)
    assert "t" not in FileGate().states()


@pytest.mark.asyncio
async def test_an_answer_of_the_wrong_kind_is_refused(tmp_path: Path) -> None:
    """The daemon's socket is a trust boundary: what it says is checked."""
    address = tmp_path / "liar.sock"

    async def liar(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        _ = await reader.readline()
        writer.write(encoded(Ping()))
        await writer.drain()
        writer.close()

    async with await asyncio.start_unix_server(liar, path=address):
        with pytest.raises(TypeError, match="answered Ping"):
            await asyncio.to_thread(DaemonGate(address).hold, "k", until=1.0)
        with pytest.raises(TypeError, match="answered Ping"):
            await DaemonGate(address).admitted("k", _LONG)


@pytest.mark.asyncio
@pytest.mark.parametrize("daemon", [False, True])
async def test_an_explicit_reset_clears_the_same_hold_with_or_without_a_daemon(
    tmp_path: Path,
    daemon: bool,
) -> None:
    async with _daemon(tmp_path) as (_, address):
        gate = DaemonGate(address if daemon else tmp_path / "absent.sock")
        until = time.time() + 432_000.0
        await asyncio.to_thread(gate.hold, "k", until=until)
        await asyncio.to_thread(gate.reset, "k")
        assert FileGate().states()["k"].held_until == until
        await asyncio.to_thread(gate.reset, "k", clear_hold=True)
        assert (await gate.admitted("k", _LONG)).admitted
        assert FileGate().states()["k"].held_until == 0.0


@pytest.mark.asyncio
async def test_a_reset_keeps_the_pace_unless_told_to_forget(tmp_path: Path) -> None:
    async with _daemon(tmp_path) as (_, address):
        gate = DaemonGate(address)
        await asyncio.to_thread(gate.trip, "k", _COOLING)
        await asyncio.to_thread(gate.reset, "k")
        kept = await asyncio.to_thread(gate.states)

    assert kept == {"k": GateState()}


@pytest.mark.asyncio
async def test_a_daemon_that_closes_unanswered_is_answered_for_by_the_files(
    tmp_path: Path,
) -> None:
    """As a daemon stopping mid-call: the files hold the state it kept."""
    address = tmp_path / "mute.sock"
    FileGate().hold("k", until=5_000_000_000.0)

    async def mute(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        _ = await reader.readline()
        writer.close()

    async with await asyncio.start_unix_server(mute, path=address):
        gate = DaemonGate(address)
        await asyncio.to_thread(gate.hold, "j", until=5_000_000_000.0)
        assert (await gate.admitted("k", _LONG)).admitted is False
    assert FileGate().states()["j"].held_until == 5_000_000_000.0


@pytest.mark.asyncio
async def test_a_daemon_that_dies_mid_request_is_answered_for_by_the_files(
    tmp_path: Path,
) -> None:
    """A connection reset, not closed: the request was never read."""
    address = tmp_path / "dies.sock"
    FileGate().hold("k", until=5_000_000_000.0)
    listening = threading.Event()

    def dies_unread() -> None:
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(address))
            listener.listen()
            listening.set()
            for _ in range(2):
                accepted = listener.accept()[0]
                time.sleep(0.2)
                accepted.close()

    dying = threading.Thread(target=dies_unread)
    dying.start()
    assert listening.wait(5)
    gate = DaemonGate(address)
    await asyncio.to_thread(gate.hold, "j", until=5_000_000_000.0)
    refused = await gate.admitted("k", _LONG)
    await asyncio.to_thread(dying.join, 5)
    assert FileGate().states()["j"].held_until == 5_000_000_000.0
    assert refused.admitted is False


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


@pytest.mark.asyncio
async def test_what_the_daemon_shares_outlives_its_handoff(tmp_path: Path) -> None:
    """An exec replaces the daemon; the gate's state is the host's files."""
    async with _daemon(tmp_path) as (server, address):
        gate = DaemonGate(address)
        await asyncio.to_thread(gate.hold, "k", until=5_000_000_000.0)
        snapshot = await server.handoff()
        await server.resume(snapshot)
        states = await asyncio.to_thread(gate.states)

    assert states == {"k": GateState(held_until=5_000_000_000.0)}


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
