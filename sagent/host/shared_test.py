"""Tests for the gate, holds and ledger the daemon serves every sagent on the host."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Literal

import asyncio
import time

from wesearch.fetch.gate import Admission, FileGate, GateState, SitePolicy

import pytest

from sagent.host.server import Server
from sagent.host.shared import (
    DaemonGate,
    DaemonLedger,
    DaemonLocks,
    Shared,
    _gate_answer,
)
from sagent.host.strings import ServeDirs
from sagent.host.testing import LocalHost
from sagent.host.wire import GateCall, Held, LedgerText, encoded
from sagent.providers.login import FileLedger, FileLocks, HostFiles, Logins, Spent


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


_POLICY = SitePolicy(host="account", cooldown_sec=5.0)
_LONG = SitePolicy(host="account", max_wait_sec=60.0)
_COOLING = SitePolicy(host="account", cooldown_sec=30.0, max_wait_sec=60.0)


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
async def test_a_waiter_sees_a_hold_set_mid_wait_without_sleeping_it_out(
    tmp_path: Path,
) -> None:
    """The daemon wakes every waiter on a key when another caller's 429 holds it."""
    paced = SitePolicy(host="account", interval_sec=30.0, max_wait_sec=60.0)
    async with _daemon(tmp_path) as (_, address):
        gate = DaemonGate(address)
        assert await gate.admitted("k", paced) == Admission()
        # The pace parks the next caller for 30s.
        waiting = asyncio.create_task(gate.admitted("k", paced))
        await asyncio.sleep(0.1)
        assert not waiting.done()
        await asyncio.to_thread(gate.hold, "k", until=time.time() + 600.0)
        async with asyncio.timeout(5):
            refused = await waiting

    assert refused.admitted is False
    assert refused.wait_sec > 590.0


@pytest.mark.asyncio
async def test_a_hold_lifted_mid_wait_admits_at_once(tmp_path: Path) -> None:
    async with _daemon(tmp_path) as (_, address):
        gate = DaemonGate(address)
        await asyncio.to_thread(gate.trip, "k", _COOLING)
        waiting = asyncio.create_task(gate.admitted("k", _COOLING))
        await asyncio.sleep(0.1)
        assert not waiting.done()
        # Another caller's probe succeeded: every waiter goes at once.
        await asyncio.to_thread(gate.reset, "k")
        async with asyncio.timeout(5):
            assert await waiting == Admission()


@pytest.mark.asyncio
async def test_a_change_to_another_key_leaves_a_waiter_waiting() -> None:
    shared = Shared()
    waiting = asyncio.create_task(shared._changed_within("k", 0, 60.0))
    try:
        await asyncio.sleep(0)
        async with shared._changed:
            shared._versions["other"] = 1
            shared._changed.notify_all()
        await asyncio.sleep(0)
        assert not waiting.done()
        async with shared._changed:
            shared._versions["k"] = 1
            shared._changed.notify_all()
        async with asyncio.timeout(5):
            assert await waiting is True
        assert await shared._changed_within("k", 1, 0.0) is False
        assert await shared._changed_within("k", 0, 60.0) is True
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_waiter_that_hangs_up_stops_the_daemons_wait(tmp_path: Path) -> None:
    async with _daemon(tmp_path) as (server, address):
        gate = DaemonGate(address)
        await asyncio.to_thread(gate.hold, "k", until=time.time() + 30.0)
        waiting = asyncio.create_task(gate.admitted("k", _LONG))
        await asyncio.sleep(0.1)
        assert server.waiting() == 1
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await asyncio.sleep(0.2)
        assert server.waiting() == 0


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
    async with DaemonLocks(tmp_path / "nobody.sock").held(tmp_path / "a.lock"):
        pass
    ledger = DaemonLedger(tmp_path / "nobody.sock")
    await asyncio.to_thread(ledger.write, "{}")
    assert await asyncio.to_thread(ledger.read) == "{}"


@pytest.mark.asyncio
async def test_an_answer_of_the_wrong_kind_is_refused(tmp_path: Path) -> None:
    """The daemon's socket is a trust boundary: what it says is checked."""
    address = tmp_path / "liar.sock"

    async def liar(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        _ = await reader.readline()
        writer.write(encoded(Held()))
        await writer.drain()
        writer.close()

    async with await asyncio.start_unix_server(liar, path=address):
        with pytest.raises(TypeError, match="answered Held"):
            await asyncio.to_thread(DaemonGate(address).states)
        with pytest.raises(TypeError, match="answered Held"):
            await asyncio.to_thread(DaemonLedger(address).read)
        with pytest.raises(TypeError, match="answered Held"):
            await DaemonGate(address).admitted("k", _LONG)


@pytest.mark.asyncio
async def test_a_hold_answered_wrongly_is_refused(tmp_path: Path) -> None:
    address = tmp_path / "liar.sock"

    async def liar(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        _ = await reader.readline()
        writer.write(encoded(LedgerText(text="")))
        await writer.drain()
        _ = await reader.read()

    async with await asyncio.start_unix_server(liar, path=address):
        with pytest.raises(TypeError, match="answered LedgerText"):
            async with DaemonLocks(address).held(tmp_path / "a.lock"):
                pass


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
        assert await asyncio.to_thread(gate.states) == FileGate().states()
        assert (await gate.admitted("k", _LONG)).admitted is False
        assert await asyncio.to_thread(DaemonLedger(address).read) == ""
        async with DaemonLocks(address).held(tmp_path / "a.lock"):
            with pytest.raises(TimeoutError):
                async with (
                    asyncio.timeout(0.05),
                    FileLocks().held(tmp_path / "a.lock"),
                ):
                    pass


@pytest.mark.parametrize("op", ["admit", "admitted", "trip"])
@pytest.mark.asyncio
async def test_a_gate_call_naming_no_policy_is_refused(
    op: Literal["admit", "admitted", "trip"],
) -> None:
    with pytest.raises(ValueError, match=f"^A gate's {op} needs a policy\\.$"):
        await Shared().answer(GateCall(op=op, key="k"))


@pytest.mark.asyncio
async def test_a_wait_asked_for_without_waiting_is_refused() -> None:
    with pytest.raises(ValueError, match="admitted call waits"):
        _gate_answer(GateCall(op="admitted", key="k"), FileGate())


@pytest.mark.asyncio
async def test_a_hold_excludes_every_other_holder_until_it_ends(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "a.lock"
    order: list[str] = []
    async with _daemon(tmp_path) as (_, address):
        locks = DaemonLocks(address)

        async def take(name: str, held: FileLocks | DaemonLocks) -> None:
            async with held.held(lock):
                order.append(f"{name} in")
                await asyncio.sleep(0.02)
                order.append(f"{name} out")

        async with locks.held(lock):
            others = [
                asyncio.create_task(take("daemon", locks)),
                # A process with no daemon takes the host's file lock: the same hold.
                asyncio.create_task(take("file", FileLocks())),
            ]
            await asyncio.sleep(0.05)
            assert order == []
        await asyncio.gather(*others)

    assert sorted(order[0::2]) == ["daemon in", "file in"]
    assert [entry.split()[1] for entry in order] == ["in", "out", "in", "out"]


@pytest.mark.asyncio
async def test_a_holder_that_goes_away_lets_its_hold_go(tmp_path: Path) -> None:
    lock = tmp_path / "a.lock"
    async with _daemon(tmp_path) as (_, address):
        locks = DaemonLocks(address)
        entered = asyncio.Event()

        async def hold_forever() -> None:
            async with locks.held(lock):
                entered.set()
                await asyncio.Event().wait()

        holder = asyncio.create_task(hold_forever())
        await entered.wait()
        holder.cancel()
        await asyncio.gather(holder, return_exceptions=True)
        async with asyncio.timeout(5), locks.held(lock):
            pass


@pytest.mark.asyncio
async def test_the_logins_ledger_through_the_daemon_is_the_hosts(
    tmp_path: Path,
) -> None:
    async with _daemon(tmp_path) as (_, address):
        shared = HostFiles(locks=DaemonLocks(address), ledger=DaemonLedger(address))
        logins = Logins(tmp_path / "auth.json", shared=shared)
        spent = Spent(window="5-hour", model="sol-6.1", until=time.time() + 600.0)
        await logins.spend("default", spent)
        ranked = await asyncio.to_thread(logins.ranked, "sol-6.1", now=time.time())
        text = await asyncio.to_thread(DaemonLedger(address).read)

    assert ranked == (("default", spent),)
    assert text == FileLedger().read()
    assert Logins(tmp_path / "auth.json").ranked("sol-6.1", now=time.time()) == ranked


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
