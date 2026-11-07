"""What the per-user daemon serves every sagent on the host: a gate, holds, a ledger.

Each is the daemon's backend of a protocol whose default is the host's files:
``DaemonGate`` a ``Gate``, ``DaemonLocks`` a ``Locks``, ``DaemonLedger`` a
``Ledger``. A client sends one request on its own connection to the daemon's
owner-only socket, and the daemon answers from the host's files, so a process
with no daemon shares one state with those that have one.

What the daemon adds is the wait. An ``admitted`` call stays open in the daemon
(``Shared``) until the caller is admitted, and each change to a key -- another
caller's 429 holding it, a probe's success lifting it -- wakes every caller
waiting on that key to ask again. Waiting on the files alone, a caller sees
such a change only when its own wait ends.

A ``Hold`` lasts as long as its connection: a client that dies lets it go. A
daemon that is gone, as mid-exec, is answered for by the host's files directly.

Typical usage example:

  address = socket_address(state / "daemon.sock")
  catalog.gate, catalog.locks = DaemonGate(address), DaemonLocks(address)
  admission = await catalog.gate.admitted(key, policy)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import asyncio
import contextlib
import socket

from wesearch.fetch.gate import Admission, FileGate, Gate

from sagent.host.wire import (
    MAX_LINE_BYTES,
    GateAnswer,
    GateCall,
    Held,
    Hold,
    LedgerCall,
    LedgerText,
    decoded,
    encoded,
    received,
)
from sagent.providers.login import FileLedger, FileLocks, Ledger


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

    from wesearch.fetch.gate import GateState, SitePolicy

    from sagent.host.wire import Message


__all__ = [
    "DaemonGate",
    "DaemonLedger",
    "DaemonLocks",
    "Shared",
]


@dataclass(slots=True, kw_only=True, eq=False)
class Shared:
    """The daemon's side: its gate and ledger, and every caller waiting on a key.

    Attributes:
      gate: Where the gate's state is kept; the host's files.
      ledger: Where the logins ledger is kept; the host's file.

    """

    gate: Gate = field(default_factory=FileGate)
    ledger: Ledger = field(default_factory=FileLedger)
    _changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    _versions: dict[str, int] = field(default_factory=dict[str, int])
    _waiting: int = 0

    @property
    def waiting(self) -> int:
        """Return how many ``admitted`` calls wait in the daemon now."""
        return self._waiting

    async def answer(self, call: GateCall | LedgerCall) -> Message:
        """Return what the daemon answers ``call`` with.

        Args:
          call: An owner's request.

        Returns:
          reply: A ``GateAnswer`` or a ``LedgerText``.

        Raises:
          ValueError: An ``admit`` or ``trip`` names no policy.

        """
        if isinstance(call, LedgerCall):
            return await asyncio.to_thread(_ledger_answer, call, self.ledger)
        if call.op == "admitted":
            return GateAnswer(admission=await self._admitted(call.key, _policy(call)))
        reply = await asyncio.to_thread(_gate_answer, call, self.gate)
        if call.op in ("trip", "hold", "reset", "release_probe"):
            async with self._changed:
                self._versions[call.key] = self._versions.get(call.key, 0) + 1
                self._changed.notify_all()
        return reply

    async def _admitted(self, key: str, policy: SitePolicy) -> Admission:
        """Wait until ``key`` admits the caller, asking again at each change to it."""
        self._waiting += 1
        try:
            while True:
                seen = self._versions.get(key, 0)
                admission = await asyncio.to_thread(self.gate.admit, key, policy)
                if not admission.admitted and admission.wait_sec > policy.max_wait_sec:
                    return admission
                # Woken during a pace's wait, the slot it reserved goes unused:
                # the key is paced a little slower, never faster.
                if admission.wait_sec and await self._changed_within(
                    key,
                    seen,
                    admission.wait_sec,
                ):
                    continue
                if admission.admitted:
                    return Admission()
        finally:
            self._waiting -= 1

    async def _changed_within(self, key: str, seen: int, seconds: float) -> bool:
        """Wait ``seconds``; return early, ``True``, once ``key`` changes past ``seen``."""
        async with self._changed:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._changed.wait_for(lambda: self._versions.get(key, 0) != seen),
                    seconds,
                )
            return self._versions.get(key, 0) != seen


class DaemonGate:
    """A ``Gate`` the daemon at ``address`` keeps for every sagent on the host.

    Args:
      address: The daemon's socket.

    """

    def __init__(self, address: Path) -> None:
        self._address = address
        self._files = FileGate()

    def admit(self, key: str, policy: SitePolicy) -> Admission:
        """Admit the caller, make it the probe, or say how long to wait."""
        reply = self._ask(GateCall(op="admit", key=key, policy=policy))
        return self._files.admit(key, policy) if reply is None else reply.admission

    async def admitted(self, key: str, policy: SitePolicy) -> Admission:
        """Wait, in the daemon, until the caller is admitted to ``key``.

        Args:
          key: What is paced.
          policy: How it is paced, and the longest lock waited out.

        Returns:
          admission: ``Admission()`` once admitted; else, unwaited, the refusal
            of a lock or probe longer than ``policy.max_wait_sec``.

        Raises:
          TypeError: The daemon answered with something else.

        """
        try:
            reader, writer = await asyncio.open_unix_connection(self._address)
        except OSError:
            return await self._files.admitted(key, policy)
        try:
            writer.write(encoded(GateCall(op="admitted", key=key, policy=policy)))
            await writer.drain()
            reply = await received(reader)
        finally:
            writer.close()
        if reply is None:
            # The daemon went away mid-wait; the files hold what it kept.
            return await self._files.admitted(key, policy)
        return _expected(reply, GateAnswer).admission

    def trip(self, key: str, policy: SitePolicy, *, challenge_url: str = "") -> None:
        """Lock ``key`` for twice the previous lock, capped; the first uses the base."""
        call = GateCall(op="trip", key=key, policy=policy, challenge_url=challenge_url)
        if self._ask(call) is None:
            self._files.trip(key, policy, challenge_url=challenge_url)

    def hold(self, key: str, *, until: float) -> None:
        """Lock ``key`` until ``until``, a time the server named; no probe follows."""
        if self._ask(GateCall(op="hold", key=key, until=until)) is None:
            self._files.hold(key, until=until)

    def release_probe(self, key: str) -> None:
        """Free the probe; the failure said nothing about a block."""
        if self._ask(GateCall(op="release_probe", key=key)) is None:
            self._files.release_probe(key)

    def reset(self, key: str, *, forget: bool = False) -> None:
        """Unlock every caller of ``key``; ``forget`` also drops its pace."""
        if self._ask(GateCall(op="reset", key=key, forget=forget)) is None:
            self._files.reset(key, forget=forget)

    def states(self) -> dict[str, GateState]:
        """Return every key the gate keeps, and its state."""
        reply = self._ask(GateCall(op="states"))
        return self._files.states() if reply is None else dict(reply.states)

    def _ask(self, call: GateCall) -> GateAnswer | None:
        reply = _asked(self._address, call)
        return None if reply is None else _expected(reply, GateAnswer)


class DaemonLocks:
    """``Locks`` the daemon at ``address`` holds for every sagent on the host.

    Args:
      address: The daemon's socket.

    """

    def __init__(self, address: Path) -> None:
        self._address = address

    def held(self, path: Path) -> AbstractAsyncContextManager[None]:
        """Return the daemon's hold on ``path``, kept until the block ends."""
        return self._held(path)

    @asynccontextmanager
    async def _held(self, path: Path) -> AsyncGenerator[None]:
        async with contextlib.AsyncExitStack() as stack:
            try:
                reader, writer = await asyncio.open_unix_connection(
                    self._address,
                    limit=MAX_LINE_BYTES,
                )
            except OSError:
                reply = None
            else:
                stack.push_async_callback(writer.wait_closed)
                stack.callback(writer.close)
                writer.write(encoded(Hold(name=str(path))))
                await writer.drain()
                reply = await received(reader)
            if reply is None:
                # The daemon holds the same file lock; with none, this process takes it.
                await stack.enter_async_context(FileLocks().held(path))
            else:
                _ = _expected(reply, Held)
            yield


class DaemonLedger:
    """The logins ``Ledger`` the daemon at ``address`` keeps.

    Args:
      address: The daemon's socket.

    """

    def __init__(self, address: Path) -> None:
        self._address = address

    def read(self) -> str:
        """Return the ledger's text; ``""`` when there is none."""
        return self._ask(LedgerCall())

    def write(self, text: str) -> None:
        """Replace the ledger's text with ``text``."""
        self._ask(LedgerCall(text=text))

    def _ask(self, call: LedgerCall) -> str:
        reply = _asked(self._address, call)
        if reply is None:
            return _ledger_answer(call, FileLedger()).text
        return _expected(reply, LedgerText).text


def _gate_answer(call: GateCall, gate: Gate) -> GateAnswer:
    """Return what ``gate`` answers a call that does not wait."""
    match call.op:
        case "admit":
            return GateAnswer(admission=gate.admit(call.key, _policy(call)))
        case "trip":
            gate.trip(call.key, _policy(call), challenge_url=call.challenge_url)
        case "hold":
            gate.hold(call.key, until=call.until)
        case "release_probe":
            gate.release_probe(call.key)
        case "reset":
            gate.reset(call.key, forget=call.forget)
        case "states":
            return GateAnswer(states=gate.states())
        case "admitted":
            raise ValueError("A gate's admitted call waits; Shared answers it.")
    return GateAnswer()


def _ledger_answer(call: LedgerCall, ledger: Ledger) -> LedgerText:
    """Write ``call``'s text to ``ledger`` if it has any; return what it holds."""
    if call.text is not None:
        ledger.write(call.text)
    return LedgerText(text=ledger.read())


def _policy(call: GateCall) -> SitePolicy:
    """Return the policy ``call`` names; the daemon's socket is a trust boundary."""
    if call.policy is None:
        raise ValueError(f"A gate's {call.op} needs a policy.")
    return call.policy


def _expected[M](reply: Message, kind: type[M]) -> M:
    """Return ``reply`` as ``kind``; another kind is a daemon this client cannot trust."""
    if not isinstance(reply, kind):
        raise TypeError(
            f"The daemon answered {type(reply).__name__}, not {kind.__name__}.",
        )
    return reply


def _asked(address: Path, request: GateCall | LedgerCall) -> Message | None:
    """Send ``request`` alone on a connection to ``address``; return the answer."""
    line = b""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(30.0)
        try:
            connection.connect(str(address))
        except OSError:
            return None
        connection.sendall(encoded(request))
        while not line.endswith(b"\n"):
            chunk = connection.recv(65_536)
            if not chunk:
                return None
            line += chunk
    return decoded(line)
