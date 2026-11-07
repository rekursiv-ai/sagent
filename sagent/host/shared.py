"""The rate gate the per-user daemon serves every sagent on the host.

``DaemonGate`` is the daemon's backend of wesearch's ``Gate``: a client sends
one request on its own connection to the daemon's owner-only socket, and the
daemon answers from the host's files, so a process with no daemon shares one
state with those that have one.

What the daemon adds is the wait. An ``admitted`` call stays open in the daemon
(``ServedGate``) until the caller is admitted, and each change to a key --
another caller's 429 holding it, a probe's success lifting it -- wakes every
caller waiting on that key to ask again. Waiting on the files alone, a caller
sees such a change only when its own wait ends.

A daemon that cannot be reached, or fails mid-request, is answered for by the
host's files directly.

Typical usage example:

  catalog.gate = fetch.gate = host_gate()
  admission = await catalog.gate.admitted(key, policy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import asyncio
import contextlib
import socket

from wesearch.fetch.gate import FileGate, Gate, waited

from sagent.host.wire import GateAnswer, GateCall, decoded, encoded, received
from sagent.lib.posix.sockets import socket_address
from sagent.lib.userdirs import state_dir


if TYPE_CHECKING:
    from pathlib import Path

    from wesearch.fetch.gate import Admission, GatePolicy, GateState

    from sagent.host.wire import Message


__all__ = [
    "DaemonGate",
    "ServedGate",
    "host_gate",
]


def host_gate() -> DaemonGate:
    """Return the gate this user's daemon on this host serves.

    Every sagent on the host reaches the daemon at its user's state directory:
    one a terminal runs, and one a daemon starts for a client's session.

    Returns:
      gate: The daemon's gate; the host's files while no daemon answers.

    """
    serve = state_dir() / "rekursiv-ai" / "sagent" / "serve"
    return DaemonGate(socket_address(serve / "daemon.sock"))


@dataclass(slots=True, kw_only=True, eq=False)
class ServedGate:
    """The daemon's side of the gate: its state, and every caller waiting on a key.

    Attributes:
      gate: Where the gate's state is kept; the host's files.

    """

    gate: Gate = field(default_factory=FileGate)
    _changed: asyncio.Condition = field(default_factory=asyncio.Condition)
    _versions: dict[str, int] = field(default_factory=dict[str, int])

    async def answer(self, call: GateCall) -> GateAnswer:
        """Return what the daemon answers ``call`` with.

        Args:
          call: An owner's request.

        Returns:
          reply: For ``admitted``, the admission; else nothing.

        Raises:
          ValueError: An ``admitted`` or ``trip`` names no policy.

        """
        if call.op == "admitted":
            key, policy = call.key, _policy(call)
            seen = [0]  # pragma: no mutate -- each ask notes it before a wait reads it.
            admission = await waited(
                partial(self._admit_noting, key, policy, seen),
                policy,
                wait=partial(self._changed_within, key, seen),
            )
            return GateAnswer(admission=admission)
        await asyncio.to_thread(_gate_answer, call, self.gate)
        async with self._changed:
            # pragma: no mutate start -- any new value wakes a waiter, which compares.
            self._versions[call.key] = self._versions.get(call.key, 0) + 1
            # pragma: no mutate end
            self._changed.notify_all()
        return GateAnswer()

    async def _admit_noting(
        self,
        key: str,
        policy: GatePolicy,
        seen: list[int],
    ) -> Admission:
        """Ask the gate, noting in ``seen`` the change it answers as of."""
        # Noted before asking, so a change made while it asks cuts the wait short.
        seen[0] = self._versions.get(key, 0)
        return await asyncio.to_thread(self.gate.admit, key, policy)

    async def _changed_within(self, key: str, seen: list[int], seconds: float) -> bool:
        """Wait ``seconds``; return early, ``True``, once ``key`` changes past ``seen``."""
        async with self._changed:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._changed.wait_for(
                        lambda: self._versions.get(key, 0) != seen[0],
                    ),
                    seconds,
                )
            return self._versions.get(key, 0) != seen[0]


class DaemonGate:
    """A ``Gate`` the daemon at ``address`` keeps for every sagent on the host.

    Args:
      address: The daemon's socket.

    """

    def __init__(self, address: Path) -> None:
        self._address = address
        self._files = FileGate()

    def admit(self, key: str, policy: GatePolicy) -> Admission:
        """Admit the caller, make it the probe, or say how long to wait.

        From the host's files: the daemon would answer from them too, and an
        admission changes nothing a waiter waits on.

        Args:
          key: What is paced.
          policy: How it is paced and recovered.

        Returns:
          admission: The wait, and whether to send after it.

        """
        return self._files.admit(key, policy)

    async def admitted(self, key: str, policy: GatePolicy) -> Admission:
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
        except OSError:
            reply = None
        finally:
            writer.close()
        if reply is None:
            # The daemon went away mid-wait; the files hold what it kept.
            return await self._files.admitted(key, policy)
        return _expected(reply).admission

    def trip(self, key: str, policy: GatePolicy, *, challenge_url: str = "") -> None:
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

    def reset(
        self,
        key: str,
        *,
        forget: bool = False,
        clear_hold: bool = False,
    ) -> None:
        """Lift a key's lock, optionally forgetting its pace or server hold."""
        if (
            self._ask(
                GateCall(op="reset", key=key, forget=forget, clear_hold=clear_hold),
            )
            is None
        ):
            self._files.reset(key, forget=forget, clear_hold=clear_hold)

    def states(self) -> dict[str, GateState]:
        """Return every key the gate keeps, and its state, from the host's files."""
        return self._files.states()

    def _ask(self, call: GateCall) -> GateAnswer | None:
        reply = _asked(self._address, call)
        return None if reply is None else _expected(reply)


def _gate_answer(call: GateCall, gate: Gate) -> None:
    """Make on ``gate`` the change to a key ``call`` asks for."""
    match call.op:
        case "trip":
            gate.trip(call.key, _policy(call), challenge_url=call.challenge_url)
        case "hold":
            gate.hold(call.key, until=call.until)
        case "release_probe":
            gate.release_probe(call.key)
        case "reset":
            gate.reset(call.key, forget=call.forget, clear_hold=call.clear_hold)
        case "admitted":
            raise ValueError("A gate's admitted call waits; ServedGate answers it.")


def _policy(call: GateCall) -> GatePolicy:
    """Return the policy ``call`` names; the daemon's socket is a trust boundary."""
    if call.policy is None:
        raise ValueError(f"A gate's {call.op} needs a policy.")
    return call.policy


def _expected(reply: Message) -> GateAnswer:
    """Return ``reply``; another kind is a daemon this client cannot trust."""
    if not isinstance(reply, GateAnswer):
        raise TypeError(f"The daemon answered {type(reply).__name__}, not GateAnswer.")
    return reply


def _asked(address: Path, request: GateCall) -> Message | None:
    """Send ``request`` alone on a connection to ``address``; return the answer."""
    line = b""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(30.0)
        try:
            connection.connect(str(address))
            connection.sendall(encoded(request))
            while not line.endswith(b"\n"):
                chunk = connection.recv(65_536)
                if not chunk:
                    return None
                line += chunk
        except OSError:
            return None
    return decoded(line)
