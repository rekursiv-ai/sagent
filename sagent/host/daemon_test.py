"""Lifecycle tests for the shared daemon and its local control channel."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import asyncio
import contextlib
import fcntl
import os
import random
import shlex
import sys
import tempfile

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.host import daemon, strings
from sagent.host.daemon import control, ensure
from sagent.host.strings import ServeDirs
from sagent.host.wire import (
    VERSION,
    Ack,
    Bye,
    DaemonControl,
    DaemonStatus,
    Describe,
    Described,
    Hello,
    Note,
    Peers,
    Ping,
    Refused,
    Welcome,
    encoded,
    received,
)
from sagent.lib.files.owner_only import open_private
from sagent.lib.posix.child import Child
from sagent.lib.posix.pids import unlocked_within
from sagent.lib.posix.sockets import socket_address
from sagent.lib.userdirs import config_dir, state_dir
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.session.types import AgentToAgentMessage
from sagent.tools.remote import RemoteSession, SavedRemote


if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import BinaryIO

    from sagent.host.wire import Message


def test_fingerprint_tracks_daemon_and_shared_sources_not_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [
        tmp_path / name
        for name in (
            "sagent/host/daemon.py",
            "lib/helper.py",
            "treekle/codec.py",
        )
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n")
    monkeypatch.setattr(daemon, "_CWD", paths[0].parent)
    before = daemon.fingerprint()
    (paths[1].parent / "helper_test.py").write_text("test = 1\n")
    assert daemon.fingerprint() == before
    for path in paths:
        path.write_text("value = 222\n")
        after = daemon.fingerprint()
        assert after != before
        before = after


@pytest.mark.asyncio
async def test_simultaneous_clients_share_one_daemon_and_upgrade_it(
    tmp_path: Path,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    try:
        a, b = await asyncio.gather(
            ensure(dirs, version="old"),
            ensure(dirs, version="old"),
        )
        assert a.pid == b.pid
        assert a.source_version == "old"
        assert a.listen == ()
        status = await control(dirs, DaemonControl())
        assert status.pid == a.pid
        updated = await ensure(dirs, version="new")
        assert updated.pid == a.pid
        assert updated.source_version == "new"
        assert not updated.restarting
    finally:
        with contextlib.suppress(OSError):
            _ = await control(dirs, DaemonControl(stop=True))


@pytest.mark.asyncio
async def test_an_upgrade_gap_does_not_spawn_a_competing_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    answers: list[DaemonStatus | OSError] = [
        DaemonStatus(pid=123, source_version="old", listen=(), restarting=True),
        ConnectionRefusedError(),
        DaemonStatus(pid=123, source_version="new", listen=()),
    ]
    spawned: list[str] = []

    def spawn(dirs: ServeDirs, *, version: str) -> None:
        del dirs
        spawned.append(version)

    monkeypatch.setattr(daemon, "control", partial(_control_answers, answers=answers))
    monkeypatch.setattr(daemon, "_spawn", spawn)
    assert (await ensure(dirs, version="new")).pid == 123
    assert spawned == []


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 7, -9])
async def test_startup_reports_child_exit_without_waiting_for_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    answers: list[DaemonStatus | OSError] = [
        FileNotFoundError(),
        ConnectionRefusedError(),
    ]
    spawned: list[str] = []
    sleeps: list[float] = []

    def spawn(dirs: ServeDirs, *, version: str) -> SimpleNamespace:
        del dirs
        spawned.append(version)
        return SimpleNamespace(poll=lambda: exit_code)

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(daemon, "control", partial(_control_answers, answers=answers))
    monkeypatch.setattr(daemon, "_spawn", spawn)
    monkeypatch.setattr(
        daemon,
        "time",
        SimpleNamespace(monotonic=iter([0, 1, 2, 100]).__next__),
    )
    monkeypatch.setattr(
        daemon,
        "asyncio",
        SimpleNamespace(to_thread=asyncio.to_thread, sleep=sleep),
    )
    with pytest.raises(RuntimeError, match=f"exited with status {exit_code}") as exc:
        _ = await ensure(dirs, version="test")
    assert str(dirs.state / "daemon.log") in str(exc.value)
    assert spawned == ["test"]
    assert sleeps == [0.02]
    with (dirs.state / "daemon-start.lock").open() as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.asyncio
async def test_ready_competing_daemon_wins_over_exited_startup_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    ready = DaemonStatus(pid=123, source_version="new", listen=())
    answers: list[DaemonStatus | OSError] = [FileNotFoundError(), ready]

    def spawn(dirs: ServeDirs, *, version: str) -> SimpleNamespace:
        del dirs, version
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(daemon, "control", partial(_control_answers, answers=answers))
    monkeypatch.setattr(daemon, "_spawn", spawn)
    assert await ensure(dirs, version="new") == ready


async def _control_answers(
    dirs: ServeDirs,
    request: DaemonControl,
    *,
    answers: list[DaemonStatus | OSError],
) -> DaemonStatus:
    del dirs, request
    answer = answers.pop(0)
    if isinstance(answer, OSError):
        raise answer
    return answer


def test_daemon_entry_parses_owner_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    served: list[tuple[ServeDirs, str]] = []

    async def serve(dirs: ServeDirs, *, version: str, handoff: str = "") -> None:
        assert handoff == ""
        served.append((dirs, version))

    monkeypatch.setattr(daemon, "_serve", serve)
    assert (
        daemon.main(
            [
                "--state",
                str(tmp_path / "state"),
                "--secrets",
                str(tmp_path / "secrets"),
                "--source-version",
                "v1",
            ],
        )
        == 0
    )
    assert served == [
        (ServeDirs(state=tmp_path / "state", secrets=tmp_path / "secrets"), "v1"),
    ]


@pytest.mark.parametrize("missing", ["--state", "--secrets", "--source-version"])
def test_daemon_entry_requires_each_owner_argument(missing: str) -> None:
    argv = ["--state", "/state", "--secrets", "/secrets", "--source-version", "v1"]
    index = argv.index(missing)
    del argv[index : index + 2]
    with pytest.raises(SystemExit) as exc:
        _ = daemon.main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("documented", [True, False])
def test_daemon_entry_help_has_only_its_docstring(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    documented: bool,
) -> None:
    if not documented:
        monkeypatch.setattr(daemon, "__doc__", None)
    with pytest.raises(SystemExit) as exc:
        _ = daemon.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert ("The per-user daemon lifecycle" in out) is documented
    assert "XXXX" not in out


@pytest.mark.asyncio
async def test_upgrade_preserves_accepted_remote_work_without_replaying_it(
    tmp_path: Path,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    try:
        _ = await ensure(dirs, version="old")
        _ = await control(dirs, DaemonControl(listen=("127.0.0.1:0",), tools=("Bash",)))
        (connection,) = await strings.add(dirs)
        a, b = _local_team("a"), _local_team("b")
        async with (
            RemoteSession([SavedRemote(url=connection.url)], retry_sec=0.01) as remote,
            RemoteSession(
                [],
                team=a,
                local_socket=socket_address(dirs.state / "daemon.sock"),
                retry_sec=0.01,
            ) as local_a,
            RemoteSession(
                [],
                team=b,
                local_socket=socket_address(dirs.state / "daemon.sock"),
                retry_sec=0.01,
            ) as local_b,
        ):
            endpoint = next(iter(remote.endpoints().values()))
            async with asyncio.timeout(10):
                await endpoint.until(lambda e: e.up)
            started, release, counter = (
                tmp_path / name for name in ("started", "release", "counter")
            )
            command = f"echo once >> {shlex.quote(str(counter))}; touch {shlex.quote(str(started))}; until test -e {shlex.quote(str(release))}; do sleep 0.01; done; pwd"
            running = asyncio.create_task(
                endpoint.call(
                    agent="Agent",
                    window="w",
                    tool="Bash",
                    arguments={
                        "command": command,
                        "description": "Hold accepted work across upgrade",
                    },
                ),
            )
            async with asyncio.timeout(10):
                while not started.exists():  # noqa: ASYNC110 -- The subprocess signals through a file.
                    await asyncio.sleep(0.01)
            _ = await ensure(dirs, version="new")
            release.touch()
            try:
                result = await asyncio.wait_for(running, timeout=10)
            except TimeoutError:
                pytest.fail(
                    f"{endpoint.error}\n{(dirs.state / 'daemon.log').read_text()}",
                )
            assert result.is_error is not True
            assert counter.read_text().splitlines() == ["once"]
            result = await endpoint.call(
                agent="Agent",
                window="w",
                tool="Bash",
                arguments={"command": "pwd", "description": "Verify restored worker"},
            )
            assert result.is_error is not True
            peer = f"b#{local_b.session_id}"
            local = next(iter(local_a.endpoints().values()))
            async with asyncio.timeout(10):
                await local.until(lambda _: peer in a.outside)
                a.send("a", peer, "after upgrade")
                messages = await b["b"].runtime.inbox.drain()
            assert [m for m in messages if isinstance(m, AgentToAgentMessage)] == [
                AgentToAgentMessage(
                    sender=f"a#{local_a.session_id}",
                    recipient="b",
                    content="after upgrade",
                ),
            ]
            b.set_peer_messaging("b", False)
            async with asyncio.timeout(10):
                await local.until(lambda _: peer not in a.outside)
            b.set_peer_messaging("b", True)
            async with asyncio.timeout(10):
                await local.until(lambda _: peer in a.outside)
    finally:
        with contextlib.suppress(OSError):
            _ = await control(dirs, DaemonControl(stop=True))


def _local_team(label: str) -> Team:
    team = Team(catalog=ModelCatalog(factories={}))
    team.join(Agent(model=ScriptedProvider()["scripted"]), label=label)
    return team


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["duplicate", "snapshot", "exec"])
async def test_failed_replacement_keeps_accepted_work_and_the_old_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    serving = asyncio.create_task(daemon._serve(dirs, version="old"))
    attempted = asyncio.Event()

    def failing(*args: object, **kwargs: object) -> None:
        del args, kwargs
        attempted.set()
        raise OSError("replacement failed")

    try:
        _ = await _ready(dirs)
        _ = await control(dirs, DaemonControl(listen=("127.0.0.1:0",), tools=("Bash",)))
        (connection,) = await strings.add(dirs)
        async with RemoteSession(
            [SavedRemote(url=connection.url)],
            retry_sec=0.01,
        ) as remote:
            endpoint = next(iter(remote.endpoints().values()))
            async with asyncio.timeout(5):
                await endpoint.until(lambda e: e.up)
            started, release, counter = (
                tmp_path / name for name in ("started", "release", "counter")
            )
            running = asyncio.create_task(
                endpoint.call(
                    agent="Agent",
                    window="w",
                    tool="Bash",
                    arguments={
                        "command": f"echo once >> {shlex.quote(str(counter))}; touch {shlex.quote(str(started))}; until test -e {shlex.quote(str(release))}; do sleep 0.01; done",
                        "description": "Hold work across a failed upgrade",
                    },
                ),
            )
            async with asyncio.timeout(5):
                while not started.exists():  # noqa: ASYNC110 -- The subprocess signals through a file.
                    await asyncio.sleep(0.01)
            duplicated: list[int] = []
            closed: list[int] = []
            dup, close = os.dup, os.close

            def duplicate(fd: int) -> int:
                copied = dup(fd)
                duplicated.append(copied)
                return copied

            def closing(fd: int) -> None:
                if fd in duplicated:
                    closed.append(fd)
                close(fd)

            monkeypatch.setattr(os, "dup", duplicate)
            monkeypatch.setattr(os, "close", closing)
            owner, name = {
                "duplicate": (os, "dup"),
                "snapshot": (tempfile, "TemporaryFile"),
                "exec": (os, "execv"),
            }[failure]
            monkeypatch.setattr(owner, name, failing)
            _ = await control(dirs, DaemonControl(source_version="new"))
            await asyncio.wait_for(attempted.wait(), 5)
            release.touch()
            assert (await asyncio.wait_for(running, 5)).is_error is not True
            assert counter.read_text().splitlines() == ["once"]
            assert (await _ready(dirs)).source_version == "old"
            assert not serving.done()
            if failure != "duplicate":
                (descriptor,) = duplicated
                assert descriptor in closed
            assert any(
                r.name == daemon.__name__
                and r.getMessage() == "Daemon replacement failed"
                and r.exc_info is not None
                for r in caplog.records
            )
    finally:
        with contextlib.suppress(OSError):
            _ = await control(dirs, DaemonControl(stop=True))
        await asyncio.gather(serving, return_exceptions=True)


def test_daemon_entry_reads_the_inherited_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "snapshot"
    path.write_text("the inherited state")
    fd = os.open(path, os.O_RDONLY)
    served: list[str] = []

    async def serve(dirs: ServeDirs, *, version: str, handoff: str) -> None:
        assert dirs == ServeDirs(state=tmp_path / "state", secrets=tmp_path / "secrets")
        assert version == "new"
        served.append(handoff)

    monkeypatch.setattr(daemon, "_serve", serve)
    assert (
        daemon.main(
            [
                "--state",
                str(tmp_path / "state"),
                "--secrets",
                str(tmp_path / "secrets"),
                "--source-version",
                "new",
                "--handoff-fd",
                str(fd),
            ],
        )
        == 0
    )
    assert served == ["the inherited state"]
    with pytest.raises(OSError, match="Bad file descriptor"):
        _ = os.fstat(fd)


@pytest.mark.asyncio
async def test_cancelled_startup_closes_its_lock_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    dirs.state.mkdir()
    with (dirs.state / "daemon-start.lock").open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        opened = asyncio.Event()
        fds: list[int] = []

        def opening(path: Path, flags: int) -> int:
            fd = open_private(path, flags)
            fds.append(fd)
            opened.set()
            return fd

        monkeypatch.setattr(daemon, "open_private", opening)
        starting = asyncio.create_task(ensure(dirs, version="test"))
        await asyncio.wait_for(opened.wait(), 5)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting
        try:
            with pytest.raises(OSError, match="Bad file descriptor"):
                _ = os.fstat(fds[0])
        finally:
            fcntl.flock(held, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(fds[0])


@pytest.mark.asyncio
async def test_daemon_serves_controls_and_hands_state_to_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    captured: list[tuple[str, list[str], str]] = []

    class ReplacedError(BaseException):
        pass

    make_snapshot = tempfile.TemporaryFile

    def snapshot(*, dir: Path) -> BinaryIO:
        assert dir == dirs.state
        return make_snapshot(dir=dir)

    def replacing(executable: str, argv: list[str]) -> None:
        fd = int(argv[-1])
        assert os.get_inheritable(fd)
        captured.append((executable, argv, os.read(fd, 1_000_000).decode()))
        raise ReplacedError

    monkeypatch.setattr(os, "execv", replacing)
    monkeypatch.setattr(tempfile, "TemporaryFile", snapshot)
    serving = asyncio.create_task(daemon._serve(dirs, version="old"))
    try:
        async with asyncio.timeout(5):
            while not (dirs.state / "daemon.sock").exists():  # noqa: ASYNC110 -- Wait for the real socket bind.
                await asyncio.sleep(0.01)
        assert (await control(dirs, DaemonControl())).source_version == "old"
        with (
            (dirs.state / "daemon.lock").open("r") as held,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        await asyncio.wait_for(daemon._serve(dirs, version="other"), 1)
        reader, writer = await asyncio.open_unix_connection(
            socket_address(dirs.state / "daemon.sock"),
        )
        writer.write(
            encoded(
                Hello(
                    version=VERSION,
                    token="",
                    session_id="kept",
                    client="here",
                ),
            ),
        )
        assert isinstance(await received(reader), Welcome)
        writer.write(encoded(Peers(labels=("registered",))))
        assert await received(reader) == Peers(labels=())
        writer.close()
        assert (await control(dirs, DaemonControl(source_version="new"))).restarting
        with pytest.raises(ReplacedError):
            await serving
        ((executable, argv, state),) = captured
        assert executable == sys.executable
        assert argv[:5] == [executable, "-P", "-m", "sagent", "daemon"]
        assert argv[-2] == "--handoff-fd"
        assert argv[5:-2] == [
            "--state",
            str(dirs.state),
            "--secrets",
            str(dirs.secrets),
            "--source-version",
            "new",
        ]
        restored = asyncio.create_task(
            daemon._serve(dirs, version="new", handoff=state),
        )
        async with asyncio.timeout(5):
            while True:
                with contextlib.suppress(FileNotFoundError, ConnectionRefusedError):
                    if (await control(dirs, DaemonControl())).source_version == "new":
                        break
                await asyncio.sleep(0.01)
        reader, writer = await asyncio.open_unix_connection(
            socket_address(dirs.state / "daemon.sock"),
        )
        writer.write(
            encoded(
                Hello(
                    version=VERSION,
                    token="",
                    session_id="kept",
                    client="here",
                ),
            ),
        )
        welcome = await received(reader)
        assert isinstance(welcome, Welcome)
        assert welcome.resumed
        writer.close()
        _ = await control(dirs, DaemonControl(stop=True))
        await asyncio.wait_for(restored, 5)
    finally:
        if not serving.done():
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)


@pytest.mark.asyncio
async def test_control_rejects_a_connection_without_status(tmp_path: Path) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    dirs.state.mkdir()

    async def refusing(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        _ = await reader.readline()
        writer.close()

    async with await asyncio.start_unix_server(
        refusing,
        path=socket_address(dirs.state / "daemon.sock"),
    ):
        with pytest.raises(ConnectionError, match="returned no status"):
            _ = await control(dirs, DaemonControl())


@pytest.mark.asyncio
async def test_startup_deadline_names_the_daemon_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    monkeypatch.setattr(
        daemon,
        "time",
        SimpleNamespace(monotonic=iter([0, 100]).__next__),
    )
    with pytest.raises(TimeoutError, match=str(dirs.state / "daemon.log")):
        _ = await ensure(dirs, version="test")


async def _ready(dirs: ServeDirs) -> DaemonStatus:
    async with asyncio.timeout(5):
        while True:
            with contextlib.suppress(FileNotFoundError, ConnectionRefusedError):
                return await control(dirs, DaemonControl())
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_restarted_daemon_retains_the_owners_listener_choice(
    tmp_path: Path,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    serving = asyncio.create_task(daemon._serve(dirs, version="old"))
    try:
        _ = await _ready(dirs)
        configured = await control(
            dirs,
            DaemonControl(listen=("127.0.0.1:0",), tools=("Read",)),
        )
        _ = await control(dirs, DaemonControl(stop=True))
        await asyncio.wait_for(serving, 5)
        serving = asyncio.create_task(daemon._serve(dirs, version="new"))
        status = await _ready(dirs)
        assert status.listen == configured.listen
        assert status.source_version == "new"
        assert strings.configuration(dirs).tools == ("Read",)
    finally:
        with contextlib.suppress(OSError):
            _ = await control(dirs, DaemonControl(stop=True))
        await asyncio.wait_for(serving, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [17, 83, 409])
async def test_message_lifecycle_matches_the_model_across_reconnects_and_upgrades(
    tmp_path: Path,
    seed: int,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    clients: list[_Client] = []
    rng = random.Random(seed)  # noqa: S311 -- Repeatable lifecycle ordering, not secrets.
    transitions = ["retry", "reconnect", "upgrade", "withdraw", "spoof", "address"] * 20
    rng.shuffle(transitions)
    try:
        original = await ensure(dirs, version="initial")
        sender = await _Client.connect(dirs, "sender", clients=clients)
        receiver = await _Client.connect(dirs, "receiver", clients=clients)
        sender.writer.write(encoded(Peers(labels=("a",))))
        receiver.writer.write(encoded(Peers(labels=("b",))))
        await sender.directory(("b#receiver",))
        for index, transition in enumerate(transitions):
            note = Note(
                id=f"{seed}-{index}",
                sender="a",
                recipient="b#receiver",
                content=f"step {index}: {transition}",
                kind="ask",
            )
            if transition == "withdraw":
                receiver.writer.write(encoded(Peers(labels=())))
                await sender.directory(())
                sender.writer.write(encoded(note))
                await sender.expect(
                    Refused(id=note.id, reason="Agent 'b#receiver' is not reachable."),
                )
                receiver.writer.write(encoded(Peers(labels=("b",))))
                await sender.directory(("b#receiver",))
            if transition == "spoof":
                sender.writer.write(encoded(replace(note, sender="unregistered")))
                await sender.expect(
                    Refused(
                        id=note.id,
                        reason="The sender is not registered by this session.",
                    ),
                )
            if transition == "address":
                address = rng.choice(
                    ("", "b", "b#", "b#../receiver", "b@host#receiver"),
                )
                sender.writer.write(encoded(replace(note, recipient=address)))
                await sender.expect(
                    Refused(id=note.id, reason=f"Agent {address!r} is not reachable."),
                )
            sender.writer.write(encoded(note) * 2)
            await sender.expect(Ack(id=note.id))
            await sender.expect(Ack(id=note.id))
            delivered = replace(note, sender="a#sender", recipient="b")
            await receiver.expect(delivered)
            if transition == "upgrade":
                sender.writer.close()
                receiver.writer.close()
                updated = await ensure(dirs, version=f"{seed}-{index}")
                assert updated.pid == original.pid
                assert updated.source_version == f"{seed}-{index}"
                sender = await _Client.connect(dirs, "sender", clients=clients)
                assert sender.welcome.resumed
            if transition in ("upgrade", "reconnect"):
                receiver = await _Client.connect(dirs, "receiver", clients=clients)
                assert receiver.welcome.resumed
                assert receiver.welcome.calls == (note.id,)
                await receiver.expect(delivered)
            receiver.writer.write(encoded(Ack(id=note.id)) * 2)
            await receiver.barrier()
            sender.writer.write(encoded(note))
            await sender.expect(Ack(id=note.id))
            await receiver.barrier()
            receiver = await _Client.connect(dirs, "receiver", clients=clients)
            assert receiver.welcome.resumed
            assert receiver.welcome.calls == ()
            await receiver.barrier()
        sender.writer.write(encoded(Bye()))
        receiver.writer.write(encoded(Bye()))
        await sender.expect(None)
        await receiver.expect(None)
        fresh = await _Client.connect(dirs, "receiver", clients=clients)
        assert not fresh.welcome.resumed
        assert fresh.welcome.calls == ()
    finally:
        for client in clients:
            client.writer.close()
        with contextlib.suppress(OSError):
            _ = await control(dirs, DaemonControl(stop=True))
        assert await asyncio.to_thread(
            unlocked_within,
            dirs.state / "daemon.lock",
            seconds=5,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Client:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    welcome: Welcome

    @classmethod
    async def connect(
        cls,
        dirs: ServeDirs,
        session_id: str,
        *,
        clients: list[_Client],
    ) -> _Client:
        reader, writer = await asyncio.open_unix_connection(
            socket_address(dirs.state / "daemon.sock"),
        )
        writer.write(
            encoded(
                Hello(version=VERSION, token="", session_id=session_id, client="here"),
            ),
        )
        welcome = await asyncio.wait_for(received(reader), 5)
        assert isinstance(welcome, Welcome)
        client = cls(reader=reader, writer=writer, welcome=welcome)
        clients.append(client)
        return client

    async def expect(self, expected: Message | None) -> None:
        async with asyncio.timeout(5):
            while isinstance(message := await received(self.reader), (Peers, Ping)):
                pass
            assert message == expected

    async def directory(self, labels: tuple[str, ...]) -> None:
        async with asyncio.timeout(5):
            while True:
                message = await received(self.reader)
                assert isinstance(message, (Peers, Ping))
                if isinstance(message, Peers) and sorted(message.labels) == sorted(
                    labels,
                ):
                    return

    async def barrier(self) -> None:
        self.writer.write(encoded(Describe(id="barrier", tool="absent")))
        await self.expect(Described(id="barrier", spec=None))


@pytest.mark.asyncio
async def test_independent_host_cli_processes_share_one_persistent_daemon() -> None:
    dirs = ServeDirs(
        secrets=config_dir() / "rekursiv-ai" / "secrets" / "sagent" / "serve",
        state=state_dir() / "rekursiv-ai" / "sagent" / "serve",
    )
    try:
        started = await asyncio.gather(
            *[
                _cli("host", "--listen", "127.0.0.1:0", "--tools", "Read")
                for _ in range(4)
            ],
        )
        status = await control(dirs, DaemonControl())
        assert all(f"as pid {status.pid};" in output for output in started)
        assert len(status.listen) == 1
        assert (await ensure(dirs)).pid == status.pid
        url = (await _cli("host", "add", "--tools", "Read")).strip()
        local_team = _local_team("local")
        async with (
            RemoteSession(
                [SavedRemote(url=url)],
                team=_local_team("tls"),
                client="cli",
                retry_sec=0.01,
            ) as remote,
            RemoteSession(
                [],
                team=local_team,
                local_socket=socket_address(dirs.state / "daemon.sock"),
                retry_sec=0.01,
            ) as local,
        ):
            endpoint = next(iter(remote.endpoints().values()))
            local_endpoint = next(iter(local.endpoints().values()))
            await asyncio.wait_for(endpoint.until(lambda e: e.up), 5)
            await asyncio.wait_for(local_endpoint.until(lambda e: e.tried), 5)
            assert local_endpoint.up, local_endpoint.error
            assert tuple(endpoint.tools) == ("Read",)
            assert local_team.outside == []
            peer = f"tls@cli#{remote.session_id}"
            (grant,) = strings.listing(dirs).grants
            _ = await _cli("host", "edit", grant.id, "--peer-messaging", "on")
            await asyncio.wait_for(
                local_endpoint.until(lambda _: peer in local_team.outside),
                5,
            )
            _ = await _cli("host", "edit", grant.id, "--peer-messaging", "off")
            await asyncio.wait_for(
                local_endpoint.until(lambda _: peer not in local_team.outside),
                5,
            )
            _ = await _cli("host", "del", grant.id)
            await asyncio.wait_for(endpoint.until(lambda e: not e.up), 5)
        _ = await _cli("host", "--disable")
        disabled = await control(dirs, DaemonControl())
        assert disabled.pid == status.pid
        assert disabled.listen == ()
        _ = await control(dirs, DaemonControl(stop=True))
        assert await asyncio.to_thread(
            unlocked_within,
            dirs.state / "daemon.lock",
            seconds=5,
        )
        _ = await _cli("host", "--disable")
        assert (await control(dirs, DaemonControl())).listen == ()
    finally:
        with contextlib.suppress(OSError):
            _ = await control(dirs, DaemonControl(stop=True))
        assert await asyncio.to_thread(
            unlocked_within,
            dirs.state / "daemon.lock",
            seconds=5,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnected", ["sender", "receiver"])
@pytest.mark.parametrize("transport", ["local", "tls"])
async def test_reply_debts_survive_a_resumable_disconnect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disconnected: str,
    transport: str,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    a, b = _local_team("a"), _local_team("b")
    try:
        _ = await ensure(dirs, version="debts")
        remotes: list[SavedRemote] = []
        address: Path | None = socket_address(dirs.state / "daemon.sock")
        if transport == "tls":
            _ = await control(dirs, DaemonControl(listen=("127.0.0.1:0",)))
            (connection,) = await strings.add(dirs, peer_messaging=True)
            remotes = [SavedRemote(url=connection.url)]
            address = None
        async with (
            RemoteSession(
                remotes,
                team=a,
                local_socket=address,
                client="here",
                retry_sec=100,
            ) as sender,
            RemoteSession(
                remotes,
                team=b,
                local_socket=address,
                client="here",
                retry_sec=100,
            ) as receiver,
        ):
            first = next(iter(sender.endpoints().values()))
            second = next(iter(receiver.endpoints().values()))
            await asyncio.wait_for(first.until(lambda e: e.up), 5)
            await asyncio.wait_for(second.until(lambda e: e.up), 5)
            source, target = (
                (f"a#{sender.session_id}", f"b#{receiver.session_id}")
                if transport == "local"
                else (
                    f"a@here#{sender.session_id}@{second.name}",
                    f"b@here#{receiver.session_id}@{first.name}",
                )
            )
            await asyncio.wait_for(first.until(lambda _: target in a.outside), 5)
            await asyncio.wait_for(second.until(lambda _: source in b.outside), 5)
            a.send("a", target, "request")
            async with asyncio.timeout(5):
                messages = await b["b"].runtime.inbox.drain()
            assert any(isinstance(m, AgentToAgentMessage) for m in messages)
            dropped, live = (
                (sender, second) if disconnected == "sender" else (receiver, first)
            )
            endpoint = next(iter(dropped.endpoints().values()))
            published = asyncio.Event()
            peered = live._peered

            def directory(labels: tuple[str, ...]) -> None:
                peered(labels)
                published.set()

            monkeypatch.setattr(live, "_peered", directory)
            assert endpoint._writer is not None
            endpoint._writer.close()
            await asyncio.wait_for(endpoint.until(lambda e: not e.up), 5)
            await asyncio.wait_for(published.wait(), 5)
            await dropped.aclose()
            assert a.owes(target) == {"a": "request"}
            assert b.owes("b") == {source: "request"}
            b.send("b", source, "reply")
            assert b.owes("b") == {}
            await dropped.aopen()
            await asyncio.wait_for(endpoint.until(lambda e: e.up), 5)
            async with asyncio.timeout(5):
                messages = await a["a"].runtime.inbox.drain()
            assert any(
                isinstance(m, AgentToAgentMessage) and m.content == "reply"
                for m in messages
            )
            assert a.owes(target) == {}
            assert b.owes("b") == {}
    finally:
        with contextlib.suppress(OSError):
            _ = await control(dirs, DaemonControl(stop=True))
        assert await asyncio.to_thread(
            unlocked_within,
            dirs.state / "daemon.lock",
            seconds=5,
        )


@pytest.mark.asyncio
async def test_local_client_connects_under_a_long_state_path(tmp_path: Path) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / ("long" * 40))
    clients: list[_Client] = []
    try:
        _ = await ensure(dirs, version="long-path")
        client = await _Client.connect(dirs, "long-path", clients=clients)
        await client.barrier()
        client.writer.write(encoded(Bye()))
        await client.expect(None)
    finally:
        for client in clients:
            client.writer.close()
        with contextlib.suppress(OSError):
            _ = await control(dirs, DaemonControl(stop=True))
        assert await asyncio.to_thread(
            unlocked_within,
            dirs.state / "daemon.lock",
            seconds=5,
        )


@pytest.mark.asyncio
async def test_malformed_clients_do_not_disrupt_existing_sessions(
    tmp_path: Path,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    clients: list[_Client] = []
    try:
        original = await ensure(dirs, version="malformed")
        witness = await _Client.connect(dirs, "witness", clients=clients)
        hello = Hello(version=VERSION, token="", session_id="probe", client="here")
        frames = (
            (b"not-json\n", None),
            (b'{"truncated":', None),
            (b"[]\n", None),
            (
                encoded(Ping()),
                Refused(reason="Say hello with the current protocol first."),
            ),
            (
                encoded(replace(hello, version=VERSION + 1)),
                Refused(reason="Say hello with the current protocol first."),
            ),
            (
                encoded(replace(hello, session_id="../escape")),
                Refused(reason="Invalid session id."),
            ),
        )
        for frame, expected in frames:
            reader, writer = await asyncio.open_unix_connection(
                socket_address(dirs.state / "daemon.sock"),
            )
            try:
                writer.write(frame)
                writer.write_eof()
                assert await asyncio.wait_for(received(reader), 5) == expected
                if expected is not None:
                    assert await asyncio.wait_for(received(reader), 5) is None
            finally:
                writer.close()
                await writer.wait_closed()
            await witness.barrier()
        assert (await control(dirs, DaemonControl())).pid == original.pid
        witness.writer.write(encoded(Bye()))
        await witness.expect(None)
    finally:
        for client in clients:
            client.writer.close()
        with contextlib.suppress(OSError):
            _ = await control(dirs, DaemonControl(stop=True))
        assert await asyncio.to_thread(
            unlocked_within,
            dirs.state / "daemon.lock",
            seconds=5,
        )


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Measures Linux process resources.")
@pytest.mark.parametrize("delayed_close", [False, True])
async def test_worker_churn_releases_processes_descriptors_and_public_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delayed_close: bool,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    clients: list[_Client] = []
    counts: list[int] = []
    pending: asyncio.StreamWriter | None = None
    try:
        status = await ensure(dirs, version="churn")
        _ = await control(dirs, DaemonControl(listen=("127.0.0.1:0",), tools=("Read",)))
        (connection,) = await strings.add(dirs, peer_messaging=True)
        observer = await _Client.connect(dirs, "observer", clients=clients)
        process = Path(f"/proc/{status.pid}")
        await observer.barrier()
        baseline = len(list((process / "fd").iterdir()))
        iterdir = Path.iterdir

        def descriptors(path: Path) -> Iterator[Path]:
            nonlocal pending
            entries = list(iterdir(path))
            if path == process / "fd" and pending is not None:
                pending.close()
                pending = None
            return iter(entries)

        monkeypatch.setattr(Path, "iterdir", descriptors)
        for iteration in range(26):
            async with contextlib.AsyncExitStack() as stack:
                sessions = [
                    await stack.enter_async_context(
                        RemoteSession(
                            [SavedRemote(url=connection.url)],
                            team=_local_team("worker"),
                            client="churn",
                            retry_sec=0.01,
                        ),
                    )
                    for _ in range(4)
                ]
                endpoints = [next(iter(s.endpoints().values())) for s in sessions]
                await asyncio.wait_for(
                    asyncio.gather(*(e.until(lambda e: e.up) for e in endpoints)),
                    5,
                )
                await observer.directory(
                    tuple(f"worker@churn#{s.session_id}" for s in sessions),
                )
                results = await asyncio.gather(*(e.spec("Read") for e in endpoints))
                assert all(spec is not None and spec.description for spec in results)
                if delayed_close and iteration == 0:
                    _, pending = await asyncio.open_unix_connection(
                        socket_address(dirs.state / "daemon.sock"),
                    )
                    _ = await control(dirs, DaemonControl())
            await observer.directory(())
            async with asyncio.timeout(5):
                while any(  # noqa: ASYNC110 -- The daemon publishes cleanup in this file.
                    row.token_id for row in strings.listing(dirs).sessions or ()
                ):
                    await asyncio.sleep(0.01)
            assert (process / "task" / str(status.pid) / "children").read_text() == ""
            async with asyncio.timeout(5):
                while (count := len(list((process / "fd").iterdir()))) > baseline:  # noqa: ASYNC110 -- TLS transport closure follows session removal.
                    await asyncio.sleep(0.01)
            counts.append(count)
            assert count == baseline, counts
        observer.writer.write(encoded(Bye()))
        await observer.expect(None)
    finally:
        if pending is not None:
            pending.close()
        for client in clients:
            client.writer.close()
        with contextlib.suppress(OSError):
            _ = await control(dirs, DaemonControl(stop=True))
        assert await asyncio.to_thread(
            unlocked_within,
            dirs.state / "daemon.lock",
            seconds=5,
        )


async def _cli(*arguments: str) -> str:
    async with await Child.start(
        sys.executable,
        "-P",
        "-m",
        "sagent",
        *arguments,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    ) as child:
        stdout, stderr = await asyncio.wait_for(child.process.communicate(), 15)
        assert child.process.returncode == 0, stderr.decode()
        return stdout.decode()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
