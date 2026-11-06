"""Lifecycle tests for the shared daemon and its local control channel."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, override

import asyncio
import contextlib
import fcntl
import os
import random
import secrets
import shlex
import subprocess
import sys
import tempfile
import threading

from treekle import json_loads

import pytest
import pytest_asyncio

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.host import (
    daemon,
    server as host_server,
    session as host_session,
    strings,
)
from sagent.host.daemon import control, ensure
from sagent.host.server import Server, _ChildHost
from sagent.host.settings import HostSettings
from sagent.host.strings import ServeDirs
from sagent.host.testing import Gate, LocalHost
from sagent.host.wire import (
    MAX_LINE_BYTES,
    VERSION,
    Ack,
    Bye,
    DaemonControl,
    DaemonStatus,
    Describe,
    Described,
    Hello,
    Message,
    Note,
    Peers,
    Ping,
    Refused,
    Tools,
    ToolSpec,
    Welcome,
    encoded,
    received,
)
from sagent.lib.files.owner_only import open_private
from sagent.lib.posix.pids import unlocked_within
from sagent.lib.posix.sockets import socket_address
from sagent.lib.userdirs import config_dir, state_dir
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.session.types import AgentToAgentMessage, stored_record
from sagent.tools.read import Read
from sagent.tools.remote import RemoteSession, SavedRemote


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
    from typing import BinaryIO

    from treekle import PlainTree

    from sagent.host.server import _Session
    from sagent.session.types import FunctionResult
    from sagent.tools.types import ToolContext


class _ExecError(BaseException):
    def __init__(self, version: str, snapshot: str) -> None:
        self.version = version
        self.snapshot = snapshot


@dataclass(frozen=True, slots=True, kw_only=True)
class _Process:
    task: asyncio.Task[None]

    def poll(self) -> int | None:
        return 0 if self.task.done() else None


@dataclass(frozen=True, slots=True, kw_only=True)
class _CountedGate(Gate):
    calls: list[str] = field(default_factory=list[str])

    @override
    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> FunctionResult:
        self.calls.append(context.label)
        return await super(_CountedGate, self).call(context, arguments)


@pytest_asyncio.fixture
async def local_daemon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> AsyncIterator[LocalHost]:
    monkeypatch.chdir(tmp_path)
    tasks: list[asyncio.Task[None]] = []
    servers: list[Server] = []
    workers = LocalHost(tools={"Read": Read(), "Gate": _CountedGate()})
    opening = Server.aopen

    async def start(tools: Sequence[str], directory: Path) -> _ChildHost:
        running = await workers(tools, directory)
        return _ChildHost(
            child=None,
            pid=len(workers.started),
            reader=running.reader,
            writer=running.writer,
        )

    async def reap(pid: int) -> None:
        await workers.started[pid - 1].task

    async def opened(server: Server) -> None:
        await opening(server)
        servers.append(server)

    async def tick(seconds: float) -> None:
        del seconds
        await asyncio.sleep(0)

    async def run(dirs: ServeDirs, *, version: str) -> None:
        snapshot = ""
        while True:
            try:
                await daemon._serve(dirs, version=version, handoff=snapshot)
            except _ExecError as replaced:
                version, snapshot = replaced.version, replaced.snapshot
            else:
                return

    def spawn(dirs: ServeDirs, *, version: str) -> _Process:
        task = asyncio.create_task(run(dirs, version=version))
        tasks.append(task)
        return _Process(task=task)

    def replacing(executable: str, argv: list[str]) -> None:
        assert executable == sys.executable
        snapshot = os.read(int(argv[-1]), 1_000_000).decode()
        server = servers[-1]
        for session in server._sessions.values():
            if session.host is not None:
                session.host.writer.close()
        server._sessions.clear()
        raise _ExecError(argv[argv.index("--source-version") + 1], snapshot)

    monkeypatch.setattr(daemon, "_spawn", spawn)
    monkeypatch.setattr(
        daemon,
        "Server",
        partial(Server, start_host=start, poll_sec=0.001),
    )
    monkeypatch.setattr(host_server, "_reap", reap)
    monkeypatch.setattr(Server, "aopen", opened)
    monkeypatch.setattr(os, "execv", replacing)
    monkeypatch.setattr(
        daemon,
        "asyncio",
        SimpleNamespace(
            sleep=tick,
            to_thread=asyncio.to_thread,
            wait_for=asyncio.wait_for,
            open_unix_connection=asyncio.open_unix_connection,
            run=asyncio.run,
        ),
    )
    monkeypatch.setattr(daemon, "fingerprint", lambda: "fixture-version")
    try:
        yield workers
    finally:
        for server in servers:
            server.replacement = ""
            server.stopping.set()
        await asyncio.gather(*tasks)


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
@pytest.mark.usefixtures("local_daemon")
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


@pytest.mark.asyncio
async def test_startup_retries_one_live_child_until_it_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    ready = DaemonStatus(pid=123, source_version="new", listen=())
    answers: list[DaemonStatus | OSError] = [
        FileNotFoundError(),
        ConnectionRefusedError(),
        FileNotFoundError(),
        ready,
    ]
    spawned: list[str] = []
    sleeps: list[float] = []

    def spawn(dirs: ServeDirs, *, version: str) -> SimpleNamespace:
        del dirs
        spawned.append(version)
        return SimpleNamespace(poll=lambda: None)

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(daemon, "control", partial(_control_answers, answers=answers))
    monkeypatch.setattr(daemon, "_spawn", spawn)
    monkeypatch.setattr(
        daemon,
        "asyncio",
        SimpleNamespace(to_thread=asyncio.to_thread, sleep=sleep),
    )
    assert await ensure(dirs, version="new") == ready
    assert spawned == ["new"]
    assert sleeps == [0.02, 0.02, 0.02]
    assert not answers


@pytest.mark.asyncio
async def test_default_startup_version_comes_from_the_source_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    ready = DaemonStatus(pid=123, source_version="from-disk", listen=())

    async def checked(dirs: ServeDirs, request: DaemonControl) -> DaemonStatus:
        del dirs
        assert request == DaemonControl(source_version="from-disk")
        return ready

    monkeypatch.setattr(daemon, "fingerprint", lambda: "from-disk")
    monkeypatch.setattr(daemon, "control", checked)
    assert await ensure(dirs) == ready


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


def test_spawn_returns_the_child_and_reaps_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    process = _SpawnedProcess()
    descriptors: list[int] = []
    threads: list[_WaitThread] = []
    dirs.state.mkdir()
    (dirs.state / "daemon.log").write_bytes(b"previous\n")

    def launch(
        argv: list[str],
        *,
        stdin: int,
        stdout: int,
        stderr: int,
        start_new_session: bool,
    ) -> _SpawnedProcess:
        assert argv == [
            sys.executable,
            "-P",
            "-m",
            "sagent",
            "daemon",
            "--state",
            str(dirs.state),
            "--secrets",
            str(dirs.secrets),
            "--source-version",
            "v1",
        ]
        assert stdin == subprocess.DEVNULL
        assert stdout == stderr
        assert start_new_session
        descriptors.append(stdout)
        assert os.write(stdout, b"started\n") == len(b"started\n")
        return process

    def make_thread(*, target: Callable[[], int], daemon: bool) -> _WaitThread:
        assert daemon
        thread = _WaitThread(target=target)
        threads.append(thread)
        return thread

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(threading, "Thread", make_thread)
    assert daemon._spawn(dirs, version="v1") is process
    (descriptor,) = descriptors
    with pytest.raises(OSError, match="Bad file descriptor"):
        _ = os.fstat(descriptor)
    (thread,) = threads
    assert thread.starts == 1
    assert process.waits == 0
    assert thread.target() == 0
    assert process.waits == 1
    assert (dirs.state / "daemon.log").read_bytes() == b"previous\nstarted\n"
    assert (dirs.state / "daemon.log").stat().st_mode & 0o777 == 0o600


@dataclass(slots=True, kw_only=True)
class _SpawnedProcess:
    waits: int = 0

    def wait(self) -> int:
        self.waits += 1
        return 0


@dataclass(slots=True, kw_only=True)
class _WaitThread:
    target: Callable[[], int]
    starts: int = 0

    def start(self) -> None:
        self.starts += 1


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
    local_daemon: LocalHost,
) -> None:
    gate = local_daemon.tools["Gate"]
    assert isinstance(gate, _CountedGate)
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    try:
        _ = await ensure(dirs, version="old")
        _ = await control(dirs, DaemonControl(listen=("127.0.0.1:0",), tools=("Gate",)))
        (connection,) = await strings.add(dirs)
        a, b = _local_team("a"), _local_team("b")
        async with (
            RemoteSession(
                [SavedRemote(url=connection.url)],
                client="fixture",
                retry_sec=0.01,
            ) as remote,
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
            running = asyncio.create_task(
                endpoint.call(agent="Agent", window="w", tool="Gate", arguments={}),
            )
            await asyncio.wait_for(gate.started.wait(), 5)
            _ = await ensure(dirs, version="new")
            gate.opened.set()
            result = await asyncio.wait_for(running, timeout=10)
            assert result.is_error is not True
            assert gate.calls == ["Agent@fixture"]
            assert not gate.stopped.is_set()
            result = await endpoint.call(
                agent="Agent",
                window="w",
                tool="Gate",
                arguments={},
            )
            assert result.is_error is not True
            assert gate.calls == ["Agent@fixture", "Agent@fixture"]
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
    local_daemon: LocalHost,
) -> None:
    gate = local_daemon.tools["Gate"]
    assert isinstance(gate, _CountedGate)
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    serving = asyncio.create_task(daemon._serve(dirs, version="old"))
    attempted = asyncio.Event()

    def failing(*args: object, **kwargs: object) -> None:
        del args, kwargs
        attempted.set()
        raise OSError("replacement failed")

    try:
        _ = await _ready(dirs)
        _ = await control(dirs, DaemonControl(listen=("127.0.0.1:0",), tools=("Gate",)))
        (connection,) = await strings.add(dirs)
        async with RemoteSession(
            [SavedRemote(url=connection.url)],
            client="fixture",
            retry_sec=0.01,
        ) as remote:
            endpoint = next(iter(remote.endpoints().values()))
            async with asyncio.timeout(5):
                await endpoint.until(lambda e: e.up)
            running = asyncio.create_task(
                endpoint.call(agent="Agent", window="w", tool="Gate", arguments={}),
            )
            await asyncio.wait_for(gate.started.wait(), 5)
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
            gate.opened.set()
            assert (await asyncio.wait_for(running, 5)).is_error is not True
            assert gate.calls == ["Agent@fixture"]
            assert not gate.stopped.is_set()
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
@pytest.mark.parametrize("phase", ["connection", "status"])
async def test_control_bounds_a_stalled_connection_or_status(
    control_peer: _ControlPeer,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    """Reject an unbounded wait without requiring an exact timeout duration."""
    control_peer.delay_status = phase == "status"
    if phase == "connection":
        monkeypatch.setattr(
            daemon,
            "asyncio",
            SimpleNamespace(
                open_unix_connection=control_peer.connect,
                wait_for=asyncio.wait_for,
            ),
        )
    loop = asyncio.get_running_loop()
    now = loop.time()
    with monkeypatch.context() as clock:
        clock.setattr(loop, "time", lambda: now)
        release = loop.call_later(60, control_peer.release.set)
        pending = asyncio.create_task(control(control_peer.dirs, DaemonControl()))
        try:
            await control_peer.pending.wait()
            now += 60
            with pytest.raises(TimeoutError):
                _ = await pending
        finally:
            release.cancel()
            control_peer.release.set()
            _ = pending.cancel()
            _ = await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_control_accepts_a_status_above_the_default_stream_limit(
    control_peer: _ControlPeer,
) -> None:
    """Read a supported status line larger than asyncio's default 64 KiB."""
    control_peer.status = replace(
        control_peer.status,
        listen=tuple(f"127.0.0.1:{port}" for port in range(10_000, 14_000)),
    )
    assert 64 << 10 < len(encoded(control_peer.status)) < MAX_LINE_BYTES
    assert await control(control_peer.dirs, DaemonControl()) == control_peer.status
    assert control_peer.requests == [DaemonControl()]


@pytest.mark.asyncio
async def test_startup_deadline_names_the_daemon_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    monkeypatch.setattr(
        daemon,
        "time",
        SimpleNamespace(monotonic=iter([100, 130]).__next__),
    )

    async def expired(dirs: ServeDirs, request: DaemonControl) -> DaemonStatus:
        del dirs, request
        raise AssertionError("Contacted the daemon at the startup deadline")

    monkeypatch.setattr(daemon, "control", expired)
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
@pytest.mark.usefixtures("model_codec")
@pytest.mark.parametrize("seed", [17, 83, 409])
async def test_message_lifecycle_matches_the_model_across_reconnects_and_upgrades(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    monkeypatch.setattr(host_server, "received", _FrameDecoder(target=Hello).received)
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    clients: list[_ModelClient] = []
    rng = random.Random(seed)  # noqa: S311 -- Repeatable lifecycle ordering, not secrets.
    transitions = ["retry", "reconnect", "upgrade", "withdraw", "spoof", "address"] * 20
    rng.shuffle(transitions)
    server = Server(dirs, listen=[], source_version="initial")
    await server.aopen()
    try:
        original = await server._control(DaemonControl())
        sender = await _ModelClient.connect(server, "sender", clients=clients)
        receiver = await _ModelClient.connect(server, "receiver", clients=clients)
        sender.send(Peers(labels=("a",)))
        receiver.send(Peers(labels=("b",)))
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
                receiver.send(Peers(labels=()))
                await sender.directory(())
                sender.send(note)
                await sender.expect(
                    Refused(id=note.id, reason="Agent 'b#receiver' is not reachable."),
                )
                receiver.send(Peers(labels=("b",)))
                await sender.directory(("b#receiver",))
            if transition == "spoof":
                sender.send(replace(note, sender="unregistered"))
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
                sender.send(replace(note, recipient=address))
                await sender.expect(
                    Refused(id=note.id, reason=f"Agent {address!r} is not reachable."),
                )
            sender.send(note)
            sender.send(note)
            await sender.expect(Ack(id=note.id))
            await sender.expect(Ack(id=note.id))
            delivered = replace(note, sender="a#sender", recipient="b")
            await receiver.expect(delivered)
            if transition == "upgrade":
                sender.writer.close()
                receiver.writer.close()
                status = await server._control(
                    DaemonControl(source_version=f"{seed}-{index}"),
                )
                assert status.restarting
                state = await server.handoff()
                server._sessions.clear()
                await server.aclose()
                server = Server(
                    dirs,
                    listen=[],
                    source_version=f"{seed}-{index}",
                    handoff=state,
                )
                await server.aopen()
                updated = await server._control(DaemonControl())
                assert updated.pid == original.pid
                assert updated.source_version == f"{seed}-{index}"
                sender = await _ModelClient.connect(server, "sender", clients=clients)
                assert sender.welcome.resumed
            if transition in ("upgrade", "reconnect"):
                receiver = await _ModelClient.connect(
                    server,
                    "receiver",
                    clients=clients,
                )
                assert receiver.welcome.resumed
                assert receiver.welcome.calls == (note.id,)
                await receiver.expect(delivered)
            receiver.send(Ack(id=note.id))
            receiver.send(Ack(id=note.id))
            await receiver.barrier()
            sender.send(note)
            await sender.expect(Ack(id=note.id))
            await receiver.barrier()
            receiver = await _ModelClient.connect(server, "receiver", clients=clients)
            assert receiver.welcome.resumed
            assert receiver.welcome.calls == ()
            await receiver.barrier()
        await server._end(server._sessions["sender"], reason="bye")
        await server._end(server._sessions["receiver"], reason="bye")
        await sender.expect(None)
        await receiver.expect(None)
        fresh = await _ModelClient.connect(server, "receiver", clients=clients)
        assert not fresh.welcome.resumed
        assert fresh.welcome.calls == ()
    finally:
        for client in clients:
            client.writer.close()
        await asyncio.gather(
            *(client.task for client in clients),
            return_exceptions=True,
        )
        await server.aclose()


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_daemon")
@pytest.mark.parametrize(
    "transition",
    ["retry", "reconnect", "upgrade", "withdraw", "spoof", "address"],
)
async def test_message_transition_over_the_daemon_socket(
    tmp_path: Path,
    transition: str,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    clients: list[_Client] = []
    try:
        original = await ensure(dirs, version="old")
        sender = await _Client.connect(dirs, "sender", clients=clients)
        receiver = await _Client.connect(dirs, "receiver", clients=clients)
        sender.writer.write(encoded(Peers(labels=("a",))))
        receiver.writer.write(encoded(Peers(labels=("b",))))
        await sender.directory(("b#receiver",))
        note = Note(
            id="note",
            sender="a",
            recipient="b#receiver",
            content="message",
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
            for address in ("", "b", "b#", "b#../receiver", "b@host#receiver"):
                sender.writer.write(encoded(replace(note, recipient=address)))
                await sender.expect(
                    Refused(
                        id=note.id,
                        reason=f"Agent {address!r} is not reachable.",
                    ),
                )
        sender.writer.write(encoded(note) * 2)
        await sender.expect(Ack(id=note.id))
        await sender.expect(Ack(id=note.id))
        delivered = replace(note, sender="a#sender", recipient="b")
        await receiver.expect(delivered)
        if transition == "upgrade":
            sender.writer.close()
            receiver.writer.close()
            updated = await ensure(dirs, version="new")
            assert updated.pid == original.pid
            assert updated.source_version == "new"
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
        receiver.writer.write(encoded(Bye()))
        await receiver.expect(None)
        fresh = await _Client.connect(dirs, "receiver", clients=clients)
        assert not fresh.welcome.resumed
    finally:
        for client in clients:
            client.writer.close()
        _ = await control(dirs, DaemonControl(stop=True))


class _RecordingTransport(asyncio.WriteTransport):
    def __init__(self, reader: asyncio.StreamReader) -> None:
        super().__init__()
        self.reader = reader
        self.lines: list[bytes] = []
        self.written = asyncio.Event()
        self.closed = False

    @override
    def write(self, data: bytes | bytearray | memoryview) -> None:
        self.lines.append(bytes(data))
        self.written.set()

    @override
    def is_closing(self) -> bool:
        return self.closed

    @override
    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.reader.feed_eof()


@dataclass(frozen=True, slots=True, kw_only=True)
class _ModelClient:
    server: Server
    session_id: str
    writer: asyncio.StreamWriter
    transport: _RecordingTransport
    task: asyncio.Task[None]
    welcome: Welcome
    directories: frozenset[bytes]

    @classmethod
    async def connect(
        cls,
        server: Server,
        session_id: str,
        *,
        clients: list[_ModelClient],
    ) -> _ModelClient:
        reader = asyncio.StreamReader()
        transport = _RecordingTransport(reader)
        writer = asyncio.StreamWriter(
            transport,
            asyncio.Protocol(),
            reader,
            asyncio.get_running_loop(),
        )
        reader.feed_data(
            encoded(
                Hello(version=VERSION, token="", session_id=session_id, client="model"),
            ),
        )
        task = asyncio.create_task(server._local_connected(reader, writer))
        await transport.written.wait()
        welcome = stored_record(
            json_loads(transport.lines.pop(0)),
            Welcome,
            classes={"Welcome": Welcome},
        )
        client = cls(
            server=server,
            session_id=session_id,
            writer=writer,
            transport=transport,
            task=task,
            welcome=welcome,
            directories=clients[0].directories
            if clients
            else frozenset(
                encoded(Peers(labels=labels))
                for labels in (
                    (),
                    ("a#sender",),
                    ("b#receiver",),
                )
            ),
        )
        clients.append(client)
        return client

    def send(self, message: Message) -> None:
        self.server._relay(self.server._sessions[self.session_id], message)

    async def expect(self, expected: Message | None) -> None:
        while self.transport.lines and self.transport.lines[0] in self.directories:
            self.transport.lines.pop(0)
        if expected is None:
            assert self.transport.closed
            assert not self.transport.lines
        else:
            assert self.transport.lines.pop(0) == encoded(expected)

    async def directory(self, labels: tuple[str, ...]) -> None:
        expected = encoded(Peers(labels=labels))
        while (line := self.transport.lines.pop(0)) != expected:
            assert line in self.directories

    async def barrier(self) -> None:
        assert all(line in self.directories for line in self.transport.lines)
        self.transport.lines.clear()


@dataclass(frozen=True, slots=True, kw_only=True)
class _FrameDecoder[T: Hello | Tools | Peers]:
    target: type[T]
    frames: dict[bytes, T] = field(default_factory=dict)

    async def received(self, reader: asyncio.StreamReader) -> T | None:
        line = await reader.readline()
        if not line:
            return None
        if line not in self.frames:
            self.frames[line] = stored_record(
                json_loads(line),
                self.target,
                classes={self.target.__name__: self.target, "ToolSpec": ToolSpec},
            )
        cached = self.frames[line]
        return deepcopy(cached) if isinstance(cached, Tools) else replace(cached)


@dataclass(frozen=True, slots=True, kw_only=True)
class _PeerEncoder:
    frames: dict[Peers, bytes] = field(default_factory=dict[Peers, bytes])

    def __call__(self, message: Message) -> bytes:
        if not isinstance(message, Peers):
            return encoded(message)
        if message not in self.frames:
            self.frames[message] = encoded(message)
        return self.frames[message]


@pytest.fixture
def model_codec(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = _PeerEncoder()
    monkeypatch.setattr(host_server, "encoded", encoder)
    monkeypatch.setattr(host_session, "encoded", encoder)


@pytest.mark.asyncio
async def test_cached_frames_preserve_canonical_values_without_shared_mutation() -> (
    None
):
    message = Tools(
        tools={
            "Gate": ToolSpec(
                description="gate",
                schema={"type": "object", "properties": {"x": {"enum": ["a", "b"]}}},
            ),
        },
    )
    frame = encoded(message)
    canonical = asyncio.StreamReader()
    canonical.feed_data(frame)
    expected = await received(canonical)
    reader = asyncio.StreamReader()
    reader.feed_data(frame * 2)
    changed = encoded(Tools(tools={"Other": ToolSpec(description="other", schema={})}))
    reader.feed_data(changed)
    reader.feed_eof()
    decoder = _FrameDecoder(target=Tools)
    first = await decoder.received(reader)
    assert isinstance(first, Tools)
    assert isinstance(expected, Tools)
    assert first == expected
    assert encoded(first) == encoded(expected) == frame
    schema = first.tools["Gate"].schema
    assert isinstance(schema, dict)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    parameter = properties["x"]
    assert isinstance(parameter, dict)
    choices = parameter["enum"]
    assert isinstance(choices, list)
    choices.append("changed")
    parameter["description"] = "changed"
    second = await decoder.received(reader)
    assert isinstance(second, Tools)
    assert second is not first
    assert second == expected
    assert first != second
    assert encoded(second) == frame
    different = await decoder.received(reader)
    assert different is not None
    assert tuple(different.tools) == ("Other",)
    assert different is not first
    assert await decoder.received(reader) is None
    encoder = _PeerEncoder()
    for peers in (
        Peers(labels=("a",)),
        Peers(labels=("a",), public=()),
        Peers(labels=("b",)),
    ):
        assert encoder(peers) == encoded(peers)
        assert encoder(peers) == encoded(peers)
    assert encoder(Ack(id="uncached")) == encoded(Ack(id="uncached"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        Hello(
            version=VERSION,
            token=secrets.token_hex(16),
            session_id="session",
            client="client",
        ),
        Hello(
            version=VERSION,
            token=secrets.token_hex(16),
            session_id="session",
            client="client",
            tools=("Gate",),
        ),
        Peers(labels=("a",)),
        Peers(labels=("a",), public=("a",)),
        Peers(labels=("a",), public=()),
    ],
)
async def test_cached_immutable_frames_preserve_values_and_fresh_records(
    message: Hello | Peers,
) -> None:
    frame = encoded(message)
    reader = asyncio.StreamReader()
    changed = (
        replace(message, client="changed")
        if isinstance(message, Hello)
        else replace(message, labels=("changed",))
    )
    reader.feed_data(frame * 2 + encoded(changed))
    reader.feed_eof()
    decoder = _FrameDecoder(target=type(message))
    first = await decoder.received(reader)
    second = await decoder.received(reader)
    assert first == second == message
    assert first is not second
    assert first is not None
    assert second is not None
    assert encoded(first) == encoded(second) == frame
    if isinstance(first, Hello):
        assert first.tools is None or isinstance(first.tools, tuple)
    else:
        assert isinstance(first.labels, tuple)
        assert first.public is None or isinstance(first.public, tuple)
    different = await decoder.received(reader)
    assert different == changed
    assert different != first
    assert different is not None
    assert encoded(different) == encoded(changed)
    assert await decoder.received(reader) is None


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
@pytest.mark.usefixtures("local_daemon")
async def test_independent_host_actions_share_one_persistent_daemon() -> None:
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
@pytest.mark.usefixtures("local_daemon")
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
@pytest.mark.usefixtures("local_daemon")
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
@pytest.mark.usefixtures("local_daemon")
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
@pytest.mark.usefixtures("model_codec")
@pytest.mark.skipif(sys.platform != "linux", reason="Measures Linux process resources.")
@pytest.mark.parametrize("delayed_close", [False, True])
async def test_worker_churn_releases_processes_descriptors_and_public_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delayed_close: bool,
    local_daemon: LocalHost,
) -> None:
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    clients: list[_ModelClient] = []
    counts: list[int] = []
    async with Server(
        dirs,
        listen=[strings.Address(host="127.0.0.1", port=0)],
        tools=["Gate"],
        start_host=local_daemon,
    ) as server:
        (connection,) = await strings.add(dirs, peer_messaging=True)
        observer = await _ModelClient.connect(server, "observer", clients=clients)
        monkeypatch.setattr(
            host_server,
            "received",
            _FrameDecoder(target=Tools).received,
        )
        monkeypatch.setattr(
            host_session,
            "received",
            _FrameDecoder(target=Peers).received,
        )
        descriptors = Path(f"/proc/{os.getpid()}/fd")
        baseline = _descriptor_count(descriptors)
        for iteration in range(26):
            sessions: list[_Session] = []
            for index in range(4):
                transport = _RecordingTransport(asyncio.StreamReader())
                writer = asyncio.StreamWriter(
                    transport,
                    asyncio.Protocol(),
                    None,
                    asyncio.get_running_loop(),
                )
                session = await server._admit(
                    Hello(
                        version=VERSION,
                        token=connection.token,
                        session_id=f"worker-{iteration}-{index}",
                        client="churn",
                    ),
                    ip="127.0.0.1",
                    writer=writer,
                )
                assert session is not None
                sessions.append(session)
                server._relay(session, Peers(labels=("worker@churn",)))
                server._relay(session, Describe(id="spec", tool="Gate"))
                spec = session.specs["Gate"]
                assert spec.description
                assert transport.lines[-1] == encoded(Described(id="spec", spec=spec))
            assert observer.transport.lines[-1] == encoded(
                Peers(
                    labels=tuple(f"worker@churn#{session.id}" for session in sessions),
                ),
            )
            observer.transport.lines.clear()
            held = os.open(os.devnull, os.O_RDONLY) if delayed_close else None
            try:
                await asyncio.gather(
                    *(server._end(session, reason="bye") for session in sessions),
                )
                assert observer.transport.lines[-1] == encoded(Peers(labels=()))
                observer.transport.lines.clear()
                assert not any(
                    row.token_id for row in strings.listing(dirs).sessions or ()
                )
                assert len(local_daemon.started) == 4 * (iteration + 1)
                assert all(worker.task.done() for worker in local_daemon.started)
                for worker in local_daemon.started[-len(sessions) :]:
                    await worker.writer.wait_closed()
                if held is not None:
                    assert _descriptor_count(descriptors) == baseline + 1
            finally:
                if held is not None:
                    os.close(held)
            counts.append(_descriptor_count(descriptors))
            assert counts[-1] == baseline, counts
        await server._end(server._sessions["observer"], reason="bye")
        await observer.expect(None)
        await observer.task


def _descriptor_count(path: Path) -> int:
    return len(list(path.iterdir()))


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Measures Linux process resources.")
@pytest.mark.parametrize("delayed_close", [False, True])
async def test_tls_worker_batch_releases_descriptors_and_public_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delayed_close: bool,
    local_daemon: LocalHost,
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
            if delayed_close:
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
        assert len(local_daemon.started) == 4
        assert all(worker.task.done() for worker in local_daemon.started)
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


@dataclass(slots=True, kw_only=True)
class _ControlPeer:
    """A Unix protocol peer with an externally releasable connection or response."""

    dirs: ServeDirs
    status: DaemonStatus
    delay_status: bool = False
    pending: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    requests: list[Message | None] = field(default_factory=list)
    tasks: list[asyncio.Task[None]] = field(default_factory=list)

    async def connect(
        self,
        path: Path,
        *,
        limit: int,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        self.pending.set()
        await self.release.wait()
        return await asyncio.open_unix_connection(path, limit=limit)

    def connected(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.tasks.append(asyncio.create_task(self.reply(reader, writer=writer)))

    async def reply(
        self,
        reader: asyncio.StreamReader,
        *,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            self.requests.append(await received(reader))
            if self.delay_status:
                self.pending.set()
                await self.release.wait()
            writer.write(encoded(self.status))
            await writer.drain()
        except ConnectionError:
            pass
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()


@pytest_asyncio.fixture
async def control_peer(tmp_path: Path) -> AsyncIterator[_ControlPeer]:
    """Serve real control frames and join every accepted connection on teardown."""
    dirs = ServeDirs(secrets=tmp_path / "secrets", state=tmp_path / "state")
    dirs.state.mkdir()
    peer = _ControlPeer(
        dirs=dirs,
        status=DaemonStatus(pid=os.getpid(), source_version="fixture", listen=()),
    )
    async with await asyncio.start_unix_server(
        peer.connected,
        path=socket_address(dirs.state / "daemon.sock"),
        limit=MAX_LINE_BYTES,
    ):
        try:
            yield peer
        finally:
            peer.release.set()
            await asyncio.gather(*peer.tasks)


async def _cli(*arguments: str) -> str:
    assert arguments[0] == "host"
    words = list(arguments[1:])
    verb = (words.pop(0),) if words and not words[0].startswith("--") else ()
    shown: list[str] = []
    owner = HostSettings(
        dirs=ServeDirs(
            secrets=config_dir() / "rekursiv-ai" / "secrets" / "sagent" / "serve",
            state=state_dir() / "rekursiv-ai" / "sagent" / "serve",
        ),
        show=shown.append,
    )
    action = next(setting for setting in owner.settings() if setting.name == verb)
    result = await action.apply(shlex.join(words))
    return "\n".join([*shown, result])


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
