"""Unit tests for process admission, spawn options, and shared-resource ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, call

import asyncio

import httpx2
import pytest

from sagent.lib.posix.child import Child
from sagent.providers import resources
from sagent.providers.resources import SharedResources, SubprocessPool, close_all


if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("argv", "env", "reused"),
    [
        (["cli", "chat"], {"ACCOUNT": "work", "OTHER": "value"}, True),
        (["cli", "chat"], {"OTHER": "value", "ACCOUNT": "work"}, True),
        (["cli", "chat"], {"ACCOUNT": "other", "OTHER": "value"}, False),
        (["cli", "chat", "extra"], {"ACCOUNT": "work", "OTHER": "value"}, False),
    ],
    ids=["same", "environment-order", "other-account", "other-command"],
)
async def test_acquire_reuses_only_its_command_and_environment_and_replaces_the_spare(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    env: Mapping[str, str],
    reused: bool,
) -> None:
    first, spare, replacement, successor = [Mock(returncode=None) for _ in range(4)]
    spawn = AsyncMock(side_effect=[first, spare, replacement, successor])
    pool = SubprocessPool()
    monkeypatch.setattr(pool, "_spawn", spawn)
    original = {"ACCOUNT": "work", "OTHER": "value"}
    async with pool:
        assert await pool.acquire(["cli", "chat"], env=original) is first
        assert len(pool._spares) == 1
        assert await next(iter(pool._spares.values())) is spare
        assert await pool.acquire(argv, env=env) is (spare if reused else replacement)
    assert pool._spares == {}
    assert spawn.await_args_list == [
        call(["cli", "chat"], env=original),
        call(["cli", "chat"], env=original),
        *([call(argv, env=env)] * (1 if reused else 2)),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [0, -9])
async def test_a_dead_spare_is_released_before_its_replacement_starts(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    first, spare, replacement, successor = [Mock(returncode=None) for _ in range(4)]
    spare.returncode = status
    calls = Mock()
    spawn = AsyncMock(side_effect=[first, spare, replacement, successor])
    release = AsyncMock()
    calls.attach_mock(spawn, "spawn")
    calls.attach_mock(release, "release")
    pool = SubprocessPool()
    monkeypatch.setattr(pool, "_spawn", spawn)
    monkeypatch.setattr(pool, "release", release)
    async with pool:
        assert await pool.acquire(["cli"], env={}) is first
        assert await pool.acquire(["cli"], env={}) is replacement
    assert calls.mock_calls == [
        ("spawn", (["cli"],), {"env": {}}),
        ("spawn", (["cli"],), {"env": {}}),
        ("release", (spare,), {}),
        ("spawn", (["cli"],), {"env": {}}),
        ("spawn", (["cli"],), {"env": {}}),
    ]


@pytest.mark.asyncio
async def test_a_spare_that_failed_to_start_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, replacement, successor = [Mock(returncode=None) for _ in range(3)]
    spawn = AsyncMock(
        side_effect=[first, FileNotFoundError("cli"), replacement, successor],
    )
    pool = SubprocessPool()
    monkeypatch.setattr(pool, "_spawn", spawn)
    async with pool:
        assert await pool.acquire(["cli"], env={}) is first
        assert await pool.acquire(["cli"], env={}) is replacement
    assert spawn.await_args_list == [call(["cli"], env={})] * 4


@pytest.mark.asyncio
async def test_a_failed_initial_start_propagates_without_scheduling_a_spare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = FileNotFoundError("cli")
    spawn = AsyncMock(side_effect=error)
    pool = SubprocessPool()
    monkeypatch.setattr(pool, "_spawn", spawn)
    with pytest.raises(FileNotFoundError) as caught:
        await pool.acquire(["cli"], env={})
    assert caught.value is error
    spawn.assert_awaited_once_with(["cli"], env={})
    assert pool._spares == {}


@pytest.mark.asyncio
async def test_spawn_forwards_the_command_environment_and_stream_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = Mock()
    started = AsyncMock(return_value=Mock(process=process))
    monkeypatch.setattr(Child, "start", started)
    pool = SubprocessPool()
    env = {"ACCOUNT": "work"}
    assert await pool._spawn(["cli", "chat"], env=env) is process
    assert pool._live == {process}
    started.assert_awaited_once_with(
        "cli",
        "chat",
        group=True,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
        limit=16 * 1024 * 1024,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("has_stdin", [False, True])
@pytest.mark.parametrize("grace", [None, 0.05])
async def test_release_forgets_and_closes_the_process_before_delegating_its_stop(
    monkeypatch: pytest.MonkeyPatch,
    has_stdin: bool,
    grace: float | None,
) -> None:
    pool = SubprocessPool() if grace is None else SubprocessPool(grace_sec=grace)
    closed = Mock()
    stdin = Mock(close=closed) if has_stdin else None
    process = Mock(stdin=stdin)
    pool._live.add(process)
    stop = AsyncMock()

    def stopped_child(**options: object) -> Mock:
        del options
        assert process not in pool._live
        return Mock(stop=stop)

    child = Mock(side_effect=stopped_child)
    calls = Mock()
    calls.attach_mock(closed, "close")
    calls.attach_mock(child, "child")
    calls.attach_mock(stop, "stop")
    monkeypatch.setattr(resources, "Child", child)
    assert await pool.release(process) is None
    assert pool._live == set()
    await pool.aclose()
    await pool.release(process)
    one_release: list[tuple[str, tuple[object, ...], dict[str, object]]] = [
        *([("close", (), {})] if has_stdin else []),
        (
            "child",
            (),
            {
                "process": process,
                "group": True,
                "grace_sec": 5.0 if grace is None else grace,
            },
        ),
        ("stop", (), {}),
    ]
    assert calls.mock_calls == one_release * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("stop"), asyncio.CancelledError()])
async def test_release_propagates_stop_failure_after_forgetting_and_closing(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    pool = SubprocessPool()
    closed = Mock()
    process = Mock(stdin=Mock(close=closed))
    pool._live.add(process)
    stopped = AsyncMock(side_effect=error)
    monkeypatch.setattr(resources, "Child", Mock(return_value=Mock(stop=stopped)))
    with pytest.raises(type(error)) as caught:
        await pool.release(process)
    assert caught.value is error
    assert pool._live == set()
    closed.assert_called_once_with()
    stopped.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_spare", [False, True])
async def test_closing_awaits_each_spare_and_releases_every_live_process(
    monkeypatch: pytest.MonkeyPatch,
    failed_spare: bool,
) -> None:
    pool = SubprocessPool()
    first, second = Mock(), Mock()
    pool._live.add(first)
    ready, finish = asyncio.Event(), asyncio.Event()
    spawning = asyncio.create_task(
        _pending_spare(pool, second, ready=ready, finish=finish),
    )
    pool._spares[(("cli",), ())] = spawning
    if failed_spare:
        failed = asyncio.create_task(_failed_spare())
        pool._spares[(("missing",), ())] = failed
    released: list[Mock] = []

    def release(process: Mock) -> None:
        pool._live.discard(process)
        released.append(process)

    monkeypatch.setattr(pool, "release", AsyncMock(side_effect=release))
    spares = tuple(pool._spares.values())
    closing = asyncio.create_task(pool.aclose())
    try:
        await ready.wait()
        assert not closing.done()
        assert not spawning.cancelled()
        finish.set()
        await closing
        assert set(released) == {first, second}
        assert len(released) == 2
        assert pool._spares == {}
        assert pool._live == set()
    finally:
        finish.set()
        await asyncio.gather(closing, *spares, return_exceptions=True)


async def _pending_spare(
    pool: SubprocessPool,
    process: Mock,
    *,
    ready: asyncio.Event,
    finish: asyncio.Event,
) -> Mock:
    ready.set()
    await finish.wait()
    pool._live.add(process)
    return process


async def _failed_spare() -> Mock:
    raise FileNotFoundError("cli")


@pytest.mark.asyncio
async def test_closing_continues_after_a_process_has_already_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = SubprocessPool()
    first, second = Mock(), Mock()
    pool._live.update([first, second])

    def gone(process: Mock) -> None:
        pool._live.discard(process)
        raise ProcessLookupError

    released = AsyncMock(side_effect=gone)
    monkeypatch.setattr(pool, "release", released)
    await pool.aclose()
    assert released.await_count == 2
    assert {args.args[0] for args in released.await_args_list} == {first, second}
    assert pool._live == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError("pool")])
async def test_shared_resources_open_lazily_close_in_order_and_reopen(
    monkeypatch: pytest.MonkeyPatch,
    failure: RuntimeError | None,
) -> None:
    gate = Mock()
    gates = Mock(return_value=gate)
    client, reopened_client, server, reopened_server = [Mock() for _ in range(4)]
    first, second, reopened_pool = [Mock() for _ in range(3)]
    clients = Mock(side_effect=[client, reopened_client])
    servers = Mock(side_effect=[server, reopened_server])
    pools = Mock(side_effect=[first, second, reopened_pool])
    monkeypatch.setattr(resources, "FileGate", gates)
    monkeypatch.setattr(httpx2, "AsyncClient", clients)
    monkeypatch.setattr(resources, "McpToolServer", servers)
    monkeypatch.setattr(resources, "SubprocessPool", pools)
    shared = SharedResources()
    assert shared.gate is gate
    gates.assert_called_once_with()
    for factory in (clients, servers, pools):
        factory.assert_not_called()
    assert shared.http is shared.http is client
    assert shared.tool_server is shared.tool_server is server
    assert shared.process_pool("a") is shared.process_pool("a") is first
    assert shared.process_pool("b") is second
    assert pools.call_args_list == [call(), call()]
    servers.assert_called_once_with()
    clients.assert_called_once_with(
        timeout=httpx2.Timeout(600.0, connect=30.0),
    )
    closed = Mock()

    def cleared() -> None:
        assert shared._pools == {}
        assert shared._http is shared._tool_server is None

    first_close = AsyncMock(side_effect=cleared)
    second_close = AsyncMock(side_effect=failure)
    server_close, client_close = AsyncMock(), AsyncMock()
    for obj, closer, name in (
        (first, first_close, "first"),
        (second, second_close, "second"),
        (server, server_close, "server"),
        (client, client_close, "client"),
    ):
        obj.aclose = closer
        closed.attach_mock(closer, name)
    if failure is None:
        await shared.aclose()
    else:
        with pytest.raises(RuntimeError) as caught:
            await shared.aclose()
        assert caught.value is failure
    assert closed.mock_calls == [
        ("first", (), {}),
        ("second", (), {}),
        ("server", (), {}),
        ("client", (), {}),
    ]
    await shared.aclose()
    assert closed.mock_calls == [
        ("first", (), {}),
        ("second", (), {}),
        ("server", (), {}),
        ("client", (), {}),
    ]
    assert shared.http is reopened_client
    assert shared.tool_server is reopened_server
    assert shared.process_pool("a") is reopened_pool
    assert clients.call_count == servers.call_count == 2
    assert pools.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [None, ValueError("close"), asyncio.CancelledError()])
async def test_close_all_preserves_order_and_continues_after_failure(
    error: BaseException | None,
) -> None:
    calls = Mock()
    first, second = AsyncMock(side_effect=error), AsyncMock()
    calls.attach_mock(first, "first")
    calls.attach_mock(second, "second")
    opened = [Mock(aclose=first), Mock(aclose=second)]
    if error is None:
        await close_all(opened)
    else:
        with pytest.raises(type(error)) as caught:
            await close_all(opened)
        assert caught.value is error
    assert calls.mock_calls == [("first", (), {}), ("second", (), {})]


@pytest.mark.asyncio
async def test_a_later_close_failure_retains_the_earlier_error_as_context() -> None:
    earlier, later = ValueError("first"), RuntimeError("second")
    last = AsyncMock()
    with pytest.raises(RuntimeError) as caught:
        await close_all(
            [
                Mock(aclose=AsyncMock(side_effect=earlier)),
                Mock(aclose=AsyncMock(side_effect=later)),
                Mock(aclose=last),
            ],
        )
    assert caught.value is later
    assert later.__context__ is earlier
    last.assert_awaited_once_with()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
