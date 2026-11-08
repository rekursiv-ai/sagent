"""Contracts for workload tokens, exchange decisions, and identity commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict
from unittest.mock import AsyncMock, Mock, call

import asyncio

from treekle import ReadError, json_dumps, json_loads

import httpx2
import pytest

from sagent.model.exceptions import AuthError
from sagent.providers.workload_identity import WorkloadIdentity


_URL = "https://auth.example.test/oauth/token"


class _Exchange(TypedDict):
    subject: str


@dataclass(slots=True, kw_only=True)
class _Endpoint:
    """Answer each exchange and record its subject, URL, and timeout."""

    replies: list[httpx2.Response]
    subjects: list[str] = field(default_factory=list)
    requests: list[httpx2.Request] = field(default_factory=list)
    http: httpx2.AsyncClient = field(init=False)

    def __post_init__(self) -> None:
        self.http = httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle))

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        self.subjects.append(json_loads(request.content, _Exchange)["subject"])
        return self.replies.pop(0)


def _granted(token: str, expires_in: int = 3600) -> httpx2.Response:
    body = json_dumps({"access_token": token, "expires_in": expires_in}).encode()
    return httpx2.Response(200, content=body)


def _workload_identity(
    endpoint: _Endpoint,
    *,
    command: str = "identity-token",
    clock: list[float] | None = None,
) -> WorkloadIdentity:
    now = [0.0] if clock is None else clock
    return WorkloadIdentity(
        resources=endpoint,
        url=_URL,
        command=command,
        request=lambda subject: {"subject": subject},
        clock=lambda: now[0],
    )


@pytest.fixture
def identity_command(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    command = AsyncMock(return_value="identity")
    monkeypatch.setattr(WorkloadIdentity, "_identity", command)
    return command


@pytest.mark.asyncio
async def test_a_token_is_exchanged_from_the_identity_command(
    identity_command: AsyncMock,
) -> None:
    endpoint = _Endpoint(replies=[_granted("vendor")])
    identity = _workload_identity(endpoint)
    assert (identity._token, identity._expires) == ("", 0.0)

    assert await identity.token() == "vendor"

    identity_command.assert_awaited_once_with()
    assert endpoint.subjects == ["identity"]
    assert len(endpoint.requests) == 1
    assert str(endpoint.requests[0].url) == _URL
    assert endpoint.requests[0].extensions["timeout"] == dict.fromkeys(
        ("connect", "read", "write", "pool"),
        30.0,
    )


@pytest.mark.asyncio
async def test_the_exchange_uses_its_configured_timeout(
    identity_command: AsyncMock,
) -> None:
    endpoint = _Endpoint(replies=[_granted("vendor")])
    identity = WorkloadIdentity(
        resources=endpoint,
        url=_URL,
        command="identity-token",
        request=lambda subject: {"subject": subject},
        exchange_timeout_sec=7.0,
    )

    assert await identity.token() == "vendor"

    identity_command.assert_awaited_once_with()
    assert len(endpoint.requests) == 1
    assert str(endpoint.requests[0].url) == _URL
    assert endpoint.subjects == ["identity"]
    assert endpoint.requests[0].extensions["timeout"] == dict.fromkeys(
        ("connect", "read", "write", "pool"),
        7.0,
    )


@pytest.mark.asyncio
async def test_a_live_token_is_reused_without_a_new_exchange(
    identity_command: AsyncMock,
) -> None:
    endpoint = _Endpoint(replies=[_granted("vendor")])
    identity = _workload_identity(endpoint)

    assert await identity.token() == "vendor"
    assert await identity.token() == "vendor"

    identity_command.assert_awaited_once_with()
    assert endpoint.subjects == ["identity"]


@pytest.mark.parametrize(
    ("elapsed", "expected", "exchanges"),
    [(239.0, "old", 1), (240.0, "new", 2)],
)
@pytest.mark.asyncio
async def test_a_token_is_renewed_a_minute_before_it_expires(
    identity_command: AsyncMock,
    elapsed: float,
    expected: str,
    exchanges: int,
) -> None:
    endpoint = _Endpoint(replies=[_granted("old", 300), _granted("new", 300)])
    clock = [0.0]
    identity = _workload_identity(endpoint, clock=clock)
    assert await identity.token() == "old"

    clock[0] = elapsed

    assert await identity.token() == expected
    assert identity_command.await_args_list == [call()] * exchanges
    assert endpoint.subjects == ["identity"] * exchanges


@pytest.mark.asyncio
async def test_a_forgotten_token_exchanges_a_fresh_identity(
    identity_command: AsyncMock,
) -> None:
    identity_command.side_effect = ["first", "second"]
    endpoint = _Endpoint(replies=[_granted("old"), _granted("new")])
    identity = _workload_identity(endpoint)
    assert await identity.token() == "old"

    identity.forget()

    assert await identity.token() == "new"
    assert identity_command.await_args_list == [call(), call()]
    assert endpoint.subjects == ["first", "second"]


@pytest.mark.asyncio
async def test_concurrent_requests_share_one_exchange(
    identity_command: AsyncMock,
) -> None:
    entered, admitted = asyncio.Event(), asyncio.Event()

    async def subject() -> str:
        entered.set()
        await admitted.wait()
        return "identity"

    identity_command.side_effect = subject
    endpoint = _Endpoint(replies=[_granted("vendor")])
    identity = _workload_identity(endpoint)
    first = asyncio.create_task(identity.token())
    second = asyncio.create_task(identity.token())
    try:
        await entered.wait()
        identity_command.assert_awaited_once_with()
        assert not first.done()
        assert not second.done()
        assert endpoint.requests == []
        admitted.set()
        assert await asyncio.gather(first, second) == ["vendor", "vendor"]
    finally:
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
    identity_command.assert_awaited_once_with()
    assert endpoint.subjects == ["identity"]


@pytest.mark.asyncio
async def test_a_refusal_names_the_status_and_keeps_the_body_out_of_the_error(
    identity_command: AsyncMock,
) -> None:
    endpoint = _Endpoint(replies=[httpx2.Response(401, content=b"secret-detail")])

    with pytest.raises(AuthError, match="HTTP 401") as raised:
        await _workload_identity(endpoint).token()

    identity_command.assert_awaited_once_with()
    assert "secret-detail" not in str(raised.value)


@pytest.mark.asyncio
async def test_a_reply_with_no_token_is_an_auth_error(
    identity_command: AsyncMock,
) -> None:
    endpoint = _Endpoint(replies=[httpx2.Response(200, content=b"{}")])

    with pytest.raises(AuthError, match="no access token"):
        await _workload_identity(endpoint).token()

    identity_command.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry", [{}, {"expires_in": None}], ids=["missing", "null"])
async def test_an_exchange_without_a_lifetime_expires_at_its_start(
    identity_command: AsyncMock,
    expiry: dict[str, object],
) -> None:
    endpoint = _Endpoint(
        replies=[
            httpx2.Response(200, content=json_dumps({"access_token": "old", **expiry})),
            _granted("new"),
        ],
    )
    clock = [17.0]
    identity = _workload_identity(endpoint, clock=clock)

    assert await identity.token() == "old"
    assert identity._expires == 17.0
    assert await identity.token() == "new"
    assert identity_command.await_args_list == [call(), call()]
    assert endpoint.subjects == ["identity", "identity"]


@pytest.mark.asyncio
async def test_an_exchange_reply_must_be_an_object(
    identity_command: AsyncMock,
) -> None:
    endpoint = _Endpoint(replies=[httpx2.Response(200, content=b"[]")])

    with pytest.raises(ReadError):
        await _workload_identity(endpoint).token()

    identity_command.assert_awaited_once_with()


def _command(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    output: bytes = b"identity\n",
) -> tuple[AsyncMock, AsyncMock, Mock, AsyncMock]:
    communicated = AsyncMock(return_value=(output, b""))
    killed, waited = Mock(), AsyncMock(return_value=returncode)
    process = Mock(
        spec=asyncio.subprocess.Process,
        returncode=returncode,
        communicate=communicated,
        kill=killed,
        wait=waited,
    )
    spawned = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawned)
    return spawned, communicated, killed, waited


@pytest.mark.parametrize(
    ("returncode", "output", "error", "message"),
    [
        (1, b"identity", None, "command failed: 1"),
        (0, b" \n", None, "command failed: 0"),
        (0, b"", OSError("missing command"), "Could not run"),
    ],
    ids=["nonzero", "empty", "cannot-start"],
)
@pytest.mark.asyncio
async def test_a_failed_command_never_reaches_the_vendor(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    output: bytes,
    error: OSError | None,
    message: str,
) -> None:
    spawned, communicated, killed, waited = _command(
        monkeypatch,
        returncode=returncode,
        output=output,
    )
    spawned.side_effect = error
    endpoint = _Endpoint(replies=[])

    with pytest.raises(AuthError, match=message) as raised:
        await _workload_identity(endpoint).token()

    assert raised.value.__cause__ is error
    assert communicated.await_count == (0 if error else 1)
    killed.assert_not_called()
    waited.assert_not_awaited()
    assert endpoint.requests == []


@pytest.mark.asyncio
async def test_a_stuck_command_is_killed_reaped_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, communicated, killed, waited = _command(monkeypatch)
    communicated.side_effect = TimeoutError()
    timeout = Mock(wraps=asyncio.timeout)
    monkeypatch.setattr(asyncio, "timeout", timeout)
    endpoint = _Endpoint(replies=[])
    identity = WorkloadIdentity(
        resources=endpoint,
        url=_URL,
        command="identity-token",
        request=lambda subject: {"subject": subject},
        command_timeout_sec=0.02,
    )

    with pytest.raises(AuthError, match="timed out") as raised:
        await identity.token()

    assert raised.value.__cause__ is None
    timeout.assert_called_once_with(0.02)
    communicated.assert_awaited_once_with()
    killed.assert_called_once_with()
    waited.assert_awaited_once_with()
    assert endpoint.requests == []


@pytest.mark.asyncio
async def test_each_identity_read_runs_the_command_afresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned, communicated, killed, waited = _command(monkeypatch)
    communicated.side_effect = [(b" first\n", b""), (b"second\xff\n", b"")]
    timeout = Mock(wraps=asyncio.timeout)
    monkeypatch.setattr(asyncio, "timeout", timeout)
    endpoint = _Endpoint(replies=[])
    identity = _workload_identity(endpoint, command="mint --audience 'two words'")

    assert await identity._identity() == "first"
    assert await identity._identity() == "second\ufffd"
    assert timeout.call_args_list == [call(30.0), call(30.0)]

    assert (
        spawned.await_args_list
        == [
            call(
                "mint",
                "--audience",
                "two words",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            ),
        ]
        * 2
    )
    assert communicated.await_args_list == [call(), call()]
    killed.assert_not_called()
    waited.assert_not_awaited()
    assert endpoint.requests == []


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
