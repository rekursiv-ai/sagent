"""Contracts for WorkloadIdentity, against a fake token endpoint and real small commands."""

from __future__ import annotations

from dataclasses import dataclass, field

import asyncio

from treekle import json_dumps, json_loads

import httpx2
import pytest

from sagent.model.exceptions import AuthError
from sagent.providers.workload_identity import WorkloadIdentity


_URL = "https://auth.example.test/oauth/token"


@dataclass(frozen=False, slots=True, kw_only=True)
class _Endpoint:
    """Answers each exchange with the next reply, and notes what it was sent."""

    replies: list[httpx2.Response]
    subjects: list[object] = field(default_factory=list)
    http: httpx2.AsyncClient = field(init=False)

    def __post_init__(self) -> None:
        self.http = httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle))

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.subjects.append(json_loads(request.content, dict[str, object])["subject"])
        return self.replies.pop(0)


def _granted(token: str, expires_in: int = 3600) -> httpx2.Response:
    body = json_dumps({"access_token": token, "expires_in": expires_in}).encode()
    return httpx2.Response(200, content=body)


def _workload_identity(
    endpoint: _Endpoint,
    *,
    command: str = "/bin/echo identity",
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


@pytest.mark.asyncio
async def test_a_token_is_exchanged_from_what_the_command_prints() -> None:
    endpoint = _Endpoint(replies=[_granted("vendor")])

    assert await _workload_identity(endpoint).token() == "vendor"
    assert endpoint.subjects == ["identity"]


@pytest.mark.asyncio
async def test_the_exchange_does_not_get_the_clients_stream_allowance() -> None:
    timeouts: list[object] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        timeouts.append(request.extensions["timeout"])
        return _granted("vendor")

    endpoint = _Endpoint(replies=[])
    endpoint.http = httpx2.AsyncClient(
        timeout=600.0,
        transport=httpx2.MockTransport(handle),
    )

    await _workload_identity(endpoint).token()

    assert timeouts == [dict.fromkeys(("connect", "read", "write", "pool"), 30.0)]


@pytest.mark.asyncio
async def test_a_live_token_is_reused_without_a_new_exchange() -> None:
    endpoint = _Endpoint(replies=[_granted("vendor")])
    identity = _workload_identity(endpoint)

    await identity.token()

    assert await identity.token() == "vendor"
    assert len(endpoint.subjects) == 1


@pytest.mark.parametrize(
    ("elapsed", "exchanges"),
    [(238.0, 1), (240.0, 2)],
)
@pytest.mark.asyncio
async def test_a_token_is_renewed_a_minute_before_it_expires(
    elapsed: float,
    exchanges: int,
) -> None:
    endpoint = _Endpoint(replies=[_granted("old", 300), _granted("new", 300)])
    clock = [0.0]
    identity = _workload_identity(endpoint, clock=clock)
    await identity.token()

    clock[0] = elapsed
    await identity.token()

    assert len(endpoint.subjects) == exchanges


@pytest.mark.asyncio
async def test_a_forgotten_token_is_exchanged_again() -> None:
    endpoint = _Endpoint(replies=[_granted("old"), _granted("new")])
    identity = _workload_identity(endpoint)
    await identity.token()

    identity.forget()

    assert await identity.token() == "new"


@pytest.mark.asyncio
async def test_concurrent_requests_share_one_exchange() -> None:
    endpoint = _Endpoint(replies=[_granted("vendor")])
    identity = _workload_identity(endpoint)

    tokens = await asyncio.gather(identity.token(), identity.token())

    assert tokens == ["vendor", "vendor"]
    assert len(endpoint.subjects) == 1


@pytest.mark.asyncio
async def test_a_refusal_names_the_status_and_keeps_the_body_out_of_the_error() -> None:
    endpoint = _Endpoint(replies=[httpx2.Response(401, content=b"secret-detail")])

    with pytest.raises(AuthError, match="HTTP 401") as raised:
        await _workload_identity(endpoint).token()

    assert "secret-detail" not in str(raised.value)


@pytest.mark.asyncio
async def test_a_reply_with_no_token_is_an_auth_error() -> None:
    endpoint = _Endpoint(replies=[httpx2.Response(200, content=b"{}")])

    with pytest.raises(AuthError, match="no access token"):
        await _workload_identity(endpoint).token()


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("/usr/bin/env false", "command failed"),
        ("/usr/bin/env true", "command failed"),
        ("/nonexistent/identity", "Could not run"),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_command_never_reaches_the_vendor(
    command: str,
    message: str,
) -> None:
    endpoint = _Endpoint(replies=[])

    with pytest.raises(AuthError, match=message):
        await _workload_identity(endpoint, command=command).token()

    assert endpoint.subjects == []


@pytest.mark.asyncio
async def test_a_stuck_command_is_killed_and_reported() -> None:
    endpoint = _Endpoint(replies=[])
    identity = WorkloadIdentity(
        resources=endpoint,
        url=_URL,
        command="/usr/bin/env sleep 30",
        request=lambda subject: {"subject": subject},
        command_timeout_sec=0.02,
    )

    with pytest.raises(AuthError, match="timed out"):
        await identity.token()


@pytest.mark.asyncio
async def test_each_exchange_runs_the_command_afresh(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    count = tmp_path_factory.mktemp("count") / "count"
    script = count.with_name("mint")
    script.write_text(
        f'#!/bin/sh\necho x >> "{count}"\nwc -l < "{count}"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    endpoint = _Endpoint(replies=[_granted("a"), _granted("b")])
    identity = _workload_identity(endpoint, command=str(script))
    await identity.token()
    identity.forget()

    await identity.token()

    assert [str(subject).strip() for subject in endpoint.subjects] == ["1", "2"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
