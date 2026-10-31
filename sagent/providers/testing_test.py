"""Tests for what sign-in tests share: a free port, and a visit to one."""

from __future__ import annotations

import asyncio
import socket

import pytest

from sagent.providers.testing import bound_port, free_port, send, visit


def test_a_free_port_can_be_bound() -> None:
    port = free_port()
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", port))
        assert bound_port(bound) == port


@pytest.mark.asyncio
async def test_a_visit_returns_the_status_of_the_answer() -> None:
    requests: list[bytes] = []

    async def answer(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        requests.append(await reader.readline())
        writer.write(b"HTTP/1.1 418 Teapot\r\ncontent-length: 0\r\n\r\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(answer, "127.0.0.1", 0)
    port = bound_port(server.sockets[0])
    async with server:
        assert await visit(port, "/a?b=c") == 418
        assert await send(port, b"garbage\r\n") == 418
    assert requests == [b"GET /a?b=c HTTP/1.1\r\n", b"garbage\r\n"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
