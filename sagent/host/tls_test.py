"""Tests for the server's certificate and the contexts that pin it."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import asyncio
import socket
import ssl
import stat

import pytest

from sagent.host.tls import (
    client_context,
    fingerprint,
    peer_fingerprint,
    server_context,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_the_first_fingerprint_makes_a_private_certificate(tmp_path: Path) -> None:
    found = fingerprint(tmp_path / "serve")
    assert found.startswith("SHA256:")
    assert len(found.removeprefix("SHA256:")) == 64
    for name in ("key.pem", "cert.pem"):
        mode = (tmp_path / "serve" / name).stat().st_mode
        assert stat.S_IMODE(mode) == 0o600


def test_the_certificate_is_kept(tmp_path: Path) -> None:
    assert fingerprint(tmp_path) == fingerprint(tmp_path)


def test_two_servers_have_different_certificates(tmp_path: Path) -> None:
    assert fingerprint(tmp_path / "a") != fingerprint(tmp_path / "b")


@pytest.mark.asyncio
async def test_a_client_reads_the_fingerprint_it_pins(tmp_path: Path) -> None:
    expected = fingerprint(tmp_path)
    said: list[bytes] = []

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        said.append(await reader.readline())
        writer.close()

    server = await asyncio.start_server(
        serve,
        "127.0.0.1",
        0,
        ssl=server_context(tmp_path),
    )
    port = cast(tuple[str, int], server.sockets[0].getsockname())[1]
    async with server:
        reader, writer = await asyncio.open_connection(
            "127.0.0.1",
            port,
            ssl=client_context(),
        )
        assert peer_fingerprint(writer) == expected
        writer.write(b"hello\n")
        await writer.drain()
        assert await reader.read() == b""
        writer.close()
    assert said == [b"hello\n"]


def test_a_client_context_checks_no_authority() -> None:
    context = client_context()
    assert context.verify_mode == ssl.CERT_NONE
    assert context.minimum_version == ssl.TLSVersion.TLSv1_3


@pytest.mark.asyncio
async def test_a_connection_without_tls_has_no_fingerprint() -> None:
    _, ours = socket.socketpair()
    _, writer = await asyncio.open_unix_connection(sock=ours)
    with pytest.raises(ValueError, match="showed no certificate"):
        _ = peer_fingerprint(writer)
    writer.close()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
