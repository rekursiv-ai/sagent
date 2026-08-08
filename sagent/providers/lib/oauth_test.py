"""Tests for ``providers.lib.oauth``: PKCE, account routing, code parsing."""

from __future__ import annotations

from pathlib import Path

import asyncio
import base64
import contextlib
import fcntl
import hashlib
import inspect
import os
import urllib.error
import urllib.request

import pytest

from sagent.providers.lib.oauth import (
    AuthCodeListener,
    _path_lock_for,
    credential_file_lock,
    credentials_path,
    parse_manual_auth_code,
    pkce_pair,
    resolve_account,
)


def test_resolve_account_none_returns_default() -> None:
    assert resolve_account(None) == "default"


def test_resolve_account_valid_passthrough() -> None:
    assert resolve_account("work") == "work"
    assert resolve_account("a-b_c0") == "a-b_c0"


@pytest.mark.parametrize("name", ["../escape", "a/b", ".hidden", "-leading"])
def test_resolve_account_rejects_invalid(name: str) -> None:
    with pytest.raises(ValueError, match="Invalid account name"):
        resolve_account(name)


def test_resolve_account_empty_string_maps_to_default() -> None:
    # ``account or _DEFAULT_ACCOUNT`` short-circuits empty strings.
    assert resolve_account("") == "default"


def test_credentials_path_default_returns_unchanged(tmp_path: Path) -> None:
    base = tmp_path / "oauth_creds.json"
    assert credentials_path(base, None) == base
    assert credentials_path(base, "default") == base


def test_credentials_path_named_inserts_suffix(tmp_path: Path) -> None:
    base = tmp_path / "oauth_creds.json"
    assert credentials_path(base, "work") == tmp_path / "oauth_creds-work.json"


def test_credentials_path_no_suffix(tmp_path: Path) -> None:
    base = tmp_path / "creds"
    assert credentials_path(base, "alt") == tmp_path / "creds-alt"


def test_credentials_path_invalid_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid account name"):
        credentials_path(tmp_path / "x.json", "../bad")


def test_pkce_pair_challenge_matches_verifier_sha256() -> None:
    verifier, challenge = pkce_pair()
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    assert challenge == expected


def test_pkce_pair_distinct_invocations_differ() -> None:
    a_v, a_c = pkce_pair()
    b_v, b_c = pkce_pair()
    assert a_v != b_v
    assert a_c != b_c


def test_parse_manual_auth_code_url() -> None:
    url = "https://example.test/cb?code=abc123&state=stateXY"
    assert parse_manual_auth_code(url, "stateXY") == "abc123"


def test_parse_manual_auth_code_url_state_mismatch() -> None:
    url = "https://example.test/cb?code=abc&state=other"
    with pytest.raises(ValueError, match="State mismatch"):
        parse_manual_auth_code(url, "expected")


def test_parse_manual_auth_code_raw_code_with_state() -> None:
    assert parse_manual_auth_code("abc#stateXY", "stateXY") == "abc"


def test_parse_manual_auth_code_raw_code_no_state_raises() -> None:
    with pytest.raises(ValueError, match="State missing"):
        parse_manual_auth_code("abc", "ignored")


def test_parse_manual_auth_code_documents_accepted_manual_inputs() -> None:
    doc = inspect.getdoc(parse_manual_auth_code)
    assert doc is not None
    assert "Accepts ``code#state`` or a full" in doc


def test_auth_code_listener_start_twice_raises() -> None:
    listener = AuthCodeListener("state", port=0)
    listener.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            listener.start()
    finally:
        listener.stop()


def test_parse_manual_auth_code_empty_raises() -> None:
    with pytest.raises(ValueError, match="Authorization code not found"):
        parse_manual_auth_code("", "state")


def test_parse_manual_auth_code_url_missing_code() -> None:
    with pytest.raises(ValueError, match="Authorization code not found"):
        parse_manual_auth_code("https://example.test/cb?state=stateXY", "stateXY")


def test_auth_code_listener_wait_timeout_raises() -> None:
    listener = AuthCodeListener("state", port=0)
    listener.start()
    try:
        with pytest.raises(TimeoutError):
            listener.wait(timeout_sec=0.01)
    finally:
        listener.stop()


def test_auth_code_listener_redirect_uri_assigned_port() -> None:
    listener = AuthCodeListener("state", port=0, callback_path="/cb")
    listener.start()
    try:
        uri = listener.redirect_uri
        assert uri.startswith("http://127.0.0.1:")
        assert uri.endswith("/cb")
    finally:
        listener.stop()


def test_auth_code_listener_redirect_uri_for_host() -> None:
    listener = AuthCodeListener("state", port=0, callback_path="/cb")
    listener.start()
    try:
        uri = listener.redirect_uri_for_host("localhost")
        assert uri.startswith("http://localhost:")
        assert uri.endswith("/cb")
        # Port matches the IPv4 bind even though host is advertised differently.
        port = listener.redirect_uri.split(":")[2].split("/")[0]
        assert f":{port}/cb" in uri
    finally:
        listener.stop()


def test_auth_code_listener_stop_idempotent() -> None:
    listener = AuthCodeListener("state", port=0)
    listener.start()
    listener.stop()
    listener.stop()  # Must not raise on a second invocation.


@pytest.mark.asyncio
async def test_credential_file_lock_serializes_in_process(tmp_path: Path) -> None:
    """Two coroutines for the same cred path see each other inside the lock."""
    cred = tmp_path / "creds.json"
    order: list[str] = []

    async def hold(name: str, delay: float) -> None:
        async with credential_file_lock(cred):
            order.append(f"{name}-enter")
            await asyncio.sleep(delay)
            order.append(f"{name}-exit")

    await asyncio.gather(hold("a", 0.05), hold("b", 0.01))
    assert order in (
        ["a-enter", "a-exit", "b-enter", "b-exit"],
        ["b-enter", "b-exit", "a-enter", "a-exit"],
    )


@pytest.mark.asyncio
async def test_credential_file_lock_creates_sidecar(tmp_path: Path) -> None:
    """The lock targets ``<cred>.lock`` so atomic-rename of cred can't break it."""
    cred = tmp_path / "creds.json"
    async with credential_file_lock(cred):
        assert cred.with_suffix(".json.lock").exists()


@pytest.mark.asyncio
async def test_credential_file_lock_blocks_on_external_holder(
    tmp_path: Path,
) -> None:
    """A second process holding the file lock blocks until release."""
    cred = tmp_path / "creds.json"
    lock_path = cred.with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        released = asyncio.Event()

        async def waiter() -> None:
            async with credential_file_lock(cred):
                released.set()

        task = asyncio.create_task(waiter())
        # Give the waiter time to attempt-and-block on the file lock.
        await asyncio.sleep(0.05)
        assert not released.is_set()
        fcntl.flock(fd, fcntl.LOCK_UN)
        await asyncio.wait_for(released.wait(), timeout=1.0)
        await task
    finally:
        os.close(fd)


@pytest.mark.asyncio
async def test_cancelled_acquire_leaves_no_orphaned_lock(tmp_path: Path) -> None:
    """Cancelling a blocked acquire must not strand the ``flock``.

    Needs a REAL suspension: without one the waiter is cancelled before it
    ever reaches the blocking ``flock``, so the test passes against the
    defect. Nothing fakes ``asyncio.sleep`` any more, so the duration below
    is load-bearing -- do not replace it with a bare yield.

    ``asyncio.to_thread`` cannot cancel the worker it dispatched: the
    thread goes on to acquire, while the cancelled coroutine never
    reaches the release in its ``finally``. The lock is then held by
    nobody and no process can ever take it again.
    """
    cred = tmp_path / "creds.json"
    lock_path = cred.with_suffix(".json.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    blocker = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(blocker, fcntl.LOCK_EX)

        async def waiter() -> None:
            async with credential_file_lock(cred):
                pass

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)  # let it reach the blocking flock
        _ = task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        fcntl.flock(blocker, fcntl.LOCK_UN)
        await asyncio.sleep(0.05)  # let any orphan thread take it
    finally:
        os.close(blocker)
    probe = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(probe)


def test_path_lock_identity_survives_sidecar_creation(tmp_path: Path) -> None:
    """One credential file must map to one ``_PathLock``, always.

    The key resolves only when the path already exists, but the first
    acquisition is what creates it -- so acquisition two can land on a
    different lock, and the in-process half stops excluding anything.
    """
    lock_path = tmp_path / "link" / "creds.json.lock"
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    first = _path_lock_for(lock_path)
    _ = first.open_fd()
    assert _path_lock_for(lock_path) is first


def test_resolve_account_rejects_a_trailing_newline() -> None:
    r"""``$`` matches before a final newline; the grammar admits no newline.

    An accepted ``"work\n"`` reaches ``credentials_path`` and yields a
    credential filename containing a line break.
    """
    with pytest.raises(ValueError, match="Invalid account name"):
        _ = resolve_account("work\n")


def test_callback_without_a_code_is_an_error() -> None:
    """A state-matching callback carrying no ``code`` is not a success.

    ``parse_manual_auth_code`` rejects the same input; the HTTP path
    reports "Authentication complete" and hands back an empty code.
    """
    listener = AuthCodeListener("expected")
    listener.start()
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(  # noqa: S310 -- fixed http:// localhost callback
            f"{listener.redirect_uri}?state=expected",
            timeout=5,
        ).close()
    assert raised.value.code == 400
    raised.value.close()
    with pytest.raises(RuntimeError, match="code"):
        _ = listener.wait(1.0)
    listener.stop()


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
