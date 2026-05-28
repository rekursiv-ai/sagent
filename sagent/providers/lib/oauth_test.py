"""Tests for ``providers.lib.oauth``: PKCE, account routing, code parsing."""

from __future__ import annotations

from pathlib import Path

import asyncio
import base64
import fcntl
import hashlib
import os

import pytest

from sagent.providers.lib.oauth import (
    AuthCodeListener,
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


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
