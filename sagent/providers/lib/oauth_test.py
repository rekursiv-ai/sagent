"""Tests for lib.oauth."""

from __future__ import annotations

from pathlib import Path

import base64
import hashlib
import urllib.error
import urllib.parse
import urllib.request

import pytest

from sagent.providers.lib.oauth import (
    AuthCodeListener,
    credentials_path,
    parse_manual_auth_code,
    pkce_pair,
    resolve_account,
)


class TestPkcePair:
    def test_returns_two_strings(self) -> None:
        verifier, challenge = pkce_pair()
        assert isinstance(verifier, str)
        assert isinstance(challenge, str)

    def test_verifier_length(self) -> None:
        verifier, _ = pkce_pair()
        assert len(verifier) >= 43

    def test_challenge_matches_verifier(self) -> None:
        verifier, challenge = pkce_pair()
        digest = hashlib.sha256(verifier.encode()).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        assert challenge == expected

    def test_unique_per_call(self) -> None:
        a = pkce_pair()
        b = pkce_pair()
        assert a[0] != b[0]
        assert a[1] != b[1]


class TestAuthCodeListener:
    def test_happy_path(self) -> None:
        state = "test-state-123"
        listener = AuthCodeListener(state, port=0)
        listener.start()
        try:
            uri = listener.redirect_uri
            url = f"{uri}?code=AUTH_CODE&state={state}"
            _get(url)
            code = listener.wait(timeout_sec=2.0)
            assert code == "AUTH_CODE"
        finally:
            listener.stop()

    def test_state_mismatch(self) -> None:
        listener = AuthCodeListener("expected", port=0)
        listener.start()
        try:
            uri = listener.redirect_uri
            with pytest.raises(urllib.error.HTTPError):
                _get(f"{uri}?code=x&state=wrong")
            with pytest.raises(RuntimeError, match="state mismatch"):
                listener.wait(timeout_sec=2.0)
        finally:
            listener.stop()

    def test_error_param(self) -> None:
        listener = AuthCodeListener("s", port=0)
        listener.start()
        try:
            uri = listener.redirect_uri
            with pytest.raises(urllib.error.HTTPError):
                _get(f"{uri}?error=access_denied")
            with pytest.raises(RuntimeError, match="access_denied"):
                listener.wait(timeout_sec=2.0)
        finally:
            listener.stop()

    def test_wrong_path_404(self) -> None:
        listener = AuthCodeListener("s", port=0)
        listener.start()
        try:
            assert listener._server is not None
            port = listener._server.server_address[1]
            url = f"http://127.0.0.1:{port}/wrong?code=x&state=s"
            with pytest.raises(urllib.error.HTTPError, match="404"):
                _get(url)
        finally:
            listener.stop()

    def test_timeout(self) -> None:
        listener = AuthCodeListener("s", port=0)
        listener.start()
        try:
            with pytest.raises(TimeoutError):
                listener.wait(timeout_sec=0.05)
        finally:
            listener.stop()

    def test_stop_is_idempotent(self) -> None:
        listener = AuthCodeListener("s", port=0)
        listener.start()
        listener.stop()
        listener.stop()

    def test_custom_callback_path(self) -> None:
        listener = AuthCodeListener("s", port=0, callback_path="/oauth/cb")
        listener.start()
        try:
            assert "/oauth/cb" in listener.redirect_uri
            _get(f"{listener.redirect_uri}?code=OK&state=s")
            assert listener.wait(timeout_sec=2.0) == "OK"
        finally:
            listener.stop()


class TestParseManualAuthCode:
    def test_raw_code(self) -> None:
        assert parse_manual_auth_code(" AUTH_CODE ", "state") == "AUTH_CODE"

    def test_code_state_fragment(self) -> None:
        assert parse_manual_auth_code("AUTH_CODE#state", "state") == "AUTH_CODE"

    def test_full_redirect_url(self) -> None:
        url = "http://127.0.0.1:1455/auth/callback?code=AUTH_CODE&state=state"
        assert parse_manual_auth_code(url, "state") == "AUTH_CODE"

    def test_state_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="State mismatch"):
            parse_manual_auth_code("AUTH_CODE#wrong", "state")

    def test_missing_code_rejected(self) -> None:
        with pytest.raises(ValueError, match="Authorization code not found"):
            parse_manual_auth_code("https://example.test/callback?state=state", "state")


def _get(url: str) -> bytes:
    return urllib.request.urlopen(url, timeout=2).read()  # noqa: S310 -- test hits local server


class TestResolveAccount:
    def test_none_means_default(self) -> None:
        assert resolve_account(None) == "default"

    def test_empty_string_means_default(self) -> None:
        # Convention from cli.py: ``--account ""`` should behave like
        # omitting the flag entirely.
        assert resolve_account("") == "default"

    def test_explicit_default(self) -> None:
        assert resolve_account("default") == "default"

    def test_named_passes_through(self) -> None:
        assert resolve_account("work") == "work"
        assert resolve_account("personal_2") == "personal_2"
        assert resolve_account("a-b-c") == "a-b-c"

    @pytest.mark.parametrize(
        "bad",
        ["..", "work/evil", "a b", "/abs", ".", "_leading_underscore", "-leading"],
    )
    def test_rejects_path_traversal(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Invalid account name"):
            resolve_account(bad)


class TestCredentialsPath:
    def _base(self, tmp_path: Path) -> Path:
        return tmp_path / ".gemini" / "oauth_creds.json"

    def test_default_account_returns_base_unchanged(self, tmp_path: Path) -> None:
        base = self._base(tmp_path)
        assert credentials_path(base, None) == base
        assert credentials_path(base, "default") == base

    def test_named_inserts_suffix_before_extension(self, tmp_path: Path) -> None:
        base = self._base(tmp_path)
        p = credentials_path(base, "work")
        assert p == base.parent / "oauth_creds-work.json"

    def test_named_keeps_parent(self, tmp_path: Path) -> None:
        base = self._base(tmp_path)
        assert credentials_path(base, "x").parent == base.parent

    def test_dotfile_base(self, tmp_path: Path) -> None:
        # A hypothetical provider rooted at ``~/.codex/.auth.json`` -- the
        # filename starts with a dot, so ``Path.stem`` returns ".auth" and
        # the named variant becomes ``.auth-work.json``.
        base = tmp_path / ".codex" / ".auth.json"
        assert credentials_path(base, "work") == base.parent / ".auth-work.json"

    def test_invalid_account_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Invalid account name"):
            credentials_path(self._base(tmp_path), "../escape")


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
