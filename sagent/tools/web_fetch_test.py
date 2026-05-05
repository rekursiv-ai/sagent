"""Tests for WebFetch tool."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from unittest.mock import patch

import importlib
import json
import socket

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, JSONValue, json_freeze
from sagent.lib.web.fetch import FetchError, HTTPConn
from sagent.tools.web_fetch import WebFetch

import sagent.tools.web_fetch as wfm


fetch_mod = importlib.import_module("sagent.lib.web.fetch")
webfetch = WebFetch()


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-webfetch"),),
        "multipart/x-tool-call",
    )


class _FetchRouter:
    """Mock fetch() that returns bytes by URL pattern."""

    def __init__(
        self,
        responses: dict[str, bytes | Exception] | None = None,
        *,
        default: bytes | None = None,
    ) -> None:
        self._responses = responses or {}
        self._default = default
        self.calls: list[str] = []

    def __call__(
        self,
        url: str,
        *,
        method: str = "GET",
        data: dict[str, str] | None = None,
        json: JSONValue = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        retries: int = 0,
        timeout_sec: float = 30,
        max_redirects: int = 10,
        on_redirect: Callable[[str], None] | None = None,
        return_connection: bool = False,
        http_conn: HTTPConn | None = None,
    ) -> bytes:
        del method, data, json, headers, cookies, retries, timeout_sec
        del max_redirects, on_redirect, return_connection, http_conn
        self.calls.append(url)
        for pattern, val in self._responses.items():
            if pattern in url:
                if isinstance(val, Exception):
                    raise val
                return val
        if self._default is not None:
            return self._default
        raise FetchError(url, 404, {}, b"not found")


@contextmanager
def _patch_fetch(
    responses: dict[str, bytes | Exception] | None = None,
    *,
    default: bytes | None = None,
) -> Generator[None]:
    router = _FetchRouter(responses, default=default)

    def _side_effect(url: str) -> bytes:
        return router(url)

    with patch.object(wfm, "_safe_fetch", side_effect=_side_effect):
        yield


class TestWebfetch:
    @pytest.fixture(autouse=True)
    def _reset(self) -> Iterator[None]:
        wfm._WEBFETCH_CACHE.clear()
        with patch.object(wfm, "_url_is_safe", return_value=None):
            yield

    @pytest.mark.anyio
    async def test_html_extraction(self) -> None:
        html = b"""<html><body>
        <nav>Menu</nav>
        <article><p>Main content here.</p></article>
        <footer>Footer</footer>
        </body></html>"""
        with _patch_fetch(default=html):
            result = await webfetch.run(
                _msg(json_freeze({"url": "https://example.com"}))
            )
        assert isinstance(result, TextMessage)
        assert len(result.content) > 0

    @pytest.mark.anyio
    async def test_plain_text_passthrough(self) -> None:
        with _patch_fetch(default=b"Plain text content"):
            result = await webfetch.run(
                _msg(json_freeze({"url": "https://example.com/file.txt"}))
            )
        assert isinstance(result, TextMessage)
        assert "Plain text content" in result.content

    @pytest.mark.anyio
    async def test_empty_response(self) -> None:
        with _patch_fetch(default=b""):
            result = await webfetch.run(
                _msg(json_freeze({"url": "https://example.com/empty"}))
            )
        assert isinstance(result, TextMessage)

    @pytest.mark.anyio
    async def test_http_error(self) -> None:
        err = FetchError("https://example.com/404", 404, {}, b"not found")
        with _patch_fetch({"example.com": err}):
            r = await webfetch.run(
                _msg(json_freeze({"url": "https://example.com/404"}))
            )
        assert r.descriptor == "text/x-error"
        assert "404" in str(r.content)

    @pytest.mark.anyio
    async def test_html_fallback_when_extraction_empty(self) -> None:
        html = b"<html><body><div>Raw content</div></body></html>"
        with (
            _patch_fetch(default=html),
            patch("trafilatura.extract", return_value=None),
        ):
            result = await webfetch.run(
                _msg(json_freeze({"url": "https://example.com"}))
            )
        assert isinstance(result, TextMessage)
        assert "Raw content" in result.content


class TestRedditJson:
    @pytest.fixture(autouse=True)
    def _reset(self) -> Iterator[None]:
        wfm._WEBFETCH_CACHE.clear()
        with patch.object(wfm, "_url_is_safe", return_value=None):
            yield

    @pytest.mark.anyio
    async def test_reddit_thread_uses_json_endpoint(self) -> None:
        reddit_json = [
            {
                "kind": "Listing",
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "title": "Test Post",
                                "selftext": "Post body here.",
                                "author": "testuser",
                                "score": 42,
                            },
                        }
                    ]
                },
            },
            {
                "kind": "Listing",
                "data": {
                    "children": [
                        {
                            "kind": "t1",
                            "data": {
                                "author": "commenter1",
                                "score": 10,
                                "body": "Great post!",
                                "replies": "",
                            },
                        }
                    ]
                },
            },
        ]
        with _patch_fetch(default=json.dumps(reddit_json).encode()):
            result = await webfetch.run(
                _msg(
                    json_freeze(
                        {
                            "url": "https://www.reddit.com/r/Masks4All/comments/1jibbc7/which_one_to_use_in_a_riot/"
                        }
                    )
                )
            )
        assert isinstance(result, TextMessage)
        assert "Test Post" in result.content
        assert "Post body here" in result.content
        assert "commenter1" in result.content
        assert "Great post!" in result.content

    @pytest.mark.anyio
    async def test_non_reddit_url_unchanged(self) -> None:
        html = b"<html><body><article><p>Article text.</p></article></body></html>"
        with _patch_fetch(default=html):
            result = await webfetch.run(
                _msg(json_freeze({"url": "https://example.com/article"}))
            )
        assert isinstance(result, TextMessage)
        assert len(result.content) > 0


class TestSSRFGuard:
    def test_rejects_unsupported_scheme(self) -> None:
        assert wfm._url_is_safe("file:///etc/passwd") is not None
        assert wfm._url_is_safe("ftp://host/x") is not None

    def test_rejects_missing_host(self) -> None:
        assert wfm._url_is_safe("http:///path") is not None

    def test_rejects_loopback(self) -> None:
        err = wfm._url_is_safe("http://127.0.0.1:8080/")
        assert err is not None
        assert "non-public" in err

    def test_rejects_cloud_metadata(self) -> None:
        err = wfm._url_is_safe("http://169.254.169.254/latest/meta-data/")
        assert err is not None
        assert "non-public" in err

    def test_rejects_rfc1918(self) -> None:
        assert wfm._url_is_safe("http://192.168.1.1/") is not None
        assert wfm._url_is_safe("http://10.0.0.1/") is not None

    @pytest.mark.anyio
    async def test_blocks_fetch_to_private(self) -> None:
        result = await webfetch.run(_msg(json_freeze({"url": "http://127.0.0.1:1/"})))
        assert result.descriptor == "text/x-error"
        assert isinstance(result, TextMessage)
        assert "non-public" in result.content

    @pytest.mark.anyio
    async def test_ssrf_callback_blocks_redirect(self) -> None:
        """_check_ssrf raises on private IPs, which on_redirect propagates."""
        with pytest.raises(ValueError, match="non-public"):
            wfm._check_ssrf("http://127.0.0.1:1/")

    def test_safe_fetch_connects_to_validated_ip(self) -> None:
        class FakeConnection:
            def __init__(self, host: str, timeout: float) -> None:
                self.host = host
                self.timeout = timeout

        def fake_http_connection(host: str, timeout: float) -> FakeConnection:
            return FakeConnection(host, timeout)

        with (
            patch.object(
                socket,
                "getaddrinfo",
                return_value=[(0, 0, 0, "", ("93.184.216.34", 0))],
            ),
            patch.object(
                fetch_mod.http.client,
                "HTTPConnection",
                side_effect=fake_http_connection,
            ),
        ):
            conn = fetch_mod._open_connection(
                "http",
                "example.com",
                timeout_sec=15,
                resolved_ip="93.184.216.34",
            )

        assert isinstance(conn, FakeConnection)
        assert conn.host == "93.184.216.34"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
