"""Tests for sagent.lib.web.fetch."""

from __future__ import annotations

from email.message import Message
from typing import Any
from unittest.mock import Mock, patch

import base64
import gzip
import http.client
import io
import urllib.error
import urllib.request
import zlib

import brotli
import pytest
import zstandard

from sagent.lib.web.fetch import (
    FetchError,
    ValidatedHost,
    _backoff_delay,
    _bracket_ipv6,
    _CappedRedirectHandler,
    _decompress,
    _open_connection,
    _split_userinfo,
    fetch,
)


class TestDecompress:
    def test_gzip(self) -> None:
        data = b"hello world"
        assert _decompress(gzip.compress(data), "gzip") == data

    def test_deflate(self) -> None:
        data = b"hello world"
        assert _decompress(zlib.compress(data), "deflate") == data

    def test_brotli(self) -> None:
        data = b"hello world"
        assert _decompress(brotli.compress(data), "br") == data

    def test_zstd(self) -> None:
        data = b"hello world"
        compressed = zstandard.ZstdCompressor().compress(data)
        assert _decompress(compressed, "zstd") == data

    def test_zstd_streaming_frame_no_size(self) -> None:
        # Streaming-mode frames omit decompressed size from the header;
        # `ZstdDecompressor.decompress()` rejects them. Real servers
        # (e.g. Cloudflare) emit such frames -- we must handle them.
        data = b"hello world " * 1000
        buf = io.BytesIO()
        with zstandard.ZstdCompressor().stream_writer(buf, closefd=False) as w:
            _ = w.write(data)
        assert _decompress(buf.getvalue(), "zstd") == data

    def test_identity(self) -> None:
        assert _decompress(b"raw", "identity") == b"raw"

    def test_empty_encoding(self) -> None:
        assert _decompress(b"raw", "") == b"raw"

    def test_unknown_encoding_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Content-Encoding"):
            _decompress(b"raw", "unknown")

    def test_decompression_failure_raises(self) -> None:
        with pytest.raises(ValueError, match="Decompression failed"):
            _decompress(b"not gzip", "gzip")


class TestBackoffDelay:
    def test_exponential_growth(self) -> None:
        d0 = _backoff_delay(0, {})
        d2 = _backoff_delay(2, {})
        assert d0 < d2

    def test_capped_at_30(self) -> None:
        assert _backoff_delay(100, {}) <= 45  # 30 + 0.5*30

    def test_retry_after_header(self) -> None:
        assert _backoff_delay(0, {"retry-after": "5"}) == 5.0

    def test_retry_after_capped(self) -> None:
        assert _backoff_delay(0, {"retry-after": "999"}) == 30.0


class TestSplitUserinfo:
    def test_no_userinfo(self) -> None:
        assert _split_userinfo("https://example.com/p?q=1") == (
            "https://example.com/p?q=1",
            None,
        )

    def test_user_pass_stripped_and_encoded(self) -> None:
        url, auth = _split_userinfo("https://u:p@example.com:8443/x")
        assert url == "https://example.com:8443/x"
        assert auth == "Basic " + base64.b64encode(b"u:p").decode()

    def test_pct_decoded_credentials(self) -> None:
        url, auth = _split_userinfo("https://u%40x:p%3Aw@example.com/")
        assert url == "https://example.com/"
        assert auth == "Basic " + base64.b64encode(b"u@x:p:w").decode()

    def test_user_only(self) -> None:
        url, auth = _split_userinfo("https://u@example.com/")
        assert url == "https://example.com/"
        assert auth == "Basic " + base64.b64encode(b"u:").decode()


class TestFetchError:
    def test_attributes(self) -> None:
        err = FetchError(
            "https://x.com",
            404,
            {"content-type": "text/html"},
            b"nope",
        )
        assert err.url == "https://x.com"
        assert err.status == 404
        assert err.headers == {"content-type": "text/html"}
        assert err.body == b"nope"
        assert "404" in str(err)


class TestFetchSimplePath:
    def _mock_urlopen(
        self,
        body: bytes = b"hello",
        encoding: str = "identity",
    ) -> Any:
        response = Mock()
        response.read.return_value = body
        response.headers.get.return_value = encoding
        return patch(
            "sagent.lib.web.fetch._open_request",
            return_value=response,
        )

    def test_basic_get(self) -> None:
        with (
            patch("sagent.lib.web.fetch.urllib.request.Request") as mock_req,
            self._mock_urlopen(),
        ):
            result = fetch("https://example.com")
        assert result == b"hello"
        mock_req.assert_called_once()
        assert mock_req.call_args.kwargs["method"] == "GET"

    def test_gzip_decompression(self) -> None:
        compressed = gzip.compress(b"hello")
        with (
            patch("sagent.lib.web.fetch.urllib.request.Request"),
            self._mock_urlopen(body=compressed, encoding="gzip"),
        ):
            assert fetch("https://example.com") == b"hello"

    def test_post_with_data(self) -> None:
        with (
            patch("sagent.lib.web.fetch.urllib.request.Request") as mock_req,
            self._mock_urlopen(),
        ):
            fetch(
                "https://example.com",
                method="POST",
                data={"q": "test"},
            )
        assert mock_req.call_args.kwargs["method"] == "POST"
        assert mock_req.call_args.kwargs["data"] == b"q=test"

    def test_post_with_json(self) -> None:
        with (
            patch("sagent.lib.web.fetch.urllib.request.Request") as mock_req,
            self._mock_urlopen(),
        ):
            fetch(
                "https://example.com",
                method="POST",
                json={"key": "value"},
            )
        assert mock_req.call_args.kwargs["data"] == b'{"key": "value"}'
        headers = mock_req.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"

    def test_data_and_json_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            fetch(
                "https://example.com",
                data={"a": "1"},
                json={"b": 2},
            )

    def test_cookies_serialized(self) -> None:
        with (
            patch("sagent.lib.web.fetch.urllib.request.Request") as mock_req,
            self._mock_urlopen(),
        ):
            fetch("https://example.com", cookies={"a": "1", "b": "2"})
        headers = mock_req.call_args.kwargs["headers"]
        assert "a=1" in headers["Cookie"]
        assert "b=2" in headers["Cookie"]

    def test_custom_headers_override_defaults(self) -> None:
        with (
            patch("sagent.lib.web.fetch.urllib.request.Request") as mock_req,
            self._mock_urlopen(),
        ):
            fetch("https://example.com", headers={"User-Agent": "custom"})
        headers = mock_req.call_args.kwargs["headers"]
        assert headers["User-Agent"] == "custom"

    def test_raw_headers_skip_defaults(self) -> None:
        with (
            patch("sagent.lib.web.fetch.urllib.request.Request") as mock_req,
            self._mock_urlopen(),
        ):
            fetch(
                "https://example.com",
                method="POST",
                data={"q": "test"},
                headers={"User-Agent": "custom"},
                raw_headers=True,
            )
        assert mock_req.call_args.kwargs["headers"] == {"User-Agent": "custom"}

    def test_raw_headers_still_add_cookies_and_auth(self) -> None:
        with (
            patch("sagent.lib.web.fetch.urllib.request.Request") as mock_req,
            self._mock_urlopen(),
        ):
            fetch(
                "https://u:p@example.com",
                headers={"User-Agent": "custom"},
                cookies={"a": "1"},
                raw_headers=True,
            )
        headers = mock_req.call_args.kwargs["headers"]
        assert headers == {
            "User-Agent": "custom",
            "Authorization": "Basic " + base64.b64encode(b"u:p").decode(),
            "Cookie": "a=1",
        }

    def test_userinfo_url_stripped_and_basic_auth_injected(self) -> None:
        with (
            patch("sagent.lib.web.fetch.urllib.request.Request") as mock_req,
            self._mock_urlopen(),
        ):
            fetch("https://u:p@example.com:8443/x")
        sent_url = mock_req.call_args.args[0]
        assert sent_url == "https://example.com:8443/x"
        headers = mock_req.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Basic " + base64.b64encode(b"u:p").decode()

    def test_caller_authorization_wins_over_userinfo(self) -> None:
        with (
            patch("sagent.lib.web.fetch.urllib.request.Request") as mock_req,
            self._mock_urlopen(),
        ):
            fetch(
                "https://u:p@example.com/",
                headers={"Authorization": "Bearer xyz"},
            )
        headers = mock_req.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer xyz"

    def test_http_error_raises_fetch_error(self) -> None:
        err = urllib.error.HTTPError(
            "https://example.com",
            403,
            "Forbidden",
            Message(),
            None,
        )
        with (
            patch("sagent.lib.web.fetch.urllib.request.Request"),
            patch(
                "sagent.lib.web.fetch._open_request",
                side_effect=err,
            ),
            pytest.raises(FetchError, match="403"),
        ):
            fetch("https://example.com")

    def test_timeout_passed(self) -> None:
        with (
            patch("sagent.lib.web.fetch.urllib.request.Request"),
            self._mock_urlopen() as mock_open,
        ):
            fetch("https://example.com", timeout_sec=60)
        assert mock_open.call_args.kwargs["timeout_sec"] == 60

    def test_simple_path_honors_redirect_cap(self) -> None:
        # INF-022: the urllib simple path must enforce the caller's
        # ``max_redirects`` cap, not urllib's own default (~10). With the
        # default sentinel and no other connection-path trigger, the simple
        # path is taken; exceeding the cap must raise FetchError.
        handler = _CappedRedirectHandler(2)
        req = urllib.request.Request("https://example.com")
        for _ in range(2):
            redirected = handler.redirect_request(
                req,
                Mock(),
                302,
                "Found",
                http.client.HTTPMessage(),
                "https://example.com/next",
            )
            assert redirected is not None
            req = redirected
        with pytest.raises(FetchError) as exc_info:
            handler.redirect_request(
                req,
                Mock(),
                302,
                "Found",
                http.client.HTTPMessage(),
                "https://example.com/last",
            )
        assert exc_info.value.body == b"Exceeded redirect limit"


class TestFetchRetry:
    def test_retries_on_500(self) -> None:
        err = urllib.error.HTTPError(
            "https://example.com",
            500,
            "ISE",
            Message(),
            None,
        )
        ok_response = Mock()
        ok_response.read.return_value = b"ok"
        ok_response.headers.get.return_value = "identity"

        with (
            patch("sagent.lib.web.fetch.urllib.request.Request"),
            patch(
                "sagent.lib.web.fetch._open_request",
                side_effect=[err, ok_response],
            ),
            patch("sagent.lib.web.fetch.time.sleep"),
        ):
            assert fetch("https://example.com", retries=1) == b"ok"

    def test_no_retry_on_404(self) -> None:
        err = urllib.error.HTTPError(
            "https://example.com",
            404,
            "NF",
            Message(),
            None,
        )
        with (
            patch("sagent.lib.web.fetch.urllib.request.Request"),
            patch(
                "sagent.lib.web.fetch._open_request",
                side_effect=err,
            ),
            pytest.raises(FetchError, match="404"),
        ):
            fetch("https://example.com", retries=3)

    def test_retries_on_network_error(self) -> None:
        ok_response = Mock()
        ok_response.read.return_value = b"ok"
        ok_response.headers.get.return_value = "identity"

        with (
            patch("sagent.lib.web.fetch.urllib.request.Request"),
            patch(
                "sagent.lib.web.fetch._open_request",
                side_effect=[OSError("refused"), ok_response],
            ),
            patch("sagent.lib.web.fetch.time.sleep"),
        ):
            assert fetch("https://example.com", retries=1) == b"ok"


class TestFetchConnectionPath:
    def _mock_http_response(
        self,
        status: int = 200,
        body: bytes = b"hello",
        headers: list[tuple[str, str]] | None = None,
    ) -> Mock:
        resp = Mock(spec=http.client.HTTPResponse)
        resp.status = status
        resp.read.return_value = body
        resp.getheaders.return_value = headers or [
            ("content-encoding", "identity"),
        ]
        return resp

    def test_return_connection(self) -> None:
        resp = self._mock_http_response()
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with patch(
            "sagent.lib.web.fetch._open_connection",
            return_value=mock_conn,
        ):
            body, http_conn = fetch(
                "https://example.com/page",
                return_connection=True,
            )
        assert body == b"hello"
        assert http_conn is mock_conn

    def test_reuse_existing_connection(self) -> None:
        resp = self._mock_http_response()
        mock_raw = Mock(spec=http.client.HTTPSConnection)
        mock_raw.host = "example.com"
        mock_raw.port = 443
        mock_raw.request = Mock()
        mock_raw.getresponse.return_value = resp

        with patch("sagent.lib.web.fetch._open_connection") as mock_open:
            body, http_conn = fetch(
                "https://example.com/next",
                return_connection=True,
                http_conn=mock_raw,
            )
        mock_open.assert_not_called()
        assert body == b"hello"
        http_conn.close()

    def test_redirect_followed(self) -> None:
        redir_resp = self._mock_http_response(
            status=302,
            body=b"",
            headers=[("location", "https://example.com/final")],
        )
        ok_resp = self._mock_http_response(body=b"final")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = [redir_resp, ok_resp]

        with patch(
            "sagent.lib.web.fetch._open_connection",
            return_value=mock_conn,
        ):
            body, http_conn = fetch(
                "https://example.com/start",
                return_connection=True,
            )
        assert body == b"final"
        http_conn.close()

    def test_on_redirect_called(self) -> None:
        redir_resp = self._mock_http_response(
            status=302,
            body=b"",
            headers=[("location", "https://example.com/final")],
        )
        ok_resp = self._mock_http_response(body=b"done")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = [redir_resp, ok_resp]

        urls: list[str] = []
        with patch(
            "sagent.lib.web.fetch._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://example.com/start",
                on_redirect=urls.append,
            )
        assert urls == ["https://example.com/final"]

    def test_on_redirect_raise_aborts(self) -> None:
        redir_resp = self._mock_http_response(
            status=302,
            body=b"",
            headers=[("location", "https://bad.com/sorry")],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = redir_resp

        def reject(url: str) -> None:
            raise ValueError(f"bad redirect: {url}")

        with (
            patch(
                "sagent.lib.web.fetch._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(ValueError, match="bad redirect"),
        ):
            fetch("https://example.com", on_redirect=reject)

    def test_max_redirects_zero_returns_3xx_body(self) -> None:
        resp = self._mock_http_response(
            status=302,
            body=b"redirect body",
            headers=[
                ("content-encoding", "identity"),
                ("location", "https://example.com/other"),
            ],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with patch(
            "sagent.lib.web.fetch._open_connection",
            return_value=mock_conn,
        ):
            result = fetch("https://example.com", max_redirects=0)
        assert result == b"redirect body"

    def test_cross_host_redirect(self) -> None:
        redir_resp = self._mock_http_response(
            status=301,
            body=b"",
            headers=[("location", "https://other.com/page")],
        )
        ok_resp = self._mock_http_response(body=b"other")
        mock_conn1 = Mock()
        mock_conn1.request = Mock()
        mock_conn1.getresponse.return_value = redir_resp
        mock_conn1.close = Mock()

        mock_conn2 = Mock()
        mock_conn2.request = Mock()
        mock_conn2.getresponse.return_value = ok_resp

        with patch(
            "sagent.lib.web.fetch._open_connection",
            side_effect=[mock_conn1, mock_conn2],
        ):
            body, http_conn = fetch(
                "https://example.com/start",
                return_connection=True,
            )
        assert body == b"other"
        assert http_conn is mock_conn2
        mock_conn1.close.assert_called_once()

    def test_303_converts_post_to_get(self) -> None:
        redir_resp = self._mock_http_response(
            status=303,
            body=b"",
            headers=[("location", "https://example.com/result")],
        )
        ok_resp = self._mock_http_response(body=b"got it")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = [redir_resp, ok_resp]

        with patch(
            "sagent.lib.web.fetch._open_connection",
            return_value=mock_conn,
        ):
            body, http_conn = fetch(
                "https://example.com/submit",
                method="POST",
                data={"x": "1"},
                return_connection=True,
            )
        assert body == b"got it"
        second_call = mock_conn.request.call_args_list[1]
        assert second_call.args[0] == "GET"
        assert second_call.kwargs.get("body") is None
        http_conn.close()

    def test_redirect_no_location_raises(self) -> None:
        resp = self._mock_http_response(
            status=302,
            body=b"",
            headers=[("content-type", "text/html")],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with (
            patch(
                "sagent.lib.web.fetch._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError, match="302"),
        ):
            fetch("https://example.com", return_connection=True)

    def test_http_error_raises_fetch_error(self) -> None:
        resp = self._mock_http_response(status=404, body=b"not found")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with (
            patch(
                "sagent.lib.web.fetch._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError, match="404"),
        ):
            fetch("https://example.com", return_connection=True)

    def test_reuse_connection_for_ported_url(self) -> None:
        # INF-001: http.client stores a bare ``.host`` (port lives in
        # ``.port``), so a ported URL must reuse a connection whose host
        # matches the URL's hostname rather than its netloc.
        resp = self._mock_http_response()
        mock_raw = Mock(spec=http.client.HTTPSConnection)
        mock_raw.host = "example.com"
        mock_raw.port = 8080
        mock_raw.request = Mock()
        mock_raw.getresponse.return_value = resp

        with patch("sagent.lib.web.fetch._open_connection") as mock_open:
            body, http_conn = fetch(
                "https://example.com:8080/next",
                return_connection=True,
                http_conn=mock_raw,
            )
        mock_open.assert_not_called()
        assert body == b"hello"
        http_conn.close()

    def test_validated_hosts_receives_hostname_not_netloc(self) -> None:
        # INF-002: the validated_hosts resolver must see the bare hostname,
        # never a host:port netloc.
        resp = self._mock_http_response()
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp
        seen: list[str] = []

        def _vh(hostname: str) -> ValidatedHost:
            seen.append(hostname)
            return ValidatedHost(host=hostname, ip="93.184.216.34")

        with patch(
            "sagent.lib.web.fetch._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://example.com:8443/page",
                return_connection=True,
                validated_hosts=_vh,
            )
        assert seen == ["example.com"]

    def test_mismatched_host_opens_new_connection(self) -> None:
        resp = self._mock_http_response()
        mock_old = Mock(spec=http.client.HTTPSConnection)
        mock_old.host = "old.com"
        mock_new = Mock()
        mock_new.request = Mock()
        mock_new.getresponse.return_value = resp

        with patch(
            "sagent.lib.web.fetch._open_connection",
            return_value=mock_new,
        ):
            body, http_conn = fetch(
                "https://new.com/page",
                return_connection=True,
                http_conn=mock_old,
            )
        assert http_conn is mock_new
        assert body == b"hello"


class TestHeaderOrder:
    """Lock the canonical Chrome header order on the wire.

    http.client emits user-supplied headers in dict insertion order, so
    asserting the dict's key order asserts the wire order. ``Host`` and
    ``Content-Length`` are added by http.client itself (right after the
    request line) and are not part of the user-headers dict here.
    """

    def _capture_headers(self, **fetch_kwargs: Any) -> dict[str, str]:
        resp = Mock(spec=http.client.HTTPResponse)
        resp.status = 200
        resp.read.return_value = b"ok"
        resp.getheaders.return_value = [("content-encoding", "identity")]
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with patch(
            "sagent.lib.web.fetch._open_connection",
            return_value=mock_conn,
        ):
            fetch("https://example.com/", return_connection=True, **fetch_kwargs)
        return dict(mock_conn.request.call_args.kwargs["headers"])

    def test_get_navigation_order(self) -> None:
        headers = self._capture_headers()
        assert list(headers) == [
            "Connection",
            "sec-ch-ua",
            "sec-ch-ua-mobile",
            "sec-ch-ua-platform",
            "Upgrade-Insecure-Requests",
            "User-Agent",
            "Accept",
            "Sec-Fetch-Site",
            "Sec-Fetch-Mode",
            "Sec-Fetch-User",
            "Sec-Fetch-Dest",
            "Accept-Encoding",
            "Accept-Language",
        ]
        assert headers["Connection"] == "keep-alive"
        assert headers["Sec-Fetch-Mode"] == "navigate"
        assert "Chrome/" in headers["User-Agent"]

    def test_post_xhr_order_with_json(self) -> None:
        headers = self._capture_headers(method="POST", json={"q": "x"})
        assert list(headers) == [
            "Connection",
            "sec-ch-ua",
            "sec-ch-ua-mobile",
            "sec-ch-ua-platform",
            "User-Agent",
            "Accept",
            "Content-Type",
            "Origin",
            "Sec-Fetch-Site",
            "Sec-Fetch-Mode",
            "Sec-Fetch-Dest",
            "Accept-Encoding",
            "Accept-Language",
        ]
        assert headers["Accept"] == "*/*"
        assert headers["Content-Type"] == "application/json"
        assert headers["Sec-Fetch-Mode"] == "cors"
        assert headers["Origin"] == "https://example.com"
        assert "Upgrade-Insecure-Requests" not in headers

    def test_post_xhr_order_with_form(self) -> None:
        headers = self._capture_headers(method="POST", data={"q": "x"})
        assert headers["Content-Type"] == "application/x-www-form-urlencoded"
        # Content-Type lives between Accept and Origin.
        keys = list(headers)
        assert keys.index("Content-Type") == keys.index("Accept") + 1
        assert keys.index("Origin") == keys.index("Content-Type") + 1

    def test_post_without_body_omits_content_type(self) -> None:
        headers = self._capture_headers(method="POST")
        assert "Content-Type" not in headers

    def test_caller_override_preserves_slot(self) -> None:
        headers = self._capture_headers(headers={"User-Agent": "Custom/1.0"})
        keys = list(headers)
        assert headers["User-Agent"] == "Custom/1.0"
        # Slot is the same as the default User-Agent slot (after
        # Upgrade-Insecure-Requests, before Accept).
        assert (
            keys.index("Upgrade-Insecure-Requests")
            < keys.index("User-Agent")
            < keys.index("Accept")
        )

    def test_caller_new_header_appended(self) -> None:
        headers = self._capture_headers(headers={"X-Trace": "abc"})
        assert list(headers)[-1] == "X-Trace"

    def test_validated_hosts_puts_host_first(self) -> None:
        def _vh(netloc: str) -> ValidatedHost:
            return ValidatedHost(host=netloc, ip="93.184.216.34")

        resp = Mock(spec=http.client.HTTPResponse)
        resp.status = 200
        resp.read.return_value = b"ok"
        resp.getheaders.return_value = [("content-encoding", "identity")]
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with patch(
            "sagent.lib.web.fetch._open_connection",
            return_value=mock_conn,
        ):
            fetch("https://example.com/", validated_hosts=_vh)

        captured = dict(mock_conn.request.call_args.kwargs["headers"])
        assert next(iter(captured)) == "Host"
        assert captured["Host"] == "example.com"


class TestIPv6Bracketing:
    def test_ipv6_address_bracketed(self) -> None:
        assert _bracket_ipv6("2606:4700::6810:7c60") == "[2606:4700::6810:7c60]"

    def test_already_bracketed_unchanged(self) -> None:
        assert _bracket_ipv6("[::1]") == "[::1]"

    def test_ipv4_unchanged(self) -> None:
        assert _bracket_ipv6("93.184.216.34") == "93.184.216.34"

    def test_hostname_unchanged(self) -> None:
        assert _bracket_ipv6("example.com") == "example.com"

    def test_open_connection_brackets_ipv6_resolved_ip(self) -> None:
        captured: list[str] = []

        class _Stub:
            def __init__(
                self,
                host: str,
                *,
                port: int | None = None,
                timeout: float,
                context: Any = None,
            ) -> None:
                del port, timeout, context
                captured.append(host)

        with patch("sagent.lib.web.fetch.http.client.HTTPSConnection", _Stub):
            # No resolved_ip path uses the validated subclass; pass plain
            # path with resolved_ip to skip the SNI override.
            _open_connection(
                "https",
                "example.com",
                timeout_sec=10,
                resolved_ip="",
            )
        assert captured == ["example.com"]


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
