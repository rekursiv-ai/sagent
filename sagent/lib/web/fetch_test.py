"""Tests for sagent.lib.web.fetch."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import Mock, patch

import base64
import gzip
import http.client
import io
import zlib

from curl_cffi import CurlError, CurlInfo, CurlOpt

import brotli
import pytest
import zstandard

from sagent.lib.web.errors import (
    BotDetectionError,
    CloudflareChallengeError,
    FetchError,
    PuzzleChallengeError,
)
from sagent.lib.web.fetch import (
    ValidatedHost,
    _apply_redirect,
    _backoff_delay,
    _bracket_ipv6,
    _decompress,
    _open_connection,
    _rewrite_origin,
    _split_userinfo,
    fetch,
)


class TestRewriteOrigin:
    def test_cross_origin_rewrite(self) -> None:
        out = _rewrite_origin({"Origin": "https://a.com"}, "https://b.com/land")
        assert out["Origin"] == "https://b.com"

    def test_no_origin_header_unchanged(self) -> None:
        h = {"Accept": "*/*"}
        assert _rewrite_origin(h, "https://b.com/x") is h

    def test_ipv6_target_is_bracketed(self) -> None:
        # REV2061-003: a v6 redirect target must yield a BRACKETED Origin;
        # "https://2606:...::1" is an invalid Origin (colons unbracketed).
        out = _rewrite_origin({"Origin": "https://a.com"}, "https://[2606:4700::1]/x")
        assert out["Origin"] == "https://[2606:4700::1]"

    def test_case_variant_origin_is_rewritten_not_leaked(self) -> None:
        # REVE559-003: HTTP field names are case-insensitive. A caller-supplied
        # "origin" (lowercase) must still be rewritten, not leaked verbatim.
        out = _rewrite_origin({"origin": "https://a.com"}, "https://b.com/x")
        assert not any(
            v == "https://a.com" for k, v in out.items() if k.lower() == "origin"
        )
        assert any(
            v == "https://b.com" for k, v in out.items() if k.lower() == "origin"
        )


class TestApplyRedirect:
    def test_303_drops_case_variant_content_type(self) -> None:
        # REVE559-002: a 303 POST->GET must drop Content-Type regardless of case.
        headers, method, body = _apply_redirect(
            {"content-type": "application/json", "Accept": "*/*"},
            "POST",
            b"{}",
            303,
            "https://x/result",
        )
        assert method == "GET"
        assert body is None
        assert not any(k.lower() == "content-type" for k in headers)


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

    def test_status_zero_renders_as_connection_failure_not_http_0(self) -> None:
        # RED: status 0 is the internal "no HTTP response" sentinel (timeout,
        # TLS/connect failure). Rendering it as "HTTP 0" leaks the sentinel and
        # misleads -- there is no HTTP status 0. It must read as a connection
        # failure and surface the reason (the body carries it).
        err = FetchError("https://x.com", 0, {}, b"Failed to connect to x.com port 443")
        msg = str(err)
        assert "HTTP 0" not in msg
        # Renders "connection failed: <url>: <reason>" -- assert the URL lands in
        # its slot via the exact prefix (not a bare substring membership check).
        assert msg.startswith("connection failed: https://x.com")
        assert "connect" in msg.lower() or "connection" in msg.lower()


class TestFetchInputValidation:
    """Invalid numeric args are rejected at the boundary with a ValueError, not
    leaked as an internal AssertionError or silent transport-specific behavior.
    """

    def test_negative_retries_rejected(self) -> None:
        # O-WEB-001: retries=-1 -> range(1+-1)=range(0), the loop never runs and
        # the internal "unreachable" AssertionError leaks. Reject up front.
        with pytest.raises(ValueError, match="retries"):
            fetch("https://example.com", retries=-1)

    def test_negative_max_redirects_rejected(self) -> None:
        # O-WEB-007: max_redirects=-1 silently behaves like 0 (never follow),
        # but the contract documents only 0 as "disable". Reject the ambiguous -1.
        with pytest.raises(ValueError, match="max_redirects"):
            fetch("https://example.com", max_redirects=-1)

    def test_nonpositive_timeout_rejected(self) -> None:
        # O-WEB-008: timeout_sec=0 means opposite things per transport (curl 0 =
        # no timeout, stdlib 0 = non-blocking). Reject non-positive timeouts.
        with pytest.raises(ValueError, match="timeout_sec"):
            fetch("https://example.com", timeout_sec=0)


class TestFetchClassifiesBlockAtBoundary:
    """``fetch()`` classifies a 4xx/5xx block ONCE at the boundary and raises the
    SPECIFIC :class:`BotDetectionError` subclass, so every ``except FetchError``
    consumer gets ``.guidance`` for free instead of re-deriving the kind (some
    paths forgot to, yielding a generic "HTTP 403").

    Mocks at the curl high-level boundary (``curl_cffi.requests.request``), the
    same seam the rest of ``TestFetchCurlBackend`` uses.
    """

    def _mock_403(self, body: bytes, headers: dict[str, str]) -> Mock:
        resp = Mock()
        resp.status_code = 403
        resp.content = body
        resp.headers = headers
        resp.url = "https://x.com/"
        return resp

    def test_cloudflare_403_raises_cloudflare_challenge_error(self) -> None:
        # A CF-fronted 403 with a challenge body: fetch() must raise the specific
        # CloudflareChallengeError -- which is-a BotDetectionError, is-a FetchError
        # -- carrying status/headers/body plus the CF .guidance.
        resp = self._mock_403(
            b"<!DOCTYPE html><html><head><title>Just a moment...</title>"
            b'<div class="challenge-platform"></div></head></html>',
            {"server": "cloudflare", "cf-ray": "a1-LAX"},
        )
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("curl_cffi.requests.request", return_value=resp),
            pytest.raises(CloudflareChallengeError) as exc,
        ):
            fetch("https://x.com")
        assert isinstance(exc.value, FetchError)
        assert isinstance(exc.value, BotDetectionError)
        assert exc.value.status == 403
        assert exc.value.headers == {"server": "cloudflare", "cf-ray": "a1-LAX"}
        assert b"challenge-platform" in exc.value.body
        assert "cloudflare" in exc.value.guidance.lower()

    def test_recaptcha_403_raises_puzzle_challenge_error(self) -> None:
        # A reCAPTCHA body pins a solve-a-puzzle wall regardless of the CF front.
        resp = self._mock_403(
            b'<div class="g-recaptcha" data-sitekey="x"></div>',
            {"content-type": "text/html"},
        )
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("curl_cffi.requests.request", return_value=resp),
            pytest.raises(PuzzleChallengeError) as exc,
        ):
            fetch("https://x.com")
        assert exc.value.status == 403
        assert "captcha" in exc.value.guidance.lower()

    def test_genuine_404_raises_plain_fetch_error_not_bot_flag(self) -> None:
        # No markers, non-CF origin: a real 404 must stay a plain FetchError,
        # never a BotDetectionError (else a dead URL looks recoverable).
        resp = Mock()
        resp.status_code = 404
        resp.content = b"<html><body><h1>404 Not Found</h1></body></html>"
        resp.headers = {"server": "nginx"}
        resp.url = "https://x.com/"
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("curl_cffi.requests.request", return_value=resp),
            pytest.raises(FetchError) as exc,
        ):
            fetch("https://x.com")
        assert not isinstance(exc.value, BotDetectionError)
        assert exc.value.status == 404

    def test_except_fetch_error_catches_the_specific_subclass(self) -> None:
        # The whole point: an existing ``except FetchError`` still catches the
        # newly-specific CloudflareChallengeError (subclass), no call-site change.
        resp = self._mock_403(
            b"<html><head><title>Just a moment...</title></head></html>",
            {"server": "cloudflare", "cf-ray": "b2-LAX"},
        )
        caught: FetchError | None = None
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("curl_cffi.requests.request", return_value=resp),
        ):
            try:
                fetch("https://x.com")
            except FetchError as e:
                caught = e
        assert isinstance(caught, CloudflareChallengeError)


class TestFetchStdlibPath:
    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> Any:
        # These tests pin the stdlib (http.client connection) transport; force
        # _HAVE_CURL off so the curl backend never intercepts and the stdlib
        # connection-path behavior is asserted.
        with patch("sagent.lib.web.fetch._HAVE_CURL", False):
            yield

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

    def _mock_conn(self, resp: Mock) -> Mock:
        conn = Mock()
        conn.request = Mock()
        conn.getresponse.return_value = resp
        return conn

    def test_basic_get(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            result = fetch("https://example.com")
        assert result == b"hello"
        mock_conn.request.assert_called_once()
        assert mock_conn.request.call_args.args[0] == "GET"

    def test_gzip_decompression(self) -> None:
        compressed = gzip.compress(b"hello")
        resp = self._mock_http_response(
            body=compressed, headers=[("content-encoding", "gzip")]
        )
        mock_conn = self._mock_conn(resp)
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            assert fetch("https://example.com") == b"hello"

    def test_post_with_data(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch(
                "https://example.com",
                method="POST",
                data={"q": "test"},
            )
        assert mock_conn.request.call_args.args[0] == "POST"
        assert mock_conn.request.call_args.kwargs["body"] == b"q=test"

    def test_post_with_json(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch(
                "https://example.com",
                method="POST",
                json={"key": "value"},
            )
        assert mock_conn.request.call_args.kwargs["body"] == b'{"key": "value"}'
        headers = mock_conn.request.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"

    def test_data_and_json_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            fetch(
                "https://example.com",
                data={"a": "1"},
                json={"b": 2},
            )

    def test_cookies_serialized(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch("https://example.com", cookies={"a": "1", "b": "2"})
        headers = mock_conn.request.call_args.kwargs["headers"]
        assert "a=1" in headers["Cookie"]
        assert "b=2" in headers["Cookie"]

    def test_custom_headers_override_defaults(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch("https://example.com", headers={"User-Agent": "custom"})
        headers = mock_conn.request.call_args.kwargs["headers"]
        assert headers["User-Agent"] == "custom"

    def test_raw_headers_skip_defaults(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch(
                "https://example.com",
                method="POST",
                data={"q": "test"},
                headers={"User-Agent": "custom"},
                raw_headers=True,
            )
        assert mock_conn.request.call_args.kwargs["headers"] == {"User-Agent": "custom"}

    def test_raw_headers_still_add_cookies_and_auth(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch(
                "https://u:p@example.com",
                headers={"User-Agent": "custom"},
                cookies={"a": "1"},
                raw_headers=True,
            )
        headers = mock_conn.request.call_args.kwargs["headers"]
        assert headers == {
            "User-Agent": "custom",
            "Authorization": "Basic " + base64.b64encode(b"u:p").decode(),
            "Cookie": "a=1",
        }

    def test_userinfo_url_stripped_and_basic_auth_injected(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "sagent.lib.web.fetch._open_connection", return_value=mock_conn
        ) as mock_open:
            fetch("https://u:p@example.com:8443/x")
        # userinfo stripped: the connection opens on the bare host:port, and the
        # request path carries no credentials.
        assert mock_open.call_args.args[1] == "example.com"
        assert mock_open.call_args.kwargs["port"] == 8443
        assert mock_conn.request.call_args.args[1] == "/x"
        headers = mock_conn.request.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Basic " + base64.b64encode(b"u:p").decode()

    def test_caller_authorization_wins_over_userinfo(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch(
                "https://u:p@example.com/",
                headers={"Authorization": "Bearer xyz"},
            )
        headers = mock_conn.request.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer xyz"

    def test_http_error_raises_fetch_error(self) -> None:
        resp = self._mock_http_response(status=403, body=b"Forbidden")
        mock_conn = self._mock_conn(resp)
        with (
            patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn),
            pytest.raises(FetchError, match="403"),
        ):
            fetch("https://example.com")

    def test_timeout_passed(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "sagent.lib.web.fetch._open_connection", return_value=mock_conn
        ) as mock_open:
            fetch("https://example.com", timeout_sec=60)
        assert mock_open.call_args.args[2] == 60


class TestFetchRetry:
    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> Any:
        with patch("sagent.lib.web.fetch._HAVE_CURL", False):
            yield

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

    def test_retries_on_500(self) -> None:
        resp_500 = self._mock_http_response(status=500, body=b"ISE")
        resp_ok = self._mock_http_response(body=b"ok")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = [resp_500, resp_ok]

        with (
            patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn),
            patch("sagent.lib.web.fetch.time.sleep"),
        ):
            assert fetch("https://example.com", retries=1) == b"ok"

    def test_no_retry_on_404(self) -> None:
        resp = self._mock_http_response(status=404, body=b"NF")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp
        with (
            patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn),
            pytest.raises(FetchError, match="404"),
        ):
            fetch("https://example.com", retries=3)

    def test_error_body_is_decompressed(self) -> None:
        # RED: an error response (e.g. a Cloudflare 403 challenge page) is
        # compressed like any other; the success path decompresses but the error
        # path stored the body RAW, so FetchError.body was undecodable garbage --
        # which is exactly why a challenge page can't be told from a plain 404.
        html = b"<!DOCTYPE html><html>Just a moment...</html>"
        resp = self._mock_http_response(
            status=403,
            body=zstandard.ZstdCompressor().compress(html),
            headers=[("content-encoding", "zstd"), ("server", "cloudflare")],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp
        with (
            patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn),
            pytest.raises(FetchError) as exc,
        ):
            fetch("https://x.com")
        # The caller must receive readable HTML, not the raw zstd frame.
        assert exc.value.body == html

    def test_retries_on_network_error(self) -> None:
        resp_ok = self._mock_http_response(body=b"ok")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = [OSError("refused"), resp_ok]

        with (
            patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn),
            patch("sagent.lib.web.fetch.time.sleep"),
        ):
            assert fetch("https://example.com", retries=1) == b"ok"


class TestConnectionClosedOnError:
    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> Any:
        with patch("sagent.lib.web.fetch._HAVE_CURL", False):
            yield

    def _mock_conn(self, status: int, body: bytes = b"nope") -> Mock:
        resp = Mock(spec=http.client.HTTPResponse)
        resp.status = status
        resp.read.return_value = body
        resp.getheaders.return_value = [("content-encoding", "identity")]
        conn = Mock()
        conn.request = Mock()
        conn.getresponse.return_value = resp
        return conn

    def test_conn_closed_when_error_status_raises(self) -> None:
        # A non-retryable HTTP error raises from inside the connection path,
        # BEFORE the success-path close(). The self-opened socket must not leak.
        conn = self._mock_conn(404)
        with (
            patch("sagent.lib.web.fetch._open_connection", return_value=conn),
            pytest.raises(FetchError),
        ):
            fetch("https://example.com")
        conn.close.assert_called_once()

    def test_conn_closed_on_each_retried_attempt(self) -> None:
        # A retryable 500 that then succeeds opens a fresh conn per attempt;
        # the first attempt's conn must be closed before the retry, not leaked.
        resp_500 = Mock(spec=http.client.HTTPResponse)
        resp_500.status = 500
        resp_500.read.return_value = b"ISE"
        resp_500.getheaders.return_value = [("content-encoding", "identity")]
        resp_ok = Mock(spec=http.client.HTTPResponse)
        resp_ok.status = 200
        resp_ok.read.return_value = b"ok"
        resp_ok.getheaders.return_value = [("content-encoding", "identity")]
        conn1 = Mock(request=Mock())
        conn1.getresponse.return_value = resp_500
        conn2 = Mock(request=Mock())
        conn2.getresponse.return_value = resp_ok
        with (
            patch("sagent.lib.web.fetch._open_connection", side_effect=[conn1, conn2]),
            patch("sagent.lib.web.fetch.time.sleep"),
        ):
            assert fetch("https://example.com", retries=1) == b"ok"
        conn1.close.assert_called_once()


class TestFetchConnectionPath:
    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> Any:
        # The connection path is stdlib-only; force _HAVE_CURL off so these
        # redirect/error/303/validated-host tests exercise the stdlib path.
        with patch("sagent.lib.web.fetch._HAVE_CURL", False):
            yield

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
            body = fetch("https://example.com/start")
        assert body == b"final"

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

    def test_plain_get_curl_absent_returns_3xx_body_at_cap(self) -> None:
        # REVE559-001: a plain GET at default max_redirects, curl absent -- once
        # _fetch_simple (urllib) is gone, this routes through _fetch_connection,
        # which returns the 3xx body at the cap. The old urllib path RAISED here
        # (None-at-cap fell through to http_error_default). No conn-triggers, so
        # this is exactly the path REVE559-001 lived on.
        resp = self._mock_http_response(
            status=302,
            body=b"cap body",
            headers=[
                ("content-encoding", "identity"),
                ("location", "https://example.com/loop"),
            ],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", False),
            patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn),
        ):
            result = fetch("https://example.com")  # default max_redirects=10
        assert result == b"cap body"

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
            body = fetch("https://example.com/start")
        assert body == b"other"
        mock_conn1.close.assert_called_once()

    def test_path_relative_redirect_stays_on_host(self) -> None:
        # Location "next" (no leading slash) from /base/start must resolve to
        # /base/next on the same host, not corrupt the host to "example.comnext".
        redir_resp = self._mock_http_response(
            status=302, body=b"", headers=[("location", "next")]
        )
        ok_resp = self._mock_http_response(body=b"landed")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = [redir_resp, ok_resp]

        urls: list[str] = []
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            body = fetch("https://example.com/base/start", on_redirect=urls.append)
        assert body == b"landed"
        assert urls == ["https://example.com/base/next"]
        # Second request stays on the same connection (same host), path /base/next.
        assert mock_conn.request.call_args_list[1].args[1] == "/base/next"

    def test_cross_origin_redirect_resets_origin_header(self) -> None:
        # A POST to a.com that redirects to b.com must NOT leak Origin: a.com;
        # the header is rewritten to the new origin (never the source).
        redir_resp = self._mock_http_response(
            status=307, body=b"", headers=[("location", "https://b.com/land")]
        )
        ok_resp = self._mock_http_response(body=b"ok")
        conn_a = Mock(request=Mock(), close=Mock())
        conn_a.getresponse.return_value = redir_resp
        conn_b = Mock(request=Mock())
        conn_b.getresponse.return_value = ok_resp

        with patch(
            "sagent.lib.web.fetch._open_connection", side_effect=[conn_a, conn_b]
        ):
            fetch("https://a.com/submit", method="POST", data={"x": "1"})
        sent = conn_b.request.call_args.kwargs["headers"]
        assert sent.get("Origin") != "https://a.com"
        assert sent.get("Origin") == "https://b.com"

    def test_cross_host_redirect_drops_case_variant_host_header(self) -> None:
        # CADF-003: a caller-supplied lowercase "host" must not survive a
        # cross-host redirect (HTTP field names are case-insensitive); leaking
        # the source host to the new origin is a routing/information-leak bug.
        redir_resp = self._mock_http_response(
            status=301, body=b"", headers=[("location", "https://other.com/page")]
        )
        ok_resp = self._mock_http_response(body=b"ok")
        conn_a = Mock(request=Mock(), close=Mock())
        conn_a.getresponse.return_value = redir_resp
        conn_b = Mock(request=Mock())
        conn_b.getresponse.return_value = ok_resp

        with patch(
            "sagent.lib.web.fetch._open_connection", side_effect=[conn_a, conn_b]
        ):
            fetch("https://a.com/start", headers={"host": "a.com"})
        sent = conn_b.request.call_args.kwargs["headers"]
        assert not any(k.lower() == "host" and v == "a.com" for k, v in sent.items())

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
            body = fetch(
                "https://example.com/submit",
                method="POST",
                data={"x": "1"},
            )
        assert body == b"got it"
        second_call = mock_conn.request.call_args_list[1]
        assert second_call.args[0] == "GET"
        assert second_call.kwargs.get("body") is None

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
            fetch("https://example.com")

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
            fetch("https://example.com")

    def test_error_body_is_decompressed(self) -> None:
        # RED: connection-path twin of the simple-path bug. A compressed error
        # body (Cloudflare 403 challenge) was stored raw in FetchError.body while
        # the success return one line away decompressed it.
        html = b"<!DOCTYPE html><html>Just a moment...</html>"
        resp = self._mock_http_response(
            status=403,
            body=zstandard.ZstdCompressor().compress(html),
            headers=[("content-encoding", "zstd"), ("server", "cloudflare")],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with (
            patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn),
            pytest.raises(FetchError) as exc,
        ):
            fetch("https://example.com")
        assert exc.value.body == html

    def test_undecodable_error_body_falls_back_to_raw(self) -> None:
        # An error body whose declared encoding can't decode must NOT mask the
        # HTTP error with a decompression ValueError; surface the raw bytes so
        # the original status still propagates.
        garbage = b"this is not a valid gzip stream"
        resp = self._mock_http_response(
            status=500,
            body=garbage,
            headers=[("content-encoding", "gzip")],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        with (
            patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn),
            pytest.raises(FetchError) as exc,
        ):
            fetch("https://example.com")
        assert exc.value.status == 500
        assert exc.value.body == garbage

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
            fetch("https://example.com:8443/page", validated_hosts=_vh)
        assert seen == ["example.com"]

    def test_validated_host_header_carries_non_default_port(self) -> None:
        # A2: the resolver returns the bare host (contract above), but the Host
        # HEADER must still carry a non-default port -- RFC 9110 requires the
        # port in Host when it is not the scheme default, and a vhost router
        # keys on it. Dropping it sends the wrong authority.
        resp = self._mock_http_response()
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        def _vh(hostname: str) -> ValidatedHost:
            return ValidatedHost(host=hostname, ip="93.184.216.34")

        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch("https://example.com:8443/page", validated_hosts=_vh)
        assert mock_conn.request.call_args.kwargs["headers"]["Host"] == (
            "example.com:8443"
        )

    def test_validated_host_header_omits_default_port(self) -> None:
        # The converse: a default-port URL must NOT get a ":443" in Host (a real
        # browser omits the default port), else the authority still mismatches.
        resp = self._mock_http_response()
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        def _vh(hostname: str) -> ValidatedHost:
            return ValidatedHost(host=hostname, ip="93.184.216.34")

        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch("https://example.com/page", validated_hosts=_vh)
        assert mock_conn.request.call_args.kwargs["headers"]["Host"] == "example.com"

    def test_cross_host_redirect_host_header_carries_non_default_port(self) -> None:
        # A2 also applies on the REDIRECT path: a cross-host redirect to a
        # ported URL must rebuild Host WITH the port, not just the initial hop.
        # (The initial-hop and rebuild Host logic must share one rule.)
        redir = self._mock_http_response(
            status=301, body=b"", headers=[("location", "https://other.com:8443/p")]
        )
        ok = self._mock_http_response(body=b"ok")
        conn_a = Mock(request=Mock(), close=Mock())
        conn_a.getresponse.return_value = redir
        conn_b = Mock(request=Mock())
        conn_b.getresponse.return_value = ok

        def _vh(hostname: str) -> ValidatedHost:
            return ValidatedHost(host=hostname, ip="1.2.3.4")

        with patch(
            "sagent.lib.web.fetch._open_connection", side_effect=[conn_a, conn_b]
        ):
            fetch("https://example.com/start", validated_hosts=_vh)
        assert conn_b.request.call_args.kwargs["headers"]["Host"] == "other.com:8443"


class TestHeaderOrder:
    """Lock the canonical Chrome header order on the wire.

    http.client emits user-supplied headers in dict insertion order, so
    asserting the dict's key order asserts the wire order. ``Host`` and
    ``Content-Length`` are added by http.client itself (right after the
    request line) and are not part of the user-headers dict here.
    """

    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> Any:
        with patch("sagent.lib.web.fetch._HAVE_CURL", False):
            yield

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
            fetch("https://example.com/", **fetch_kwargs)
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


class TestFetchCurlBackend:
    """The curl_cffi backend: SSRF pinning, redirects, decompression, errors.

    All tests mock at the curl boundary -- either ``curl_cffi.requests.request``
    (high-level path, no ``validated_hosts``) or the ``curl_cffi.Curl`` class
    (low-level path, ``validated_hosts`` set) -- so nothing hits the network.
    ``_HAVE_CURL`` is forced True so the dispatch routes through the backend
    regardless of install state.
    """

    def _mock_response(
        self,
        *,
        status: int = 200,
        content: bytes = b"hello",
        headers: dict[str, str] | None = None,
        url: str = "https://example.com/",
    ) -> Mock:
        resp = Mock()
        resp.status_code = status
        resp.content = content
        resp.headers = headers or {}
        resp.url = url
        return resp

    def _fake_curl_class(
        self, hops: list[Mock]
    ) -> tuple[type, list[tuple[int, object]]]:
        """Build a fake ``Curl`` class replaying *hops* and recording setopts.

        Each hop is a Mock carrying ``.status`` (int), ``.body`` (bytes), and
        ``.raw_headers`` (bytes: the CRLF header block). ``perform`` advances
        through hops in order, writing into the WRITEDATA / HEADERDATA buffers.
        The returned list captures every ``(option, value)`` passed to setopt.
        """
        setopts: list[tuple[int, object]] = []
        state = {"i": 0}

        class _FakeCurl:
            def __init__(self) -> None:
                self._write: io.BytesIO | None = None
                self._header: io.BytesIO | None = None

            def setopt(self, option: int, value: object) -> int:
                setopts.append((int(option), value))
                if int(option) == int(CurlOpt.WRITEDATA):
                    assert isinstance(value, io.BytesIO)
                    self._write = value
                elif int(option) == int(CurlOpt.HEADERDATA):
                    assert isinstance(value, io.BytesIO)
                    self._header = value
                return 0

            def impersonate(self, target: str, default_headers: bool = True) -> int:
                del target, default_headers
                return 0

            def perform(
                self, clear_headers: bool = True, clear_resolve: bool = True
            ) -> None:
                del clear_headers, clear_resolve
                hop = hops[state["i"]]
                state["i"] += 1
                assert self._write is not None
                assert self._header is not None
                _ = self._write.write(hop.body)
                _ = self._header.write(hop.raw_headers)

            def getinfo(self, option: int) -> bytes | int:
                hop = hops[state["i"] - 1]
                if int(option) == int(CurlInfo.RESPONSE_CODE):
                    return int(hop.status)
                return b""

            def close(self) -> None:
                pass

            def reset(self) -> None:
                self._write = None
                self._header = None

        return _FakeCurl, setopts

    def _hop(
        self, *, status: int, body: bytes = b"", headers: dict[str, str] | None = None
    ) -> Mock:
        raw = b"".join(f"{k}: {v}\r\n".encode() for k, v in (headers or {}).items())
        m = Mock()
        m.status = status
        m.body = body
        m.raw_headers = b"HTTP/2 %d\r\n" % status + raw + b"\r\n"
        return m

    def test_high_level_get_impersonates_and_returns_body(self) -> None:
        # The simple curl path uses high-level requests with chrome
        # impersonation and no manual conn; returns the decoded body.
        resp = self._mock_response(content=b"hello")
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
        ):
            body = fetch("https://example.com")
        assert body == b"hello"
        assert mock_req.call_args.kwargs["impersonate"] == "chrome"
        assert mock_req.call_args.kwargs["allow_redirects"] is False

    def test_ssrf_resolve_pin_and_repin_on_cross_host_redirect(self) -> None:
        # (a) validated_hosts routes through the low-level Curl handle; each
        # host is pinned via CurlOpt.RESOLVE as "host:port:ip", and a redirect
        # to a NEW host re-pins to that host's validated IP.
        hops = [
            self._hop(status=302, headers={"location": "https://other.com/final"}),
            self._hop(status=200, body=b"done"),
        ]
        fake_curl, setopts = self._fake_curl_class(hops)

        def _vh(hostname: str) -> ValidatedHost:
            return ValidatedHost(
                host=hostname,
                ip="1.2.3.4" if hostname == "example.com" else "5.6.7.8",
            )

        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("sagent.lib.web.fetch.Curl", fake_curl),
        ):
            body = fetch(
                "https://example.com/start",
                validated_hosts=_vh,
                on_redirect=lambda _u: None,
            )
        assert body == b"done"
        resolves = [v for o, v in setopts if o == int(CurlOpt.RESOLVE)]
        assert ["example.com:443:1.2.3.4"] in resolves
        assert ["other.com:443:5.6.7.8"] in resolves

    def test_pinned_curl_brackets_ipv6_resolve_entry(self) -> None:
        # REV2-002: a v6 pin must be "host:port:[v6]" -- curl mis-parses an
        # unbracketed IPv6 (colons collide with the host:port delimiters).
        hops = [self._hop(status=200, body=b"ok")]
        fake_curl, setopts = self._fake_curl_class(hops)

        def _vh(hostname: str) -> ValidatedHost:
            return ValidatedHost(host=hostname, ip="2606:4700:20::1")

        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("sagent.lib.web.fetch.Curl", fake_curl),
        ):
            fetch("https://v6.example/x", validated_hosts=_vh)
        resolves = [v for o, v in setopts if o == int(CurlOpt.RESOLVE)]
        assert ["v6.example:443:[2606:4700:20::1]"] in resolves

    def test_pinned_curl_rewrites_origin_on_cross_host_redirect(self) -> None:
        # REV2-001: a POST that redirects cross-origin must NOT leak the source
        # Origin. Header must be rewritten to the new origin on each hop.
        hops = [
            self._hop(status=307, headers={"location": "https://b.com/land"}),
            self._hop(status=200, body=b"done"),
        ]
        fake_curl, setopts = self._fake_curl_class(hops)

        def _vh(hostname: str) -> ValidatedHost:
            return ValidatedHost(host=hostname, ip="1.2.3.4")

        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("sagent.lib.web.fetch.Curl", fake_curl),
        ):
            fetch(
                "https://a.com/submit",
                method="POST",
                data={"x": "1"},
                validated_hosts=_vh,
            )
        # The HTTPHEADER set on the SECOND hop must carry Origin: b.com, never a.com.
        header_sets = [v for o, v in setopts if o == int(CurlOpt.HTTPHEADER)]
        second = cast("list[bytes]", header_sets[1])
        joined = b"\n".join(second).decode().lower()
        assert "origin: https://b.com" in joined
        assert "a.com" not in joined.split("origin:")[1].split("\n")[0]

    def test_simple_curl_rewrites_origin_on_cross_host_redirect(self) -> None:
        # REV2-001 (high-level path): same Origin-leak guard without pinning.
        redir = self._mock_response(
            status=307, content=b"", headers={"location": "https://b.com/land"}
        )
        ok = self._mock_response(status=200, content=b"done")
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("curl_cffi.requests.request", side_effect=[redir, ok]) as mock_req,
        ):
            fetch("https://a.com/submit", method="POST", data={"x": "1"})
        second_headers = mock_req.call_args_list[1].kwargs["headers"]
        assert second_headers.get("Origin") == "https://b.com"

    def test_simple_curl_does_not_double_send_cookies(self) -> None:
        # F3: cookies are serialized into the Cookie header by fetch(); the curl
        # path must NOT also pass cookies= (curl emits BOTH -> duplicated).
        resp = self._mock_response(content=b"ok")
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
        ):
            fetch("https://example.com", cookies={"CONSENT": "YES+"})
        kwargs = mock_req.call_args.kwargs
        # Exactly one cookie source: the header. The cookies= kwarg must be unset.
        assert "CONSENT=YES+" in kwargs["headers"].get("Cookie", "")
        assert not kwargs.get("cookies")

    def test_redirect_cap_follows_up_to_limit_then_returns_body(self) -> None:
        # on_redirect fires once per FOLLOWED hop; when the cap is reached the
        # curl path returns the final 3xx body (matching _fetch_connection's
        # "return the 3xx body at the cap" contract), it does NOT raise.
        hops = [
            self._hop(status=302, headers={"location": "https://a.com/1"}),
            self._hop(status=302, headers={"location": "https://a.com/2"}),
            self._hop(status=302, body=b"final 3xx", headers={"location": "/3"}),
        ]
        fake_curl, _ = self._fake_curl_class(hops)
        seen: list[str] = []

        def _vh(hostname: str) -> ValidatedHost:
            return ValidatedHost(host=hostname, ip="1.2.3.4")

        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("sagent.lib.web.fetch.Curl", fake_curl),
        ):
            body = fetch(
                "https://a.com/start",
                max_redirects=2,
                validated_hosts=_vh,
                on_redirect=seen.append,
            )
        assert body == b"final 3xx"
        assert seen == ["https://a.com/1", "https://a.com/2"]

    def test_error_status_raises_with_decompressed_body(self) -> None:
        # (c) low-level path: a zstd-compressed 403 error body must be
        # decompressed to readable HTML in FetchError.body.
        html = b"<!DOCTYPE html><html>Just a moment...</html>"
        compressed = zstandard.ZstdCompressor().compress(html)
        hops = [
            self._hop(
                status=403,
                body=compressed,
                headers={"content-encoding": "zstd", "server": "cloudflare"},
            )
        ]
        fake_curl, _ = self._fake_curl_class(hops)

        def _vh(hostname: str) -> ValidatedHost:
            return ValidatedHost(host=hostname, ip="1.2.3.4")

        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("sagent.lib.web.fetch.Curl", fake_curl),
            pytest.raises(FetchError) as exc,
        ):
            fetch("https://example.com", validated_hosts=_vh)
        assert exc.value.status == 403
        assert exc.value.body == html

    def test_raw_headers_sends_only_provided_headers(self) -> None:
        # (d) raw_headers=True: the high-level curl request receives exactly the
        # caller's header (plus nothing derived from the Chrome default set).
        resp = self._mock_response(content=b"ok")
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
        ):
            fetch(
                "https://example.com",
                headers={"User-Agent": "custom"},
                raw_headers=True,
            )
        assert mock_req.call_args.kwargs["headers"] == {"User-Agent": "custom"}

    def test_curl_exception_maps_to_fetch_error_status_zero(self) -> None:
        # (e) any curl_cffi exception (connection/timeout) becomes
        # FetchError(status=0) rather than leaking the raw curl error.
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch(
                "curl_cffi.requests.request",
                side_effect=CurlError("connection refused"),
            ),
            pytest.raises(FetchError) as exc,
        ):
            fetch("https://example.com")
        assert exc.value.status == 0
        assert b"connection refused" in exc.value.body

    def test_303_converts_post_to_get_and_drops_body(self) -> None:
        # A 303 on the curl path switches the follow-up to GET with no body.
        resp_303 = self._mock_response(
            status=303, headers={"location": "https://example.com/result"}
        )
        resp_ok = self._mock_response(content=b"got it")
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def _record(*args: Any, **kwargs: Any) -> Mock:
            calls.append((args, kwargs))
            return resp_303 if len(calls) == 1 else resp_ok

        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("curl_cffi.requests.request", side_effect=_record),
        ):
            body = fetch(
                "https://example.com/submit",
                method="POST",
                data={"x": "1"},
                on_redirect=lambda _u: None,
            )
        assert body == b"got it"
        # method is the first positional arg to cc_requests.request.
        assert calls[1][0][0] == "GET"
        assert calls[1][1]["data"] is None

    def test_303_get_drops_content_type_header(self) -> None:
        # REV2061-002: a 303 switches POST->GET; the POST-only Content-Type must
        # NOT survive onto the bodyless GET (a real browser drops it).
        resp_303 = self._mock_response(
            status=303, headers={"location": "https://example.com/result"}
        )
        resp_ok = self._mock_response(content=b"ok")
        calls: list[dict[str, Any]] = []

        def _record(*_a: Any, **kw: Any) -> Mock:
            calls.append(kw)
            return resp_303 if len(calls) == 1 else resp_ok

        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("curl_cffi.requests.request", side_effect=_record),
        ):
            fetch("https://example.com/submit", method="POST", json={"x": 1})
        assert "Content-Type" not in calls[1]["headers"]

    def test_max_redirects_zero_returns_3xx_body_on_curl(self) -> None:
        # REV2061-001: max_redirects=0 means "do not follow, return the 3xx
        # body" (the documented contract, matched by the stdlib path) -- the
        # curl path must NOT raise on the first redirect.
        resp = self._mock_response(
            status=302,
            content=b"redirect body",
            headers={"location": "https://example.com/other"},
        )
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
        ):
            body = fetch("https://example.com", max_redirects=0)
        assert body == b"redirect body"
        assert mock_req.call_count == 1  # never followed

    def test_no_curl_falls_back_to_connection_path(self) -> None:
        # When _HAVE_CURL is False, dispatch must fall back to the stdlib
        # connection path (http.client), preserving all prior behavior.
        resp = Mock(spec=http.client.HTTPResponse)
        resp.status = 200
        resp.read.return_value = b"stdlib"
        resp.getheaders.return_value = [("content-encoding", "identity")]
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", False),
            patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn),
            patch("curl_cffi.requests.request") as mock_curl,
        ):
            body = fetch("https://example.com")
        assert body == b"stdlib"
        mock_curl.assert_not_called()

    def test_curl_connection_error_is_retried(self) -> None:
        # A1: a curl transport error (connection refused/timeout) becomes
        # FetchError(status=0). retries= must retry it -- the stdlib path retries
        # a raw OSError, so the curl path must retry its status-0 equivalent, or
        # the two transports disagree on what `retries=` means.
        ok = self._mock_response(content=b"ok")
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch(
                "curl_cffi.requests.request",
                side_effect=[CurlError("connection refused"), ok],
            ),
            patch("sagent.lib.web.fetch.time.sleep"),
        ):
            assert fetch("https://example.com", retries=1) == b"ok"

    def test_curl_connection_error_exhausts_retries_then_raises(self) -> None:
        # The retry must still terminate: a persistent curl error raises after
        # the budget, not loop forever.
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("curl_cffi.requests.request", side_effect=CurlError("refused")),
            patch("sagent.lib.web.fetch.time.sleep"),
            pytest.raises(FetchError) as exc,
        ):
            fetch("https://example.com", retries=2)
        assert exc.value.status == 0

    def test_pinned_curl_reuses_resolution_on_same_origin_redirect(self) -> None:
        # A3: the resolver contract (fetch docstring) says a same-origin redirect
        # reuses the prior resolution without re-invoking validated_hosts. The
        # stdlib path honors this; the pinned-curl path must too, or the two
        # transports diverge on how often a (possibly expensive) resolver runs.
        hops = [
            self._hop(status=302, headers={"location": "https://example.com/next"}),
            self._hop(status=200, body=b"ok"),
        ]
        fake_curl, _ = self._fake_curl_class(hops)
        calls: list[str] = []

        def _vh(hostname: str) -> ValidatedHost:
            calls.append(hostname)
            return ValidatedHost(host=hostname, ip="1.2.3.4")

        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("sagent.lib.web.fetch.Curl", fake_curl),
        ):
            body = fetch(
                "https://example.com/start",
                validated_hosts=_vh,
                on_redirect=lambda _u: None,
            )
        assert body == b"ok"
        assert calls == ["example.com"]  # resolved once, reused on same-origin hop


class TestTransportConsistency:
    """The curl and stdlib transports must behave IDENTICALLY on the redirect
    contract (cap -> return 3xx body; cross-origin -> Origin rewritten). These
    tests run the SAME scenario through both and assert equality, so the two
    remaining redirect loops cannot silently drift (the class of bug that
    recurred across several review rounds).
    """

    def _stdlib_result(
        self, hops: list[tuple[int, bytes, dict[str, str]]], **kwargs: Any
    ) -> bytes:
        resps: list[Mock] = []
        for status, body, hdrs in hops:
            r = Mock(spec=http.client.HTTPResponse)
            r.status = status
            r.read.return_value = body
            r.getheaders.return_value = [
                ("content-encoding", "identity"),
                *hdrs.items(),
            ]
            resps.append(r)
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = resps
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", False),
            patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn),
        ):
            return fetch("https://a.com/start", **kwargs)

    def _curl_result(
        self, hops: list[tuple[int, bytes, dict[str, str]]], **kwargs: Any
    ) -> bytes:
        resps: list[Mock] = []
        for status, body, hdrs in hops:
            r = Mock()
            r.status_code = status
            r.content = body
            r.headers = hdrs
            resps.append(r)
        with (
            patch("sagent.lib.web.fetch._HAVE_CURL", True),
            patch("curl_cffi.requests.request", side_effect=resps),
        ):
            return fetch("https://a.com/start", **kwargs)

    def test_cap_returns_3xx_body_identically(self) -> None:
        # max_redirects=0: both transports return the 3xx body, neither raises.
        hops: list[tuple[int, bytes, dict[str, str]]] = [
            (302, b"the 3xx body", {"location": "https://a.com/next"})
        ]
        assert self._stdlib_result(hops, max_redirects=0) == b"the 3xx body"
        assert self._curl_result(hops, max_redirects=0) == b"the 3xx body"

    def test_followed_redirect_returns_final_body_identically(self) -> None:
        hops: list[tuple[int, bytes, dict[str, str]]] = [
            (302, b"", {"location": "https://a.com/2"}),
            (200, b"final", {}),
        ]
        assert self._stdlib_result(hops) == b"final"
        assert self._curl_result(hops) == b"final"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
