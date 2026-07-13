"""Tests for sagent.lib.web.fetch."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock, patch

import base64
import gzip
import http.client
import io
import warnings
import zlib

from curl_cffi import (
    CurlError,
    CurlInfo,
    CurlOpt,
    requests as cc_requests,
)

import brotli
import pytest
import zstandard

from sagent.lib.web import fetch as fetch_mod
from sagent.lib.web.errors import (
    BotDetectionError,
    CloudflareChallengeError,
    FetchError,
    PuzzleChallengeError,
)
from sagent.lib.web.fetch import (
    FetchSession,
    RequestParams,
    ValidatedHost,
    _apply_redirect,
    _bracket_ipv6,
    _decompress,
    _open_connection,
    _registrable_domain,
    _rewrite_origin,
    _seed_session_jar,
    _send_as,
    _split_userinfo,
    egress_ip as _real_egress_ip,
    fetch,
    last_known_egress_ip as _last_known_egress_ip,
    set_last_egress_ip as _set_last_egress_ip,
)
from sagent.lib.web.fetch_zendriver import BrowserResult
from sagent.lib.web.profile import Profile, ProfileStore


# Captured before any fixture stubs it, so the pool-locking tests can invoke the
# REAL curl_session (isolate_profiles replaces the module attribute).
_REAL_CURL_SESSION = fetch_mod.curl_session


def _lower_headers(kw: dict[str, Any]) -> dict[str, str]:
    """Lower-cased request headers from a curl ``request`` mock's kwargs."""
    headers = cast("dict[str, str] | None", kw.get("headers")) or {}
    return {k.lower(): v for k, v in headers.items()}


def _const_curl_session(stub: Any) -> Callable[..., Any]:
    """A ``curl_session`` replacement that always returns ``stub`` (typed)."""

    def factory(*_args: object) -> Any:
        return stub

    return factory


if TYPE_CHECKING:
    from curl_cffi.requests import Response


class _StubCookie:
    """A minimal jar entry: just a name/value, enough for the pooled-path tests."""

    def __init__(self, name: str, value: str) -> None:
        self.name = name
        self.value = value


class _StubCookies:
    """Minimal curl-cookies stand-in: a recording jar plus a ``set`` that stores."""

    def __init__(self) -> None:
        self.jar: list[Any] = []

    def set(
        self,
        name: str,
        value: str,
        *,
        domain: str = "",
        path: str = "/",
        secure: bool = False,
    ) -> None:
        del domain, path, secure
        self.jar = [c for c in self.jar if getattr(c, "name", None) != name]
        self.jar.append(_StubCookie(name, value))


class _StubSession:
    """A pooled-Session stand-in whose request delegates to the module-level
    ``curl_cffi.requests.request`` -- so one ``patch("curl_cffi.requests.request")``
    intercepts both the identity (session) and keyless paths.
    """

    def __init__(self) -> None:
        self.cookies = _StubCookies()

    def request(self, *args: Any, **kwargs: Any) -> Any:
        return cc_requests.request(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType] -- curl_cffi's **RequestParams TypedDict is unstubbed

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def isolate_profiles(tmp_path: Any, monkeypatch: Any) -> Any:
    """Hermetic identity layer: a tmp store and a fixed egress, no network.

    ``fetch`` transparently loads/saves a per-``(egress_ip, domain)`` profile.
    Without isolation these tests would share the real on-disk store (state
    leaking between cases) and call the live egress echo. Point the store at a
    tmp dir and pin the egress so the transport assertions stay deterministic.
    """

    def fixed_egress(*, cache: bool = True, **_kw: Any) -> str:
        del cache, _kw
        return "203.0.113.1"

    store = ProfileStore(base_dir=tmp_path)
    monkeypatch.setattr(ProfileStore, "shared", classmethod(lambda _cls: store))
    monkeypatch.setattr(fetch_mod, "egress_ip", fixed_egress)
    monkeypatch.setattr(fetch_mod, "_last_egress_ip", None)

    def stub_session(egress: str, domain: str, impersonate: str) -> _StubSession:
        del egress, domain, impersonate
        return _StubSession()

    monkeypatch.setattr(fetch_mod, "curl_session", stub_session)
    monkeypatch.setattr(fetch_mod, "_curl_sessions", {})
    return


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
            "https://x/submit",
            {"content-type": "application/json", "Accept": "*/*"},
            "POST",
            b"{}",
            303,
            "https://x/result",
        )
        assert method == "GET"
        assert body is None
        assert not any(k.lower() == "content-type" for k in headers)

    def test_302_downgrades_post_to_get(self) -> None:
        _headers, method, body = _apply_redirect(
            "https://x/submit", {}, "POST", b"{}", 302, "https://x/land"
        )
        assert method == "GET"
        assert body is None

    def test_307_preserves_method_and_body(self) -> None:
        _headers, method, body = _apply_redirect(
            "https://x/submit", {}, "POST", b"{}", 307, "https://x/land"
        )
        assert method == "POST"
        assert body == b"{}"

    def test_cross_origin_drops_cookie_and_hints(self) -> None:
        headers, _m, _b = _apply_redirect(
            "https://a.com/1",
            {"Cookie": "SID=x", "sec-ch-ua-arch": '"x86"', "Accept": "*/*"},
            "GET",
            None,
            302,
            "https://b.com/2",
        )
        assert "Cookie" not in headers
        assert "sec-ch-ua-arch" not in headers
        assert headers.get("Accept") == "*/*"  # non-origin-bound survives

    def test_same_origin_keeps_cookie_and_hints(self) -> None:
        headers, _m, _b = _apply_redirect(
            "https://a.com/1",
            {"Cookie": "SID=x", "sec-ch-ua-arch": '"x86"'},
            "GET",
            None,
            302,
            "https://a.com/2",
        )
        assert headers.get("Cookie") == "SID=x"
        assert headers.get("sec-ch-ua-arch") == '"x86"'


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

    def test_raw_deflate_without_zlib_header(self) -> None:
        # REV2A-002: some servers emit raw DEFLATE (no zlib wrapper); a browser
        # falls back to wbits=-MAX_WBITS. We must decode it, not raise.
        data = b"hello world"
        raw = zlib.compress(data)[2:-4]  # strip zlib header + adler checksum
        assert _decompress(raw, "deflate") == data

    def test_chained_content_encoding(self) -> None:
        # REV2A-003: chained "gzip, br" is RFC-legal; apply right-to-left.
        data = b"hello world"
        chained = gzip.compress(brotli.compress(data))
        assert _decompress(chained, "br, gzip") == data


class TestBackoffDelay:
    def test_exponential_growth(self) -> None:
        d0 = RequestParams().backoff_delay(0, {})
        d2 = RequestParams().backoff_delay(2, {})
        assert d0 < d2

    def test_capped_at_30(self) -> None:
        assert RequestParams().backoff_delay(100, {}) <= 45  # 30 + 0.5*30

    def test_retry_after_header(self) -> None:
        assert RequestParams().backoff_delay(0, {"retry-after": "5"}) == 5.0

    def test_retry_after_capped(self) -> None:
        assert RequestParams().backoff_delay(0, {"retry-after": "999"}) == 30.0

    def test_retry_after_http_date_honored(self) -> None:
        # REV2A-007: Retry-After may be an HTTP-date, not just delta-seconds.
        # A near-future date must produce a positive delay (honored), not fall
        # through to exponential backoff.
        future = datetime.now(tz=UTC) + timedelta(seconds=10)
        delay = RequestParams().backoff_delay(
            0, {"retry-after": format_datetime(future)}
        )
        assert 5 <= delay <= 30  # ~10s, capped at 30; not the ~1s exp backoff

    def test_retry_after_past_date_is_zero(self) -> None:
        # A past HTTP-date means "retry now": non-negative, small.
        assert (
            RequestParams().backoff_delay(
                0, {"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}
            )
            == 0.0
        )


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
            fetch("https://example.com", request=RequestParams(retries=-1))

    def test_negative_max_redirects_rejected(self) -> None:
        # O-WEB-007: max_redirects=-1 silently behaves like 0 (never follow),
        # but the contract documents only 0 as "disable". Reject the ambiguous -1.
        with pytest.raises(ValueError, match="max_redirects"):
            fetch("https://example.com", request=RequestParams(max_redirects=-1))

    def test_nonpositive_timeout_rejected(self) -> None:
        # O-WEB-008: timeout_sec=0 means opposite things per transport (curl 0 =
        # no timeout, stdlib 0 = non-blocking). Reject non-positive timeouts.
        with pytest.raises(ValueError, match="timeout_sec"):
            fetch("https://example.com", request=RequestParams(timeout_sec=0))


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
        # These tests pin the stdlib (http.client connection) transport by
        # passing backend="stdlib" on each fetch call.
        return

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
            result, _ = fetch(
                "https://example.com", request=RequestParams(backend="stdlib")
            )
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
            assert (
                fetch("https://example.com", request=RequestParams(backend="stdlib"))[0]
                == b"hello"
            )

    def test_post_with_data(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch(
                "https://example.com",
                request=RequestParams(
                    method="POST", data={"q": "test"}, backend="stdlib"
                ),
            )
        assert mock_conn.request.call_args.args[0] == "POST"
        assert mock_conn.request.call_args.kwargs["body"] == b"q=test"

    def test_post_with_json(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch(
                "https://example.com",
                request=RequestParams(
                    method="POST", json={"key": "value"}, backend="stdlib"
                ),
            )
        assert mock_conn.request.call_args.kwargs["body"] == b'{"key": "value"}'
        headers = mock_conn.request.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"

    def test_data_and_json_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            fetch(
                "https://example.com",
                request=RequestParams(data={"a": "1"}, json={"b": 2}, backend="stdlib"),
            )

    def test_cookies_serialized(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch(
                "https://example.com",
                request=RequestParams(cookies={"a": "1", "b": "2"}, backend="stdlib"),
            )
        headers = mock_conn.request.call_args.kwargs["headers"]
        assert "a=1" in headers["Cookie"]
        assert "b=2" in headers["Cookie"]

    def test_custom_headers_override_defaults(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch(
                "https://example.com",
                request=RequestParams(
                    headers={"User-Agent": "custom"}, backend="stdlib"
                ),
            )
        headers = mock_conn.request.call_args.kwargs["headers"]
        assert headers["User-Agent"] == "custom"

    def test_raw_headers_skip_defaults(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch(
                "https://example.com",
                request=RequestParams(
                    method="POST",
                    data={"q": "test"},
                    headers={"User-Agent": "custom"},
                    raw_headers=True,
                    backend="stdlib",
                ),
            )
        assert mock_conn.request.call_args.kwargs["headers"] == {"User-Agent": "custom"}

    def test_raw_headers_still_add_cookies_and_auth(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            fetch(
                "https://u:p@example.com",
                request=RequestParams(
                    headers={"User-Agent": "custom"},
                    cookies={"a": "1"},
                    raw_headers=True,
                    backend="stdlib",
                ),
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
            fetch(
                "https://u:p@example.com:8443/x",
                request=RequestParams(backend="stdlib"),
            )
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
                request=RequestParams(
                    headers={"Authorization": "Bearer xyz"}, backend="stdlib"
                ),
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
            fetch("https://example.com", request=RequestParams(backend="stdlib"))

    def test_timeout_passed(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "sagent.lib.web.fetch._open_connection", return_value=mock_conn
        ) as mock_open:
            fetch(
                "https://example.com",
                request=RequestParams(timeout_sec=60, backend="stdlib"),
            )
        assert mock_open.call_args.args[2] == 60


class TestFetchRetry:
    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> Any:
        # Stdlib path is selected per-call via backend="stdlib", not a global.
        return

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
            assert (
                fetch(
                    "https://example.com",
                    request=RequestParams(retries=1, backend="stdlib"),
                )[0]
                == b"ok"
            )

    def test_no_retry_on_404(self) -> None:
        resp = self._mock_http_response(status=404, body=b"NF")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp
        with (
            patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn),
            pytest.raises(FetchError, match="404"),
        ):
            fetch(
                "https://example.com",
                request=RequestParams(retries=3, backend="stdlib"),
            )

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
            fetch("https://x.com", request=RequestParams(backend="stdlib"))
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
            assert (
                fetch(
                    "https://example.com",
                    request=RequestParams(retries=1, backend="stdlib"),
                )[0]
                == b"ok"
            )


class TestConnectionClosedOnError:
    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> Any:
        # Stdlib path is selected per-call via backend="stdlib", not a global.
        return

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
            fetch("https://example.com", request=RequestParams(backend="stdlib"))
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
            assert (
                fetch(
                    "https://example.com",
                    request=RequestParams(retries=1, backend="stdlib"),
                )[0]
                == b"ok"
            )
        conn1.close.assert_called_once()


class TestFetchStdlibBackend:
    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> Any:
        # The stdlib backend is http.client-only; each fetch call passes
        # backend="stdlib" so these redirect/error/303/validated-host tests
        # exercise the stdlib path.
        return

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
            body, _ = fetch(
                "https://example.com/start", request=RequestParams(backend="stdlib")
            )
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
                request=RequestParams(on_redirect=urls.append, backend="stdlib"),
            )
        assert urls == ["https://example.com/final"]

    def test_set_cookie_value_with_comma_not_missplit(self) -> None:
        # RFC 9110 exempts Set-Cookie from comma-folding. A cookie VALUE that
        # itself contains ", " must not be split into two bogus cookies. Two
        # separate Set-Cookie headers must both survive intact.
        resp = self._mock_http_response(
            body=b"ok",
            headers=[
                ("content-encoding", "identity"),
                ("Set-Cookie", "pref=a, b, c; Path=/"),
                ("Set-Cookie", "SID=xyz; Path=/"),
            ],
        )
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp
        with patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn):
            _body, session = fetch(
                "https://example.com/", request=RequestParams(backend="stdlib")
            )
        assert session.cookies.get("pref") == "a, b, c"
        assert session.cookies.get("SID") == "xyz"

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
            fetch(
                "https://example.com",
                request=RequestParams(on_redirect=reject, backend="stdlib"),
            )

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
            result, _ = fetch(
                "https://example.com",
                request=RequestParams(max_redirects=0, backend="stdlib"),
            )
        assert result == b"redirect body"

    def test_plain_get_curl_absent_returns_3xx_body_at_cap(self) -> None:
        # REVE559-001: a plain GET at default max_redirects, curl absent -- once
        # _fetch_simple (urllib) is gone, this routes through _fetch_stdlib,
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
            patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn),
        ):
            result, _ = fetch(
                "https://example.com", request=RequestParams(backend="stdlib")
            )  # default max_redirects=10
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
            body, _ = fetch(
                "https://example.com/start", request=RequestParams(backend="stdlib")
            )
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
            body, _ = fetch(
                "https://example.com/base/start",
                request=RequestParams(on_redirect=urls.append, backend="stdlib"),
            )
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
            fetch(
                "https://a.com/submit",
                request=RequestParams(method="POST", data={"x": "1"}, backend="stdlib"),
            )
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
            fetch(
                "https://a.com/start",
                request=RequestParams(headers={"host": "a.com"}, backend="stdlib"),
            )
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
            body, _ = fetch(
                "https://example.com/submit",
                request=RequestParams(method="POST", data={"x": "1"}, backend="stdlib"),
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
            fetch("https://example.com", request=RequestParams(backend="stdlib"))

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
            fetch("https://example.com", request=RequestParams(backend="stdlib"))

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
            fetch("https://example.com", request=RequestParams(backend="stdlib"))
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
            fetch("https://example.com", request=RequestParams(backend="stdlib"))
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
            fetch(
                "https://example.com:8443/page",
                request=RequestParams(validated_hosts=_vh, backend="stdlib"),
            )
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
            fetch(
                "https://example.com:8443/page",
                request=RequestParams(validated_hosts=_vh, backend="stdlib"),
            )
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
            fetch(
                "https://example.com/page",
                request=RequestParams(validated_hosts=_vh, backend="stdlib"),
            )
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
            fetch(
                "https://example.com/start",
                request=RequestParams(validated_hosts=_vh, backend="stdlib"),
            )
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
        # Stdlib path is selected per-call via backend="stdlib", not a global.
        return

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
            fetch(
                "https://example.com/",
                request=RequestParams(backend="stdlib", **fetch_kwargs),
            )
        return dict(mock_conn.request.call_args.kwargs["headers"])

    def test_get_navigation_order(self) -> None:
        # The exact order a real Chrome 146 navigation sends (captured on the
        # wire): no Connection header, sec-ch-ua first, Priority last.
        headers = self._capture_headers()
        assert list(headers) == [
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
            "Priority",
        ]
        assert headers["Sec-Fetch-Mode"] == "navigate"
        assert "Chrome/" in headers["User-Agent"]

    def test_post_xhr_order_with_json(self) -> None:
        headers = self._capture_headers(method="POST", json={"q": "x"})
        assert list(headers) == [
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
            "Priority",
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
            fetch(
                "https://example.com/",
                request=RequestParams(validated_hosts=_vh, backend="stdlib"),
            )

        captured = dict(mock_conn.request.call_args.kwargs["headers"])
        assert next(iter(captured)) == "Host"
        assert captured["Host"] == "example.com"


class TestRegistrableDomain:
    """The session pool keys on eTLD+1 so sibling subdomains coalesce onto one
    connection (a browser's HTTP/2 coalescing); ``www.google.com`` and
    ``scholar.google.com`` must map to the same key.
    """

    def test_subdomains_share_registrable_domain(self) -> None:
        assert _registrable_domain("www.google.com") == "google.com"
        assert _registrable_domain("scholar.google.com") == "google.com"

    def test_bare_domain_unchanged(self) -> None:
        assert _registrable_domain("google.com") == "google.com"

    def test_single_label_unchanged(self) -> None:
        assert _registrable_domain("localhost") == "localhost"

    def test_cc_second_level_tld_keeps_three_labels(self) -> None:
        assert _registrable_domain("a.example.co.uk") == "example.co.uk"
        assert _registrable_domain("example.co.uk") == "example.co.uk"
        assert _registrable_domain("x.y.example.com.au") == "example.com.au"

    def test_plain_gtld_keeps_two_labels(self) -> None:
        assert _registrable_domain("deep.sub.example.org") == "example.org"


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
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
        ):
            body, _ = fetch("https://example.com")
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
            patch("curl_cffi.Curl", fake_curl),
        ):
            body, _ = fetch(
                "https://example.com/start",
                request=RequestParams(validated_hosts=_vh, on_redirect=lambda _u: None),
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
            patch("curl_cffi.Curl", fake_curl),
        ):
            fetch(
                "https://v6.example/x",
                request=RequestParams(validated_hosts=_vh),
            )
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
            patch("curl_cffi.Curl", fake_curl),
        ):
            fetch(
                "https://a.com/submit",
                request=RequestParams(
                    method="POST", data={"x": "1"}, validated_hosts=_vh
                ),
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
            patch("curl_cffi.requests.request", side_effect=[redir, ok]) as mock_req,
        ):
            fetch(
                "https://a.com/submit",
                request=RequestParams(method="POST", data={"x": "1"}),
            )
        second_headers = mock_req.call_args_list[1].kwargs["headers"]
        assert second_headers.get("Origin") == "https://b.com"

    def test_pooled_curl_loads_caller_cookies_into_jar_not_header(self) -> None:
        # F3 / S1: on the pooled-curl path a caller cookie is loaded INTO the
        # session jar (the single cookie source), never ALSO sent via a Cookie
        # header -- curl would then emit both, duplicating a name the jar holds.
        stub = _StubSession()
        resp = self._mock_response(content=b"ok")
        with (
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
            patch.object(fetch_mod, "curl_session", _const_curl_session(stub)),
        ):
            fetch(
                "https://example.com",
                request=RequestParams(cookies={"CONSENT": "YES+"}),
            )
        kwargs = mock_req.call_args.kwargs
        # Cookie is in the jar, not the header, and cookies= kwarg is unset.
        assert {(c.name, c.value) for c in stub.cookies.jar} == {("CONSENT", "YES+")}
        assert "Cookie" not in kwargs["headers"]
        assert not kwargs.get("cookies")

    def test_case_variant_cookie_header_not_duplicated(self) -> None:
        # REV2A-008: a caller lowercase headers={"cookie":...} plus a cookies=
        # param must collapse to ONE cookie header key (HTTP header names are
        # case-insensitive; two dict keys -> two Cookie lines on the wire).
        resp = self._mock_response(content=b"ok")
        with patch("curl_cffi.requests.request", return_value=resp) as mock_req:
            fetch(
                "https://example.com",
                request=RequestParams(headers={"cookie": "a=1"}, cookies={"b": "2"}),
            )
        sent = mock_req.call_args.kwargs["headers"]
        cookie_keys = [k for k in sent if k.lower() == "cookie"]
        assert len(cookie_keys) == 1, f"duplicate cookie header keys: {cookie_keys}"

    def test_redirect_cap_follows_up_to_limit_then_returns_body(self) -> None:
        # on_redirect fires once per FOLLOWED hop; when the cap is reached the
        # curl path returns the final 3xx body (matching _fetch_stdlib's
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
            patch("curl_cffi.Curl", fake_curl),
        ):
            body, _ = fetch(
                "https://a.com/start",
                request=RequestParams(
                    max_redirects=2, validated_hosts=_vh, on_redirect=seen.append
                ),
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
            patch("curl_cffi.Curl", fake_curl),
            pytest.raises(FetchError) as exc,
        ):
            fetch(
                "https://example.com",
                request=RequestParams(validated_hosts=_vh),
            )
        assert exc.value.status == 403
        assert exc.value.body == html

    def test_raw_headers_sends_only_provided_headers(self) -> None:
        # (d) raw_headers=True: the high-level curl request receives exactly the
        # caller's header (plus nothing derived from the Chrome default set).
        resp = self._mock_response(content=b"ok")
        with (
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
        ):
            fetch(
                "https://example.com",
                request=RequestParams(
                    headers={"User-Agent": "custom"}, raw_headers=True
                ),
            )
        assert mock_req.call_args.kwargs["headers"] == {"User-Agent": "custom"}

    def test_curl_exception_maps_to_fetch_error_status_zero(self) -> None:
        # (e) any curl_cffi exception (connection/timeout) becomes
        # FetchError(status=0) rather than leaking the raw curl error.
        with (
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
            patch("curl_cffi.requests.request", side_effect=_record),
        ):
            body, _ = fetch(
                "https://example.com/submit",
                request=RequestParams(
                    method="POST", data={"x": "1"}, on_redirect=lambda _u: None
                ),
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
            patch("curl_cffi.requests.request", side_effect=_record),
        ):
            fetch(
                "https://example.com/submit",
                request=RequestParams(method="POST", json={"x": 1}),
            )
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
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
        ):
            body, _ = fetch(
                "https://example.com",
                request=RequestParams(max_redirects=0),
            )
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
            patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn),
            patch("curl_cffi.requests.request") as mock_curl,
        ):
            body, _ = fetch(
                "https://example.com", request=RequestParams(backend="stdlib")
            )
        assert body == b"stdlib"
        mock_curl.assert_not_called()

    def test_curl_connection_error_is_retried(self) -> None:
        # A1: a curl transport error (connection refused/timeout) becomes
        # FetchError(status=0). retries= must retry it -- the stdlib path retries
        # a raw OSError, so the curl path must retry its status-0 equivalent, or
        # the two transports disagree on what `retries=` means.
        ok = self._mock_response(content=b"ok")
        with (
            patch(
                "curl_cffi.requests.request",
                side_effect=[CurlError("connection refused"), ok],
            ),
            patch("sagent.lib.web.fetch.time.sleep"),
        ):
            assert (
                fetch("https://example.com", request=RequestParams(retries=1))[0]
                == b"ok"
            )

    def test_curl_connection_error_exhausts_retries_then_raises(self) -> None:
        # The retry must still terminate: a persistent curl error raises after
        # the budget, not loop forever.
        with (
            patch("curl_cffi.requests.request", side_effect=CurlError("refused")),
            patch("sagent.lib.web.fetch.time.sleep"),
            pytest.raises(FetchError) as exc,
        ):
            fetch("https://example.com", request=RequestParams(retries=2))
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
            patch("curl_cffi.Curl", fake_curl),
        ):
            body, _ = fetch(
                "https://example.com/start",
                request=RequestParams(validated_hosts=_vh, on_redirect=lambda _u: None),
            )
        assert body == b"ok"
        assert calls == ["example.com"]  # resolved once, reused on same-origin hop


class TestOnResponse:
    """``on_response(status, headers)`` fires once per received response -- on
    success, on an HTTP error before it raises, and on every redirect hop -- for
    both transports. It is the seam a cookie jar uses to observe Set-Cookie.
    """

    def _stdlib_resp(
        self, status: int, headers: list[tuple[str, str]], body: bytes = b"ok"
    ) -> Mock:
        r = Mock(spec=http.client.HTTPResponse)
        r.status = status
        r.read.return_value = body
        r.getheaders.return_value = [("content-encoding", "identity"), *headers]
        return r

    def test_stdlib_success_reports_status_and_headers(self) -> None:
        conn = Mock(request=Mock())
        conn.getresponse.return_value = self._stdlib_resp(
            200, [("set-cookie", "GSP=abc")]
        )
        seen: list[tuple[int, dict[str, str]]] = []
        with (
            patch("sagent.lib.web.fetch._open_connection", return_value=conn),
        ):
            fetch(
                "https://x.com",
                request=RequestParams(
                    on_response=lambda s, h: seen.append((s, h)), backend="stdlib"
                ),
            )
        assert len(seen) == 1
        status, headers = seen[0]
        assert status == 200
        assert headers.get("set-cookie") == "GSP=abc"

    def test_stdlib_fires_per_redirect_hop_then_final(self) -> None:
        redir = self._stdlib_resp(
            302, [("location", "https://x.com/2"), ("set-cookie", "a=1")], b""
        )
        final = self._stdlib_resp(200, [("set-cookie", "b=2")])
        conn = Mock(request=Mock())
        conn.getresponse.side_effect = [redir, final]
        seen: list[int] = []
        with (
            patch("sagent.lib.web.fetch._open_connection", return_value=conn),
        ):
            fetch(
                "https://x.com/1",
                request=RequestParams(
                    on_response=lambda s, _h: seen.append(s), backend="stdlib"
                ),
            )
        assert seen == [302, 200]

    def test_stdlib_error_reports_before_raising(self) -> None:
        conn = Mock(request=Mock())
        conn.getresponse.return_value = self._stdlib_resp(
            404, [("set-cookie", "x=1")], b"nope"
        )
        seen: list[int] = []
        with (
            patch("sagent.lib.web.fetch._open_connection", return_value=conn),
            pytest.raises(FetchError),
        ):
            fetch(
                "https://x.com",
                request=RequestParams(
                    on_response=lambda s, _h: seen.append(s), backend="stdlib"
                ),
            )
        assert seen == [404]

    def test_curl_success_reports_status_and_headers(self) -> None:
        resp = Mock()
        resp.status_code = 200
        resp.content = b"ok"
        resp.headers = {"set-cookie": "GSP=xyz"}
        resp.url = "https://x.com/"
        seen: list[tuple[int, dict[str, str]]] = []
        with (
            patch("curl_cffi.requests.request", return_value=resp),
        ):
            fetch(
                "https://x.com",
                request=RequestParams(on_response=lambda s, h: seen.append((s, h))),
            )
        assert len(seen) == 1
        assert seen[0][0] == 200
        assert seen[0][1].get("set-cookie") == "GSP=xyz"


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
            patch("sagent.lib.web.fetch._open_connection", return_value=mock_conn),
        ):
            return fetch(
                "https://a.com/start",
                request=RequestParams(backend="stdlib", **kwargs),
            )[0]

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
            patch("curl_cffi.requests.request", side_effect=resps),
        ):
            return fetch(
                "https://a.com/start",
                request=RequestParams(**kwargs),
            )[0]

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


class TestFetchSession:
    """``FetchSession`` is a frozen browsing identity a caller threads across
    requests: ``fetch_session`` returns the session updated with what each
    response taught it (cookies set, ``Accept-CH`` opt-ins), so the next request
    is more browser-like -- the value-typed, functional API for reuse.
    """

    def _curl_response(
        self, *, headers: dict[str, str], content: bytes = b"ok"
    ) -> Mock:
        resp = Mock()
        resp.status_code = 200
        resp.content = content
        resp.headers = headers
        resp.url = "https://x.com/"
        return resp

    def test_defaults_are_empty_and_frozen(self) -> None:
        session = FetchSession()
        assert session.impersonate == "chrome"
        assert session.egress_ip == ""
        assert dict(session.cookies) == {}
        assert dict(session.accept_ch) == {}
        with pytest.raises(AttributeError):
            session.egress_ip = "1.2.3.4"  # ty: ignore[invalid-assignment]  # pyright: ignore[reportAttributeAccessIssue]

    def test_with_cookies_returns_a_merged_copy(self) -> None:
        base = FetchSession(cookies={"a": "1"})
        updated = base.with_cookies({"b": "2"})
        assert dict(updated.cookies) == {"a": "1", "b": "2"}
        assert dict(base.cookies) == {"a": "1"}  # original unchanged

    def test_with_accept_ch_records_origin_opt_in(self) -> None:
        session = FetchSession().with_accept_ch(
            "https://x.com", frozenset({"sec-ch-ua-arch"})
        )
        assert session.accept_ch["https://x.com"] == frozenset({"sec-ch-ua-arch"})

    def test_with_egress_pins_and_is_idempotent(self) -> None:
        session = FetchSession().with_egress("9.9.9.9")
        assert session.egress_ip == "9.9.9.9"
        assert session.with_egress("9.9.9.9") is session

    def test_fetch_session_returns_body_and_session(self) -> None:
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={}),
        ):
            body, session = fetch("https://x.com/p")
        assert body == b"ok"
        assert isinstance(session, FetchSession)

    def test_session_learns_set_cookie(self) -> None:
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={"set-cookie": "GSP=z; Path=/"}),
        ):
            _body, session = fetch("https://x.com/p")
        assert session.cookies["GSP"] == "z"

    def test_session_learns_accept_ch(self) -> None:
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(
                headers={"accept-ch": "Sec-CH-UA-Arch, Sec-CH-UA-Bitness"}
            ),
        ):
            _body, session = fetch("https://x.com/p")
        assert session.accept_ch["https://x.com"] == frozenset(
            {"sec-ch-ua-arch", "sec-ch-ua-bitness"}
        )

    def test_threaded_accept_ch_emits_extended_hints(self) -> None:
        # A session that opted into Accept-CH must, on the NEXT request to that
        # origin, send exactly those extended client hints -- the behavior once
        # backed by a module global, now threaded through the session.
        prior = FetchSession().with_accept_ch(
            "https://x.com", frozenset({"sec-ch-ua-arch", "sec-ch-ua-bitness"})
        )
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={}),
        ) as req:
            fetch("https://x.com/p", session=prior)
        sent = req.call_args.kwargs["headers"]
        assert "sec-ch-ua-arch" in sent
        assert "sec-ch-ua-bitness" in sent
        assert "sec-ch-ua-model" not in sent  # never opted in

    def test_cold_origin_sends_no_extended_hints(self) -> None:
        # A fresh session (no Accept-CH opt-in) sends none of the extended hints,
        # exactly as Chrome's first request to an origin does.
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={}),
        ) as req:
            fetch("https://x.com/p")
        sent = req.call_args.kwargs["headers"]
        assert "sec-ch-ua-arch" not in sent

    def test_threaded_session_seeds_prior_cookies(self) -> None:
        # Prior session cookies are loaded into the pooled jar (the single cookie
        # source on the curl path), not the Cookie header.
        prior = FetchSession(cookies={"SID": "abc"})
        stub = _StubSession()
        with (
            patch(
                "curl_cffi.requests.request",
                return_value=self._curl_response(headers={}),
            ),
            patch.object(fetch_mod, "curl_session", _const_curl_session(stub)),
        ):
            fetch("https://x.com/p", session=prior)
        assert ("SID", "abc") in {(c.name, c.value) for c in stub.cookies.jar}

    def test_caller_on_response_still_fires(self) -> None:
        seen: list[int] = []
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={"set-cookie": "a=1"}),
        ):
            fetch(
                "https://x.com/p",
                request=RequestParams(on_response=lambda s, _h: seen.append(s)),
            )
        assert seen == [200]


class TestRedirectIdentityScoping:
    """Cross-origin redirects must re-scope every origin-bound identity element.

    A real browser, following a redirect to a NEW origin, does not carry the
    source origin's Cookie header or extended client hints to the target, does
    not attribute the target's Set-Cookie to the source, and downgrades a
    301/302 POST to a bodyless GET. These tests drive the curl backend through a
    two-hop redirect and assert each of those rules on the second hop.
    """

    def _two_hop(
        self,
        *,
        first_status: int,
        target_set_cookie: str | None = None,
    ) -> Callable[..., Mock]:
        """A curl ``request`` mock: a.com/start -> (status) -> b.com/next -> 200."""

        def fake_request(_verb: str, url: str, **_kw: Any) -> Mock:
            resp = Mock()
            if url == "https://a.com/start":
                resp.status_code = first_status
                resp.headers = {"location": "https://b.com/next"}
                resp.content = b""
            else:
                resp.status_code = 200
                resp.headers = (
                    {"set-cookie": target_set_cookie} if target_set_cookie else {}
                )
                resp.content = b"done"
            resp.url = url
            return resp

        return fake_request

    def test_302_post_downgrades_to_bodyless_get(self) -> None:
        # A 301/302 POST must convert to a bodyless GET on the next hop (browser
        # behavior; only 307/308 preserve the method). Currently only 303 does.
        calls: list[tuple[str, str, object]] = []

        def fake_request(verb: str, url: str, **kw: Any) -> Mock:
            calls.append((verb, url, kw.get("data")))
            resp = Mock()
            if url == "https://a.com/start":
                resp.status_code = 302
                resp.headers = {"location": "https://a.com/land"}
                resp.content = b""
            else:
                resp.status_code = 200
                resp.headers = {}
                resp.content = b"done"
            resp.url = url
            return resp

        with (
            patch("curl_cffi.requests.request", side_effect=fake_request),
            patch.object(fetch_mod, "egress_ip", return_value=None),
        ):
            fetch(
                "https://a.com/start",
                request=RequestParams(method="POST", data={"x": "1"}),
            )
        # Second hop must be a GET with no body.
        _verb, _url, second_body = calls[1]
        assert calls[1][0] == "GET"
        assert second_body is None

    def test_cross_origin_redirect_drops_cookie_header(self) -> None:
        # a.com's session cookie must NOT be sent to b.com after a cross-origin
        # redirect (a real browser scopes cookies to their origin).
        sent: list[tuple[str, dict[str, str]]] = []

        def fake_request(_verb: str, url: str, **kw: Any) -> Mock:
            sent.append((url, _lower_headers(kw)))
            resp = Mock()
            if url == "https://a.com/start":
                resp.status_code = 302
                resp.headers = {"location": "https://b.com/next"}
                resp.content = b""
            else:
                resp.status_code = 200
                resp.headers = {}
                resp.content = b"done"
            resp.url = url
            return resp

        with (
            patch("curl_cffi.requests.request", side_effect=fake_request),
            patch.object(fetch_mod, "egress_ip", return_value=None),
        ):
            fetch(
                "https://a.com/start",
                session=FetchSession(cookies={"SID": "secret"}),
            )
        b_headers = next(h for url, h in sent if url == "https://b.com/next")
        assert "cookie" not in b_headers

    def test_same_origin_redirect_keeps_cookie_header(self) -> None:
        # A same-origin redirect must PRESERVE the cookie (the scoping rule only
        # drops on origin change).
        sent: list[tuple[str, dict[str, str]]] = []

        def fake_request(_verb: str, url: str, **kw: Any) -> Mock:
            sent.append((url, _lower_headers(kw)))
            resp = Mock()
            if url == "https://a.com/start":
                resp.status_code = 302
                resp.headers = {"location": "https://a.com/next"}
                resp.content = b""
            else:
                resp.status_code = 200
                resp.headers = {}
                resp.content = b"done"
            resp.url = url
            return resp

        with (
            patch("curl_cffi.requests.request", side_effect=fake_request),
            patch.object(fetch_mod, "egress_ip", return_value=None),
        ):
            fetch(
                "https://a.com/start",
                session=FetchSession(cookies={"SID": "secret"}),
            )
        next_headers = next(h for url, h in sent if url == "https://a.com/next")
        assert next_headers.get("cookie") == "SID=secret"

    def test_cross_origin_redirect_drops_extended_hints(self) -> None:
        # a.com's opted-in extended client hints must NOT leak to b.com.
        sent: list[tuple[str, dict[str, str]]] = []

        def fake_request(_verb: str, url: str, **kw: Any) -> Mock:
            sent.append((url, _lower_headers(kw)))
            resp = Mock()
            if url == "https://a.com/start":
                resp.status_code = 302
                resp.headers = {"location": "https://b.com/next"}
                resp.content = b""
            else:
                resp.status_code = 200
                resp.headers = {}
                resp.content = b"done"
            resp.url = url
            return resp

        session = FetchSession().with_accept_ch(
            "https://a.com", frozenset({"sec-ch-ua-arch", "sec-ch-ua-bitness"})
        )
        with (
            patch("curl_cffi.requests.request", side_effect=fake_request),
            patch.object(fetch_mod, "egress_ip", return_value=None),
        ):
            fetch("https://a.com/start", session=session)
        b_headers = next(h for url, h in sent if url == "https://b.com/next")
        assert "sec-ch-ua-arch" not in b_headers

    def test_cross_origin_target_cookie_not_persisted_to_source_profile(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        # b.com's Set-Cookie must NOT be stored in a.com's (egress,domain) profile.
        store = ProfileStore(base_dir=tmp_path)

        def _fixed_egress(**_kw: Any) -> str:
            return "9.9.9.9"

        def _no_pool(*_a: Any) -> None:
            return None

        monkeypatch.setattr(ProfileStore, "shared", classmethod(lambda _cls: store))
        monkeypatch.setattr(fetch_mod, "egress_ip", _fixed_egress)
        monkeypatch.setattr(fetch_mod, "curl_session", _no_pool)
        with patch(
            "curl_cffi.requests.request",
            side_effect=self._two_hop(
                first_status=302, target_set_cookie="FOREIGN=1; Path=/"
            ),
        ):
            fetch("https://a.com/start", request=RequestParams())
        profile = store.load("9.9.9.9", "a.com")
        assert profile is not None
        assert "FOREIGN" not in profile.cookies

    def test_cross_origin_target_cookie_not_attributed_to_source_session(
        self,
    ) -> None:
        # The returned session must not record b.com's cookie under a.com.
        with (
            patch(
                "curl_cffi.requests.request",
                side_effect=self._two_hop(
                    first_status=302, target_set_cookie="FOREIGN=1; Path=/"
                ),
            ),
            patch.object(fetch_mod, "egress_ip", return_value=None),
        ):
            _body, session = fetch("https://a.com/start")
        # a.com is the request origin; FOREIGN belongs to b.com, not a.com's jar.
        assert "FOREIGN" not in session.cookies


class TestIdentityLayer:
    """``fetch`` transparently backs each call with a persistent per-(egress,
    domain) identity: it seeds the stored UA + cookies (caller values win),
    saves ``Set-Cookie`` back, and on a bot-block of a KNOWN identity discards it
    and retries once fresh. The ``isolate_profiles`` fixture pins egress to
    ``203.0.113.1`` and points the store at a tmp dir.
    """

    _EGRESS = "203.0.113.1"

    def _curl_response(
        self, *, status: int = 200, content: bytes = b"ok", headers: dict[str, str]
    ) -> Mock:
        resp = Mock()
        resp.status_code = status
        resp.content = content
        resp.headers = headers
        resp.url = "https://x.com/"
        return resp

    def _store(self) -> ProfileStore:
        return ProfileStore.shared()

    def test_delegates_ua_and_cookie_jar_to_curl_session(self) -> None:
        # On the curl path curl_cffi's impersonate emits a coherent User-Agent
        # (matching its TLS fingerprint), so fetch does NOT send a User-Agent
        # header. The stored jar is NOT seeded into the Cookie header either --
        # the pooled curl session's own jar persists + resends cookies, so
        # header-seeding them too would duplicate the Cookie header (a bot tell).
        self._store().save(
            self._EGRESS, "x.com", Profile(ua="StoredUA/9", cookies={"GSP": "s"})
        )
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={}),
        ) as req:
            fetch("https://x.com/p")
        sent = req.call_args.kwargs["headers"]
        assert "User-Agent" not in sent
        assert "Cookie" not in sent  # jar carries the stored cookie, not the header

    def test_caller_ua_and_cookie_override_profile(self) -> None:
        self._store().save(
            self._EGRESS, "x.com", Profile(ua="StoredUA/9", cookies={"GSP": "s"})
        )
        stub = _StubSession()
        with (
            patch(
                "curl_cffi.requests.request",
                return_value=self._curl_response(headers={}),
            ) as req,
            patch.object(fetch_mod, "curl_session", _const_curl_session(stub)),
        ):
            fetch(
                "https://x.com/p",
                request=RequestParams(
                    headers={"User-Agent": "Mine/1"}, cookies={"GSP": "caller"}
                ),
            )
        sent = req.call_args.kwargs["headers"]
        assert sent["User-Agent"] == "Mine/1"
        # The caller cookie overrides the profile's GSP in the jar (single source).
        assert ("GSP", "caller") in {(c.name, c.value) for c in stub.cookies.jar}
        assert ("GSP", "s") not in {(c.name, c.value) for c in stub.cookies.jar}

    def test_no_profile_delegates_ua_to_impersonate(self) -> None:
        # First contact, no profile: still no seeded User-Agent header on the
        # curl path -- impersonate supplies a coherent one at the transport.
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={}),
        ) as req:
            fetch("https://x.com/p")
        assert "User-Agent" not in req.call_args.kwargs["headers"]

    def test_set_cookie_is_persisted(self) -> None:
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(
                headers={"set-cookie": "GSP=minted; Path=/"}
            ),
        ):
            fetch("https://x.com/p")
        got = self._store().load(self._EGRESS, "x.com")
        assert got is not None
        assert got.cookies == {"GSP": "minted"}

    def test_caller_on_response_still_fires(self) -> None:
        seen: list[int] = []
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={"set-cookie": "a=1"}),
        ):
            fetch(
                "https://x.com/p",
                request=RequestParams(on_response=lambda s, _h: seen.append(s)),
            )
        assert seen == [200]

    def test_burn_on_known_identity_discards_and_retries_fresh(self) -> None:
        self._store().save(
            self._EGRESS, "x.com", Profile(ua="PoisonUA", cookies={"GSP": "old"})
        )
        blocked = self._curl_response(
            status=403,
            content=(b'<div class="g-recaptcha" data-sitekey="x"></div>'),
            headers={"content-type": "text/html"},
        )
        ok = self._curl_response(content=b"ok", headers={})
        with patch("curl_cffi.requests.request", side_effect=[blocked, ok]) as req:
            body, _ = fetch("https://x.com/p")
        assert body == b"ok"
        assert req.call_count == 2
        # The retry used a fresh identity: no poisoned cookies ride along (the UA
        # is curl's coherent impersonate UA, never seeded, so it cannot leak).
        retry_headers = req.call_args_list[1].kwargs["headers"]
        assert "GSP=old" not in retry_headers.get("Cookie", "")
        # The poisoned identity was discarded and a fresh one saved.
        got = self._store().load(self._EGRESS, "x.com")
        assert got is not None
        assert "GSP" not in got.cookies

    def test_second_burn_raises(self) -> None:
        self._store().save(self._EGRESS, "x.com", Profile(ua="U", cookies={"GSP": "x"}))
        blocked = self._curl_response(
            status=403,
            content=b'<div class="g-recaptcha" data-sitekey="x"></div>',
            headers={"content-type": "text/html"},
        )
        with (
            patch("curl_cffi.requests.request", return_value=blocked),
            pytest.raises(PuzzleChallengeError),
        ):
            fetch("https://x.com/p")

    def test_first_contact_burn_does_not_retry(self) -> None:
        blocked = self._curl_response(
            status=403,
            content=b'<div class="g-recaptcha" data-sitekey="x"></div>',
            headers={"content-type": "text/html"},
        )
        with (
            patch("curl_cffi.requests.request", return_value=blocked) as req,
            pytest.raises(PuzzleChallengeError),
        ):
            fetch("https://x.com/p")
        assert req.call_count == 1  # no retry with no known identity

    def test_raw_headers_bypasses_identity(self) -> None:
        self._store().save(
            self._EGRESS, "x.com", Profile(ua="StoredUA", cookies={"GSP": "s"})
        )
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={}),
        ) as req:
            fetch(
                "https://x.com/p",
                request=RequestParams(headers={"User-Agent": "raw"}, raw_headers=True),
            )
        sent = req.call_args.kwargs["headers"]
        assert sent == {"User-Agent": "raw"}  # no profile UA, no stored cookie

    def test_send_as_keyless_when_egress_none(self, tmp_path: Any) -> None:
        # _send_as with egress=None draws a UA, sends, persists nothing.
        request = fetch_mod._Request(
            url="https://x.com/p",
            session=FetchSession(impersonate="chrome"),
            params=RequestParams(
                method="GET",
                params=None,
                data=None,
                json=None,
                retries=0,
                timeout_sec=30,
                max_redirects=10,
                on_redirect=None,
                on_response=None,
                validated_hosts=None,
            ),
        )
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={"set-cookie": "GSP=z"}),
        ):
            body = _send_as(request, None, None, None, None)
        assert body == b"ok"
        assert not list(tmp_path.glob("*.json"))


class TestCurlSessionPoolLocking:
    """Every mutation of the ``_curl_sessions`` pool holds ``_egress_lock``, so a
    concurrent ``set_last_egress_ip`` close-sweep never races an insert/pop.
    """

    @pytest.fixture(autouse=True)
    def _real_curl_session(self, monkeypatch: Any) -> None:
        # The module isolate_profiles fixture stubs curl_session; restore the
        # real function so these tests exercise its actual locking.
        monkeypatch.setattr(fetch_mod, "curl_session", _REAL_CURL_SESSION)

    def test_curl_session_holds_egress_lock(self, monkeypatch: Any) -> None:
        acquired: list[str] = []
        real_lock = fetch_mod._egress_lock

        class _Instrumented:
            def __enter__(self) -> None:
                acquired.append("enter")
                real_lock.acquire()

            def __exit__(self, *_a: object) -> None:
                real_lock.release()

        monkeypatch.setattr(fetch_mod, "_egress_lock", _Instrumented())
        monkeypatch.setattr(fetch_mod, "_curl_sessions", {})
        with patch("curl_cffi.requests.Session", return_value=Mock()):
            fetch_mod.curl_session("1.2.3.4", "x.com", "chrome")
        assert acquired, "curl_session mutated the pool without _egress_lock"

    def test_close_curl_session_holds_egress_lock(self, monkeypatch: Any) -> None:
        acquired: list[str] = []
        real_lock = fetch_mod._egress_lock

        class _Instrumented:
            def __enter__(self) -> None:
                acquired.append("enter")
                real_lock.acquire()

            def __exit__(self, *_a: object) -> None:
                real_lock.release()

        monkeypatch.setattr(fetch_mod, "_egress_lock", _Instrumented())
        monkeypatch.setattr(fetch_mod, "_curl_sessions", {})
        fetch_mod.close_curl_session("1.2.3.4", "x.com", "chrome")  # absent: no-op
        assert acquired, "close_curl_session mutated the pool without _egress_lock"


class TestEgressIp:
    """``egress_ip`` probes an echo cascade for the host's public IP, memoizing
    into the last-known global; ``cache=True`` reads it without a network call,
    ``cache=False`` refreshes it, ``last_known_egress_ip`` is a pure read.
    """

    @pytest.fixture(autouse=True)
    def _real_egress(self, monkeypatch: Any) -> Any:
        # The module isolate_profiles fixture stubs egress_ip to a fixed value;
        # restore the REAL function here and just reset the last-known global.
        monkeypatch.setattr(fetch_mod, "egress_ip", _real_egress_ip)
        monkeypatch.setattr(fetch_mod, "_last_egress_ip", None)
        return

    def _probe(self, fetch_mock: Mock, *, ipv6: bool = False) -> str | None:
        # egress_ip unpacks fetch's (body, session) tuple; adapt the byte-valued
        # mock so a bytes return becomes (bytes, session) and an exception still
        # raises (the echo-cascade paths this test exercises).
        def adapt(*args: Any, **kwargs: Any) -> tuple[bytes, FetchSession]:
            return fetch_mock(*args, **kwargs), FetchSession()

        with patch.object(fetch_mod, "fetch", side_effect=adapt):
            return _real_egress_ip(cache=False, ipv6=ipv6)

    def test_first_echo_returned(self) -> None:
        assert self._probe(Mock(return_value=b" 203.0.113.7\n")) == "203.0.113.7"

    def test_non_v4_reply_falls_through(self) -> None:
        assert self._probe(Mock(side_effect=[b"2001:db8::1", b"198.51.100.9"])) == (
            "198.51.100.9"
        )

    def test_fetch_error_falls_through(self) -> None:
        err = FetchError(url="u", status=500, headers={}, body=b"")
        assert self._probe(Mock(side_effect=[err, b"192.0.2.5"])) == "192.0.2.5"

    def test_all_fail_resolves_none(self) -> None:
        assert self._probe(Mock(side_effect=OSError("offline"))) is None

    def test_v6_echo_returned(self) -> None:
        assert (
            self._probe(Mock(return_value=b"2606:4700:4700::1111\n"), ipv6=True)
            == "2606:4700:4700::1111"
        )

    def test_v4_reply_rejected_for_v6_request(self) -> None:
        assert self._probe(Mock(return_value=b"203.0.113.7"), ipv6=True) is None

    def test_uses_v6_endpoints(self) -> None:
        mock = Mock(return_value=b"2001:db8::5")
        self._probe(mock, ipv6=True)
        assert "ipv6" in mock.call_args.args[0] or "api64" in mock.call_args.args[0]

    def test_malformed_v6_reply_rejected(self) -> None:
        assert self._probe(Mock(return_value=b"::::"), ipv6=True) is None
        assert self._probe(Mock(return_value=b"ff:"), ipv6=True) is None

    def test_probe_records_last_known(self) -> None:
        assert _last_known_egress_ip() is None
        self._probe(Mock(return_value=b"203.0.113.7"))
        assert _last_known_egress_ip() == "203.0.113.7"

    def test_cache_true_returns_last_known_without_probing(
        self, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(fetch_mod, "_last_egress_ip", "9.9.9.9")
        echo = Mock()
        with patch.object(fetch_mod, "fetch", echo):
            assert _real_egress_ip() == "9.9.9.9"
        echo.assert_not_called()

    def test_cache_true_probes_to_fill_empty(self) -> None:
        echo = Mock(return_value=(b"1.2.3.4", FetchSession()))
        with patch.object(fetch_mod, "fetch", echo):
            assert _real_egress_ip() == "1.2.3.4"
        assert echo.call_count == 1
        assert _last_known_egress_ip() == "1.2.3.4"

    def test_cache_false_always_probes_and_refreshes(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(fetch_mod, "_last_egress_ip", "1.1.1.1")
        with patch.object(
            fetch_mod, "fetch", Mock(return_value=(b"2.2.2.2", FetchSession()))
        ):
            assert _real_egress_ip(cache=False) == "2.2.2.2"
        assert _last_known_egress_ip() == "2.2.2.2"

    def test_failed_probe_leaves_last_known_untouched(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(fetch_mod, "_last_egress_ip", "keepme")
        with patch.object(fetch_mod, "fetch", Mock(side_effect=OSError("x"))):
            assert _real_egress_ip(cache=False) is None
        assert _last_known_egress_ip() == "keepme"

    def test_set_last_egress_ip_injects_without_probing(self) -> None:
        # A caller who knows the egress (e.g. just rolled the VPN) can set it;
        # a cached read then returns it with no network.
        _set_last_egress_ip("5.5.5.5")
        echo = Mock()
        with patch.object(fetch_mod, "fetch", echo):
            assert _real_egress_ip() == "5.5.5.5"
        echo.assert_not_called()
        assert _last_known_egress_ip() == "5.5.5.5"


class TestSeedSessionJar:
    def test_secure_prefixed_cookie_seeded_without_warning(self) -> None:
        # RFC 6265bis: a __Secure-/__Host- prefixed cookie is only valid Secure;
        # seeding it without secure=True made curl_cffi emit a CurlCffiWarning
        # (which the live Google-search integration path surfaced as a failure).
        session = cast("cc_requests.Session[Response]", cc_requests.Session())
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                _seed_session_jar(
                    session,
                    "www.google.com",
                    {"__Secure-STRP": "abc", "__Host-GSP": "def", "NID": "ghi"},
                )
        finally:
            session.close()
        jar = {c.name: c for c in session.cookies.jar}
        assert jar["__Secure-STRP"].secure is True
        assert jar["__Secure-STRP"].domain == "www.google.com"
        # __Host- is host-only per spec: Secure, no Domain, Path=/.
        assert jar["__Host-GSP"].secure is True
        assert jar["__Host-GSP"].path == "/"
        # A plain cookie is seeded non-Secure (Chrome sends it over either).
        assert jar["NID"].secure is False


class TestBrowserBackend:
    """The opt-in ``backend="zendriver"`` path and its parameter guards."""

    def test_rejects_non_get_method(self) -> None:
        with pytest.raises(ValueError, match="zendriver backend supports only GET"):
            RequestParams(backend="zendriver", method="POST")

    def test_rejects_request_body(self) -> None:
        with pytest.raises(ValueError, match="cannot send a request body"):
            RequestParams(backend="zendriver", data={"a": "1"})

    def test_rejects_validated_hosts(self) -> None:
        with pytest.raises(ValueError, match="cannot honor 'validated_hosts'"):
            RequestParams(
                backend="zendriver",
                validated_hosts=lambda h: ValidatedHost(host=h, ip="1.2.3.4"),
            )

    def test_default_backend_is_curl(self) -> None:
        assert RequestParams().backend == "curl"

    def test_browser_fetch_returns_body_and_warms_session(self) -> None:
        # A browser fetch must return the rendered bytes AND fold the browser's
        # harvested cookies into the returned FetchSession, so a following curl
        # fetch on the same session is warm (the review's key requirement).
        result = BrowserResult(body=b"<html>rendered</html>", cookies={"SID": "xyz"})
        with (
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch("sagent.lib.web.fetch.fetch_zendriver", return_value=result) as via,
        ):
            body, session = fetch(
                "https://walled.example/x",
                request=RequestParams(backend="zendriver"),
            )
        assert body == b"<html>rendered</html>"
        assert session.cookies == {"SID": "xyz"}  # session warmed
        assert via.call_count == 1

    def test_browser_fetch_persists_cookies_to_profile_store(self) -> None:
        result = BrowserResult(body=b"ok", cookies={"cf_clearance": "tok"})
        store = Mock()
        store.load.return_value = None
        with (
            patch.object(fetch_mod, "egress_ip", return_value="5.5.5.5"),
            patch("sagent.lib.web.fetch.fetch_zendriver", return_value=result),
            patch("sagent.lib.web.profile.ProfileStore.shared", return_value=store),
        ):
            fetch(
                "https://walled.example/x",
                request=RequestParams(backend="zendriver"),
            )
        # A fresh (egress, domain) key is saved with the harvested cookies.
        store.save.assert_called_once()
        saved_profile = store.save.call_args.args[2]
        assert saved_profile.cookies == {"cf_clearance": "tok"}


class TestCurlThenZendriverBackend:
    """``backend="curl-then-zendriver"``: curl first, zendriver only on a bot block."""

    def test_curl_then_zendriver_inherits_zendriver_restrictions(self) -> None:
        # curl-then-zendriver may fall back to the browser, so it carries the same GET-only /
        # no-body / no-validated_hosts limits (nothing to fall back a POST to).
        with pytest.raises(
            ValueError, match="curl-then-zendriver backend supports only GET"
        ):
            RequestParams(backend="curl-then-zendriver", method="POST")
        with pytest.raises(
            ValueError, match="curl-then-zendriver backend cannot send a request"
        ):
            RequestParams(backend="curl-then-zendriver", data={"a": "1"})
        with pytest.raises(
            ValueError, match="curl-then-zendriver backend cannot honor"
        ):
            RequestParams(
                backend="curl-then-zendriver",
                validated_hosts=lambda h: ValidatedHost(host=h, ip="1.2.3.4"),
            )

    def test_curl_then_zendriver_returns_curl_body_without_touching_browser(
        self,
    ) -> None:
        # When curl succeeds, the browser backend is never invoked.
        with (
            patch.object(fetch_mod, "_send_as", return_value=b"curl body"),
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch("sagent.lib.web.fetch.fetch_zendriver") as via,
        ):
            body, _ = fetch(
                "https://ok.example/",
                request=RequestParams(backend="curl-then-zendriver"),
            )
        assert body == b"curl body"
        via.assert_not_called()

    def test_curl_then_zendriver_falls_back_to_zendriver_on_bot_block(self) -> None:
        # A curl BotDetectionError triggers the zendriver leg; its body is returned.
        result = BrowserResult(body=b"rendered", cookies={"cf_clearance": "t"})
        with (
            patch.object(fetch_mod, "_send_as", side_effect=CloudflareChallengeError()),
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch("sagent.lib.web.fetch.fetch_zendriver", return_value=result) as via,
        ):
            body, _ = fetch(
                "https://walled.example/",
                request=RequestParams(backend="curl-then-zendriver"),
            )
        assert body == b"rendered"
        assert via.call_count == 1

    def test_curl_then_zendriver_does_not_fall_back_on_non_block_error(self) -> None:
        # A plain 404 (not a bot block) propagates -- the browser would not help
        # and must not silently pay Chrome's launch cost.
        with (
            patch.object(
                fetch_mod,
                "_send_as",
                side_effect=FetchError("https://x/", 404, {}, b""),
            ),
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch("sagent.lib.web.fetch.fetch_zendriver") as via,
            pytest.raises(FetchError),
        ):
            fetch("https://x/", request=RequestParams(backend="curl-then-zendriver"))
        via.assert_not_called()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
