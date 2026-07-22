"""Tests for sagent.lib.web.fetch."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any
from unittest.mock import Mock, patch

import base64
import http.client
import importlib

import pytest
import zstandard

from sagent.lib.web.errors import (
    BotDetectionError,
    CloudflareChallengeError,
    FetchError,
    GoogleJavascriptRequiredError,
    PuzzleChallengeError,
)
from sagent.lib.web.fetch import FetchSession, RequestParams, ValidatedHost, fetch
from sagent.lib.web.fetch.fetch import (
    _send_as,
    _split_userinfo,
    egress_ip as _real_egress_ip,
    last_known_egress_ip as _last_known_egress_ip,
    set_last_egress_ip as _set_last_egress_ip,
)
from sagent.lib.web.fetch.test_helpers import (
    StubSession,
    const_curl_session,
    lower_headers,
)
from sagent.lib.web.fetch.zendriver import BrowserResult
from sagent.lib.web.profile import Profile, ProfileStore

import sagent.lib.web.fetch as fetch_package


fetch_mod = importlib.import_module("sagent.lib.web.fetch.fetch")


def test_fetch_uses_transport_package_layout() -> None:
    assert fetch_package.__file__ is not None
    assert fetch_package.__file__.endswith("/fetch/__init__.py")
    assert callable(fetch_package.fetch)
    for module in (
        "common",
        "curl",
        "fetch",
        "stdlib",
        "zendriver",
    ):
        importlib.import_module(f"sagent.lib.web.fetch.{module}")


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
            fetch("https://x.com", request=RequestParams(transport="curl"))
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
            fetch("https://x.com", request=RequestParams(transport="curl"))
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
            fetch("https://x.com", request=RequestParams(transport="curl"))
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
                fetch("https://x.com", request=RequestParams(transport="curl"))
            except FetchError as e:
                caught = e
        assert isinstance(caught, CloudflareChallengeError)


class TestFetchRetry:
    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> Any:
        # Stdlib path is selected per-call via transport="stdlib", not a global.
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
            patch(
                "sagent.lib.web.fetch.stdlib._open_connection",
                return_value=mock_conn,
            ),
            patch("sagent.lib.web.fetch.fetch.time.sleep"),
        ):
            assert (
                fetch(
                    "https://example.com",
                    request=RequestParams(retries=1, transport="stdlib"),
                )[0]
                == b"ok"
            )

    def test_no_retry_on_404(self) -> None:
        resp = self._mock_http_response(status=404, body=b"NF")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp
        with (
            patch(
                "sagent.lib.web.fetch.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError, match="404"),
        ):
            fetch(
                "https://example.com",
                request=RequestParams(retries=3, transport="stdlib"),
            )

    def test_error_body_isdecompressed(self) -> None:
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
            patch(
                "sagent.lib.web.fetch.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError) as exc,
        ):
            fetch("https://x.com", request=RequestParams(transport="stdlib"))
        # The caller must receive readable HTML, not the raw zstd frame.
        assert exc.value.body == html

    def test_retries_on_network_error(self) -> None:
        resp_ok = self._mock_http_response(body=b"ok")
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.side_effect = [OSError("refused"), resp_ok]

        with (
            patch(
                "sagent.lib.web.fetch.stdlib._open_connection",
                return_value=mock_conn,
            ),
            patch("sagent.lib.web.fetch.fetch.time.sleep"),
        ):
            assert (
                fetch(
                    "https://example.com",
                    request=RequestParams(retries=1, transport="stdlib"),
                )[0]
                == b"ok"
            )


class TestHeaderOrder:
    """Lock the canonical Chrome header order on the wire.

    http.client emits user-supplied headers in dict insertion order, so
    asserting the dict's key order asserts the wire order. ``Host`` and
    ``Content-Length`` are added by http.client itself (right after the
    request line) and are not part of the user-headers dict here.
    """

    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> Any:
        # Stdlib path is selected per-call via transport="stdlib", not a global.
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
            "sagent.lib.web.fetch.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://example.com/",
                request=RequestParams(transport="stdlib", **fetch_kwargs),
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
            "sagent.lib.web.fetch.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://example.com/",
                request=RequestParams(validated_hosts=_vh, transport="stdlib"),
            )

        captured = dict(mock_conn.request.call_args.kwargs["headers"])
        assert next(iter(captured)) == "Host"
        assert captured["Host"] == "example.com"


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
            patch("sagent.lib.web.fetch.stdlib._open_connection", return_value=conn),
        ):
            fetch(
                "https://x.com",
                request=RequestParams(
                    on_response=lambda s, h: seen.append((s, h)), transport="stdlib"
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
            patch("sagent.lib.web.fetch.stdlib._open_connection", return_value=conn),
        ):
            fetch(
                "https://x.com/1",
                request=RequestParams(
                    on_response=lambda s, _h: seen.append(s), transport="stdlib"
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
            patch("sagent.lib.web.fetch.stdlib._open_connection", return_value=conn),
            pytest.raises(FetchError),
        ):
            fetch(
                "https://x.com",
                request=RequestParams(
                    on_response=lambda s, _h: seen.append(s), transport="stdlib"
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
            patch(
                "sagent.lib.web.fetch.stdlib._open_connection",
                return_value=mock_conn,
            ),
        ):
            return fetch(
                "https://a.com/start",
                request=RequestParams(transport="stdlib", **kwargs),
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
            fetch("https://x.com/p", request=RequestParams(transport="curl"))
        sent = req.call_args.kwargs["headers"]
        assert "sec-ch-ua-arch" not in sent

    def test_threaded_session_seeds_prior_cookies(self) -> None:
        # Prior session cookies are loaded into the pooled jar (the single cookie
        # source on the curl path), not the Cookie header.
        prior = FetchSession(cookies={"SID": "abc"})
        stub = StubSession()
        with (
            patch(
                "curl_cffi.requests.request",
                return_value=self._curl_response(headers={}),
            ),
            patch.object(fetch_mod, "curl_session", const_curl_session(stub)),
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
            sent.append((url, lower_headers(kw)))
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
            sent.append((url, lower_headers(kw)))
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
            sent.append((url, lower_headers(kw)))
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
            fetch("https://x.com/p", request=RequestParams(transport="curl"))
        sent = req.call_args.kwargs["headers"]
        assert "User-Agent" not in sent
        assert "Cookie" not in sent  # jar carries the stored cookie, not the header

    def test_caller_ua_and_cookie_override_profile(self) -> None:
        self._store().save(
            self._EGRESS, "x.com", Profile(ua="StoredUA/9", cookies={"GSP": "s"})
        )
        stub = StubSession()
        with (
            patch(
                "curl_cffi.requests.request",
                return_value=self._curl_response(headers={}),
            ) as req,
            patch.object(fetch_mod, "curl_session", const_curl_session(stub)),
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
            fetch("https://x.com/p", request=RequestParams(transport="curl"))
        assert "User-Agent" not in req.call_args.kwargs["headers"]

    def test_set_cookie_is_persisted(self) -> None:
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(
                headers={"set-cookie": "GSP=minted; Path=/"}
            ),
        ):
            fetch("https://x.com/p", request=RequestParams(transport="curl"))
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
            fetch("https://x.com/p", request=RequestParams(transport="curl"))

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
            fetch("https://x.com/p", request=RequestParams(transport="curl"))
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
                transport="curl",
            ),
        )
        with patch(
            "curl_cffi.requests.request",
            return_value=self._curl_response(headers={"set-cookie": "GSP=z"}),
        ):
            body = _send_as(request, None, None, None, None)
        assert body == b"ok"
        assert not list(tmp_path.glob("*.json"))


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


class TestBrowserBackend:
    """The opt-in ``transport="zendriver"`` path and its parameter guards."""

    def test_rejects_non_get_method(self) -> None:
        with pytest.raises(ValueError, match="zendriver backend supports only GET"):
            RequestParams(transport="zendriver", method="POST")

    def test_rejects_request_body(self) -> None:
        with pytest.raises(ValueError, match="cannot send a request body"):
            RequestParams(transport="zendriver", data={"a": "1"})

    def test_rejects_validated_hosts(self) -> None:
        params = RequestParams(
            transport="zendriver",
            validated_hosts=lambda h: ValidatedHost(host=h, ip="1.2.3.4"),
        )
        with pytest.raises(ValueError, match="validated_hosts"):
            fetch("https://example.com", request=params)

    def test_default_transport_is_auto(self) -> None:
        assert RequestParams().transport == "auto"

    def test_auto_uses_general_curl_then_browser_fallback(self) -> None:
        assert fetch_mod.resolve_transport("auto") == "curl-then-zendriver"

    def test_auto_uses_curl_for_post(self) -> None:
        assert fetch_mod.resolve_transport("auto", method="POST") == "curl"

    def test_auto_uses_curl_for_get_body(self) -> None:
        with patch.object(
            fetch_mod, "_fetch_with_identity", return_value=b"ok"
        ) as direct:
            body, _ = fetch(
                "https://google.com/api",
                request=RequestParams(json={"query": "value"}),
            )
        assert body == b"ok"
        assert direct.call_args.args[0].params.transport == "curl"

    def test_auto_post_to_learned_domain_uses_curl(self) -> None:
        # A domain learned to require the browser must not override method/body
        # eligibility: an automatic POST to it resolves to curl, not the GET-only
        # zendriver leg (whose construction raises "supports only GET").
        with (
            patch.object(
                fetch_mod.transport_routing,
                "zendriver_domains",
                return_value=frozenset({"walled.example"}),
            ),
            patch.object(
                fetch_mod, "_fetch_with_identity", return_value=b"ok"
            ) as direct,
        ):
            body, _ = fetch(
                "https://walled.example/api",
                request=RequestParams(json={"q": "v"}),
            )
        assert body == b"ok"
        assert direct.call_args.args[0].params.transport == "curl"

    def test_browser_fetch_forwards_url_params_headers_and_cookies(self) -> None:
        result = BrowserResult(body=b"ok", cookies={})
        with (
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch(
                "sagent.lib.web.fetch.fetch.zendriver_backend.fetch_zendriver",
                return_value=result,
            ) as via,
        ):
            fetch(
                "https://google.com/search?hl=en",
                session=FetchSession(cookies={"SID": "session"}),
                request=RequestParams(
                    transport="zendriver",
                    params={"q": "test query"},
                    headers={"X-Test": "yes"},
                    cookies={"CONSENT": "YES+"},
                ),
            )
        assert via.call_args.args[0] == ("https://google.com/search?hl=en&q=test+query")
        assert via.call_args.kwargs["headers"] == {"X-Test": "yes"}
        assert via.call_args.kwargs["cookies"] == {
            "SID": "session",
            "CONSENT": "YES+",
        }
        assert "resolve_host" not in via.call_args.kwargs

    def test_browser_fetch_returns_body_and_warms_session(self) -> None:
        # A browser fetch must return the rendered bytes AND fold the browser's
        # harvested cookies into the returned FetchSession, so a following curl
        # fetch on the same session is warm (the review's key requirement).
        result = BrowserResult(body=b"<html>rendered</html>", cookies={"SID": "xyz"})
        store = Mock()
        with (
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch(
                "sagent.lib.web.fetch.fetch.zendriver_backend.fetch_zendriver",
                return_value=result,
            ) as via,
            patch("sagent.lib.web.profile.ProfileStore.shared", return_value=store),
        ):
            body, session = fetch(
                "https://walled.example/x",
                request=RequestParams(transport="zendriver"),
            )
        assert body == b"<html>rendered</html>"
        assert session.cookies == {"SID": "xyz"}  # session warmed
        assert via.call_count == 1
        store.save.assert_not_called()

    def test_browser_fetch_persists_cookies_to_profile_store(self) -> None:
        result = BrowserResult(body=b"ok", cookies={"cf_clearance": "tok"})
        store = Mock()
        store.load.return_value = None
        with (
            patch.object(fetch_mod, "egress_ip", return_value="5.5.5.5"),
            patch(
                "sagent.lib.web.fetch.fetch.zendriver_backend.fetch_zendriver",
                return_value=result,
            ),
            patch("sagent.lib.web.profile.ProfileStore.shared", return_value=store),
        ):
            fetch(
                "https://walled.example/x",
                request=RequestParams(transport="zendriver"),
            )
        # A fresh (egress, domain) key is saved with the harvested cookies.
        store.save.assert_called_once()
        saved_profile = store.save.call_args.args[2]
        assert saved_profile.cookies == {"cf_clearance": "tok"}
        assert saved_profile.ua


class TestCurlThenZendriverBackend:
    """``transport="curl-then-zendriver"``: curl first, zendriver only on a bot block."""

    def test_curl_then_zendriver_inherits_zendriver_restrictions(self) -> None:
        # curl-then-zendriver may fall back to the browser, so it remains GET-only
        # and body-free.
        with pytest.raises(
            ValueError, match="curl-then-zendriver backend supports only GET"
        ):
            RequestParams(transport="curl-then-zendriver", method="POST")
        with pytest.raises(
            ValueError, match="curl-then-zendriver backend cannot send a request"
        ):
            RequestParams(transport="curl-then-zendriver", data={"a": "1"})
        params = RequestParams(
            transport="curl-then-zendriver",
            validated_hosts=lambda h: ValidatedHost(host=h, ip="1.2.3.4"),
        )
        with pytest.raises(ValueError, match="validated_hosts"):
            fetch("https://example.com", request=params)

    def test_curl_then_zendriver_returns_curl_body_without_touching_browser(
        self,
    ) -> None:
        # When curl succeeds, the browser backend is never invoked.
        with (
            patch.object(fetch_mod, "_send_as", return_value=b"curl body"),
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch(
                "sagent.lib.web.fetch.fetch.zendriver_backend.fetch_zendriver"
            ) as via,
        ):
            body, _ = fetch(
                "https://ok.example/",
                request=RequestParams(transport="curl-then-zendriver"),
            )
        assert body == b"curl body"
        via.assert_not_called()

    def test_curl_then_zendriver_falls_back_to_zendriver_on_bot_block(self) -> None:
        # A curl BotDetectionError triggers the zendriver leg; its body is returned.
        result = BrowserResult(body=b"rendered", cookies={"cf_clearance": "t"})
        with (
            patch.object(fetch_mod, "_send_as", side_effect=CloudflareChallengeError()),
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch.object(
                fetch_mod.transport_routing, "remember_zendriver_domain"
            ) as remember,
            patch(
                "sagent.lib.web.fetch.fetch.zendriver_backend.fetch_zendriver",
                return_value=result,
            ) as via,
        ):
            body, _ = fetch(
                "https://walled.example/",
                request=RequestParams(transport="curl-then-zendriver"),
            )
        assert body == b"rendered"
        assert via.call_count == 1
        remember.assert_called_once_with("walled.example")

    def test_success_body_challenge_falls_back_and_remembers_domain(self) -> None:
        result = BrowserResult(body=b"rendered", cookies={})

        def validate_body(body: bytes) -> None:
            if b"enablejs" in body:
                raise GoogleJavascriptRequiredError("JavaScript required")

        with (
            patch.object(fetch_mod, "_send_as", return_value=b"enablejs"),
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch.object(
                fetch_mod.transport_routing, "remember_zendriver_domain"
            ) as remember,
            patch(
                "sagent.lib.web.fetch.fetch.zendriver_backend.fetch_zendriver",
                return_value=result,
            ) as via,
        ):
            body, _ = fetch(
                "https://walled.example/",
                request=RequestParams(
                    transport="curl-then-zendriver",
                    body_validator=validate_body,
                ),
            )

        assert body == b"rendered"
        via.assert_called_once()
        remember.assert_called_once_with("walled.example")

    def test_bot_block_remembers_domain_even_when_browser_still_fails(self) -> None:
        with (
            patch.object(fetch_mod, "_send_as", side_effect=PuzzleChallengeError()),
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch.object(
                fetch_mod.transport_routing, "remember_zendriver_domain"
            ) as remember,
            patch(
                "sagent.lib.web.fetch.fetch.zendriver_backend.fetch_zendriver",
                side_effect=PuzzleChallengeError("human required"),
            ),
            pytest.raises(PuzzleChallengeError, match="human required"),
        ):
            fetch(
                "https://walled.example/",
                request=RequestParams(transport="curl-then-zendriver"),
            )

        remember.assert_called_once_with("walled.example")

    def test_auto_reuses_persisted_zendriver_fallback(self) -> None:
        domains: set[str] = set()
        result = BrowserResult(body=b"rendered", cookies={})
        with (
            patch.object(
                fetch_mod, "_send_as", side_effect=CloudflareChallengeError()
            ) as via_curl,
            patch.object(fetch_mod, "egress_ip", return_value=None),
            patch.object(
                fetch_mod.transport_routing,
                "zendriver_domains",
                side_effect=lambda: frozenset(domains),
            ),
            patch.object(
                fetch_mod.transport_routing,
                "remember_zendriver_domain",
                side_effect=domains.add,
            ),
            patch(
                "sagent.lib.web.fetch.fetch.zendriver_backend.fetch_zendriver",
                return_value=result,
            ) as via_browser,
        ):
            first, _ = fetch("https://walled.example/")
            second, _ = fetch("https://walled.example/")

        assert first == second == b"rendered"
        assert via_curl.call_count == 1
        assert via_browser.call_count == 2

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
            patch(
                "sagent.lib.web.fetch.fetch.zendriver_backend.fetch_zendriver"
            ) as via,
            pytest.raises(FetchError),
        ):
            fetch("https://x/", request=RequestParams(transport="curl-then-zendriver"))
        via.assert_not_called()


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
