"""Tests for sagent.lib.web.fetch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock, patch

import importlib
import io
import warnings

from curl_cffi import (
    CurlError,
    CurlInfo,
    CurlOpt,
    requests as cc_requests,
)

import pytest
import zstandard

from sagent.lib.web.errors import (
    FetchError,
)
from sagent.lib.web.fetch import RequestParams, ValidatedHost, fetch
from sagent.lib.web.fetch.curl import _registrable_domain, seed_session_jar
from sagent.lib.web.fetch.test_helpers import StubSession, const_curl_session

import sagent.lib.web.fetch.curl as curl_mod


fetch_mod = importlib.import_module("sagent.lib.web.fetch.fetch")


if TYPE_CHECKING:
    from curl_cffi.requests import Response


# Captured before any fixture stubs it, so the pool-locking tests can invoke the
# real curl_session.
_REAL_CURL_SESSION = curl_mod.curl_session


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
                request=RequestParams(validated_hosts=_vh, transport="curl"),
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
        stub = StubSession()
        resp = self._mock_response(content=b"ok")
        with (
            patch("curl_cffi.requests.request", return_value=resp) as mock_req,
            patch.object(fetch_mod, "curl_session", const_curl_session(stub)),
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
        # curl path returns the final 3xx body (matching fetch_stdlib's
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

    def test_error_status_raises_withdecompressed_body(self) -> None:
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
                request=RequestParams(validated_hosts=_vh, transport="curl"),
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
            patch("sagent.lib.web.fetch.fetch.time.sleep"),
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
            patch("sagent.lib.web.fetch.fetch.time.sleep"),
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


class TestCurlSessionPoolLocking:
    """Every mutation of the curl session pool holds its pool lock."""

    @pytest.fixture(autouse=True)
    def _real_curl_session(self, monkeypatch: Any) -> None:
        # The module isolate_profiles fixture stubs curl_session; restore the
        # real function so these tests exercise its actual locking.
        monkeypatch.setattr(fetch_mod, "curl_session", _REAL_CURL_SESSION)

    def test_curl_session_holds_pool_lock(self, monkeypatch: Any) -> None:
        acquired: list[str] = []
        real_lock = curl_mod._curl_lock

        class _Instrumented:
            def __enter__(self) -> None:
                acquired.append("enter")
                real_lock.acquire()

            def __exit__(self, *_a: object) -> None:
                real_lock.release()

        monkeypatch.setattr(curl_mod, "_curl_lock", _Instrumented())
        monkeypatch.setattr(curl_mod, "_curl_sessions", {})
        with patch("curl_cffi.requests.Session", return_value=Mock()):
            curl_mod.curl_session("1.2.3.4", "x.com", "chrome")
        assert acquired, "curl_session mutated the pool without _curl_lock"

    def test_close_curl_session_holds_pool_lock(self, monkeypatch: Any) -> None:
        acquired: list[str] = []
        real_lock = curl_mod._curl_lock

        class _Instrumented:
            def __enter__(self) -> None:
                acquired.append("enter")
                real_lock.acquire()

            def __exit__(self, *_a: object) -> None:
                real_lock.release()

        monkeypatch.setattr(curl_mod, "_curl_lock", _Instrumented())
        monkeypatch.setattr(curl_mod, "_curl_sessions", {})
        curl_mod.close_curl_session("1.2.3.4", "x.com", "chrome")  # absent: no-op
        assert acquired, "close_curl_session mutated the pool without _curl_lock"

    def test_close_sessions_except_preserves_current_egress(
        self, monkeypatch: Any
    ) -> None:
        current = Mock()
        stale = Mock()
        monkeypatch.setattr(
            curl_mod,
            "_curl_sessions",
            {
                ("1.2.3.4", "x.com", "chrome"): current,
                ("5.6.7.8", "x.com", "chrome"): stale,
            },
        )

        curl_mod.close_curl_sessions_except("1.2.3.4")

        assert list(curl_mod._curl_sessions) == [("1.2.3.4", "x.com", "chrome")]
        current.close.assert_not_called()
        stale.close.assert_called_once_with()


class TestSeedSessionJar:
    def test_secure_prefixed_cookie_seeded_without_warning(self) -> None:
        # RFC 6265bis: a __Secure-/__Host- prefixed cookie is only valid Secure;
        # seeding it without secure=True made curl_cffi emit a CurlCffiWarning
        # (which the live Google-search integration path surfaced as a failure).
        session = cast("cc_requests.Session[Response]", cc_requests.Session())
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                seed_session_jar(
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
