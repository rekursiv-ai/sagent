"""Tests for sagent.lib.web.fetch."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import base64
import gzip
import http.client

import pytest
import zstandard

from sagent.lib.web.errors import (
    FetchError,
)
from sagent.lib.web.fetch import RequestParams, ValidatedHost, fetch
from sagent.lib.web.fetch.stdlib import _open_connection


class TestFetchStdlibPath:
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
        with patch(
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ):
            result, _ = fetch(
                "https://example.com", request=RequestParams(transport="stdlib")
            )
        assert result == b"hello"
        mock_conn.request.assert_called_once()
        assert mock_conn.request.call_args.args[0] == "GET"

    def test_gzipdecompression(self) -> None:
        compressed = gzip.compress(b"hello")
        resp = self._mock_http_response(
            body=compressed, headers=[("content-encoding", "gzip")]
        )
        mock_conn = self._mock_conn(resp)
        with patch(
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ):
            assert (
                fetch("https://example.com", request=RequestParams(transport="stdlib"))[
                    0
                ]
                == b"hello"
            )

    def test_post_with_data(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ):
            fetch(
                "https://example.com",
                request=RequestParams(
                    method="POST", data={"q": "test"}, transport="stdlib"
                ),
            )
        assert mock_conn.request.call_args.args[0] == "POST"
        assert mock_conn.request.call_args.kwargs["body"] == b"q=test"

    def test_post_with_json(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ):
            fetch(
                "https://example.com",
                request=RequestParams(
                    method="POST", json={"key": "value"}, transport="stdlib"
                ),
            )
        assert mock_conn.request.call_args.kwargs["body"] == b'{"key": "value"}'
        headers = mock_conn.request.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"

    def test_data_and_json_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            fetch(
                "https://example.com",
                request=RequestParams(
                    data={"a": "1"}, json={"b": 2}, transport="stdlib"
                ),
            )

    def test_cookies_serialized(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ):
            fetch(
                "https://example.com",
                request=RequestParams(cookies={"a": "1", "b": "2"}, transport="stdlib"),
            )
        headers = mock_conn.request.call_args.kwargs["headers"]
        assert "a=1" in headers["Cookie"]
        assert "b=2" in headers["Cookie"]

    def test_custom_headers_override_defaults(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ):
            fetch(
                "https://example.com",
                request=RequestParams(
                    headers={"User-Agent": "custom"}, transport="stdlib"
                ),
            )
        headers = mock_conn.request.call_args.kwargs["headers"]
        assert headers["User-Agent"] == "custom"

    def test_raw_headers_skip_defaults(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ):
            fetch(
                "https://example.com",
                request=RequestParams(
                    method="POST",
                    data={"q": "test"},
                    headers={"User-Agent": "custom"},
                    raw_headers=True,
                    transport="stdlib",
                ),
            )
        assert mock_conn.request.call_args.kwargs["headers"] == {"User-Agent": "custom"}

    def test_raw_headers_still_add_cookies_and_auth(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ):
            fetch(
                "https://u:p@example.com",
                request=RequestParams(
                    headers={"User-Agent": "custom"},
                    cookies={"a": "1"},
                    raw_headers=True,
                    transport="stdlib",
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
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ) as mock_open:
            fetch(
                "https://u:p@example.com:8443/x",
                request=RequestParams(transport="stdlib"),
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
        with patch(
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ):
            fetch(
                "https://u:p@example.com/",
                request=RequestParams(
                    headers={"Authorization": "Bearer xyz"}, transport="stdlib"
                ),
            )
        headers = mock_conn.request.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer xyz"

    def test_http_error_raises_fetch_error(self) -> None:
        resp = self._mock_http_response(status=403, body=b"Forbidden")
        mock_conn = self._mock_conn(resp)
        with (
            patch(
                "sagent.lib.web.fetch.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError, match="403"),
        ):
            fetch("https://example.com", request=RequestParams(transport="stdlib"))

    def test_timeout_passed(self) -> None:
        mock_conn = self._mock_conn(self._mock_http_response())
        with patch(
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ) as mock_open:
            fetch(
                "https://example.com",
                request=RequestParams(timeout_sec=60, transport="stdlib"),
            )
        assert mock_open.call_args.args[2] == 60


class TestConnectionClosedOnError:
    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> Any:
        # Stdlib path is selected per-call via transport="stdlib", not a global.
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
            patch("sagent.lib.web.fetch.stdlib._open_connection", return_value=conn),
            pytest.raises(FetchError),
        ):
            fetch("https://example.com", request=RequestParams(transport="stdlib"))
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
            patch(
                "sagent.lib.web.fetch.stdlib._open_connection",
                side_effect=[conn1, conn2],
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
        conn1.close.assert_called_once()


class TestFetchStdlibBackend:
    @pytest.fixture(autouse=True)
    def _force_stdlib(self) -> Any:
        # The stdlib backend is http.client-only; each fetch call passes
        # transport="stdlib" so these redirect/error/303/validated-host tests
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
            "sagent.lib.web.fetch.stdlib._open_connection",
            return_value=mock_conn,
        ):
            body, _ = fetch(
                "https://example.com/start", request=RequestParams(transport="stdlib")
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
            "sagent.lib.web.fetch.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://example.com/start",
                request=RequestParams(on_redirect=urls.append, transport="stdlib"),
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
        with patch(
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ):
            _body, session = fetch(
                "https://example.com/", request=RequestParams(transport="stdlib")
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
                "sagent.lib.web.fetch.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(ValueError, match="bad redirect"),
        ):
            fetch(
                "https://example.com",
                request=RequestParams(on_redirect=reject, transport="stdlib"),
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
            "sagent.lib.web.fetch.stdlib._open_connection",
            return_value=mock_conn,
        ):
            result, _ = fetch(
                "https://example.com",
                request=RequestParams(max_redirects=0, transport="stdlib"),
            )
        assert result == b"redirect body"

    def test_plain_get_curl_absent_returns_3xx_body_at_cap(self) -> None:
        # REVE559-001: a plain GET at default max_redirects, curl absent -- once
        # _fetch_simple (urllib) is gone; this routes through fetch_stdlib,
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
            patch(
                "sagent.lib.web.fetch.stdlib._open_connection",
                return_value=mock_conn,
            ),
        ):
            result, _ = fetch(
                "https://example.com", request=RequestParams(transport="stdlib")
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
            "sagent.lib.web.fetch.stdlib._open_connection",
            side_effect=[mock_conn1, mock_conn2],
        ):
            body, _ = fetch(
                "https://example.com/start", request=RequestParams(transport="stdlib")
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
        with patch(
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ):
            body, _ = fetch(
                "https://example.com/base/start",
                request=RequestParams(on_redirect=urls.append, transport="stdlib"),
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
            "sagent.lib.web.fetch.stdlib._open_connection",
            side_effect=[conn_a, conn_b],
        ):
            fetch(
                "https://a.com/submit",
                request=RequestParams(
                    method="POST", data={"x": "1"}, transport="stdlib"
                ),
            )
        sent = conn_b.request.call_args.kwargs["headers"]
        assert sent.get("Origin") != "https://a.com"
        assert sent.get("Origin") == "https://b.com"

    def test_cross_host_redirect_drops_case_varianthost_header(self) -> None:
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
            "sagent.lib.web.fetch.stdlib._open_connection",
            side_effect=[conn_a, conn_b],
        ):
            fetch(
                "https://a.com/start",
                request=RequestParams(headers={"host": "a.com"}, transport="stdlib"),
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
            "sagent.lib.web.fetch.stdlib._open_connection",
            return_value=mock_conn,
        ):
            body, _ = fetch(
                "https://example.com/submit",
                request=RequestParams(
                    method="POST", data={"x": "1"}, transport="stdlib"
                ),
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
                "sagent.lib.web.fetch.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError, match="302"),
        ):
            fetch("https://example.com", request=RequestParams(transport="stdlib"))

    def test_http_error_raises_fetch_error(self) -> None:
        resp = self._mock_http_response(status=404, body=b"not found")
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
            fetch("https://example.com", request=RequestParams(transport="stdlib"))

    def test_error_body_isdecompressed(self) -> None:
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
            patch(
                "sagent.lib.web.fetch.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError) as exc,
        ):
            fetch("https://example.com", request=RequestParams(transport="stdlib"))
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
            patch(
                "sagent.lib.web.fetch.stdlib._open_connection",
                return_value=mock_conn,
            ),
            pytest.raises(FetchError) as exc,
        ):
            fetch("https://example.com", request=RequestParams(transport="stdlib"))
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
            "sagent.lib.web.fetch.stdlib._open_connection",
            return_value=mock_conn,
        ):
            fetch(
                "https://example.com:8443/page",
                request=RequestParams(validated_hosts=_vh, transport="stdlib"),
            )
        assert seen == ["example.com"]

    def test_validatedhost_header_carries_nondefault_port(self) -> None:
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

        with patch(
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ):
            fetch(
                "https://example.com:8443/page",
                request=RequestParams(validated_hosts=_vh, transport="stdlib"),
            )
        assert mock_conn.request.call_args.kwargs["headers"]["Host"] == (
            "example.com:8443"
        )

    def test_validatedhost_header_omitsdefault_port(self) -> None:
        # The converse: a default-port URL must NOT get a ":443" in Host (a real
        # browser omits the default port), else the authority still mismatches.
        resp = self._mock_http_response()
        mock_conn = Mock()
        mock_conn.request = Mock()
        mock_conn.getresponse.return_value = resp

        def _vh(hostname: str) -> ValidatedHost:
            return ValidatedHost(host=hostname, ip="93.184.216.34")

        with patch(
            "sagent.lib.web.fetch.stdlib._open_connection", return_value=mock_conn
        ):
            fetch(
                "https://example.com/page",
                request=RequestParams(validated_hosts=_vh, transport="stdlib"),
            )
        assert mock_conn.request.call_args.kwargs["headers"]["Host"] == "example.com"

    def test_cross_host_redirecthost_header_carries_nondefault_port(self) -> None:
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
            "sagent.lib.web.fetch.stdlib._open_connection",
            side_effect=[conn_a, conn_b],
        ):
            fetch(
                "https://example.com/start",
                request=RequestParams(validated_hosts=_vh, transport="stdlib"),
            )
        assert conn_b.request.call_args.kwargs["headers"]["Host"] == "other.com:8443"


class TestOpenConnection:
    def test_hostname_is_not_bracketed(self) -> None:
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

        with patch("sagent.lib.web.fetch.stdlib.http.client.HTTPSConnection", _Stub):
            _open_connection("https", "example.com", timeout_sec=10)
        assert captured == ["example.com"]

    def test_explicit_stdlib_bypasses_curl(self) -> None:
        response = Mock(spec=http.client.HTTPResponse)
        response.status = 200
        response.read.return_value = b"stdlib"
        response.getheaders.return_value = [("content-encoding", "identity")]
        connection = Mock()
        connection.getresponse.return_value = response
        with (
            patch(
                "sagent.lib.web.fetch.stdlib._open_connection",
                return_value=connection,
            ),
            patch("curl_cffi.requests.request") as curl_request,
        ):
            body, _ = fetch(
                "https://example.com", request=RequestParams(transport="stdlib")
            )
        assert body == b"stdlib"
        curl_request.assert_not_called()
