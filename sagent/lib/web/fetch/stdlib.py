"""http.client reference transport for :mod:`sagent.lib.web.fetch`."""

from __future__ import annotations

from collections.abc import Callable
from typing import override
from urllib.parse import urljoin, urlparse

import http.client
import ssl

from sagent.lib.web.challenge import classify_http_error
from sagent.lib.web.errors import FetchError
from sagent.lib.web.fetch.common import (
    Observer,
    ValidatedHosts,
    apply_redirect,
    bracket_ipv6,
    decompress,
    decompress_error_body,
    host_header,
    join_headers,
)


__all__ = ["fetch_stdlib"]

HTTPConn = http.client.HTTPConnection | http.client.HTTPSConnection


class _ValidatedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        *,
        port: int | None = None,
        server_hostname: str,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(host, port=port, timeout=timeout, context=context)
        self._server_hostname = server_hostname
        self._ssl_context = context

    @override
    def connect(self) -> None:
        http.client.HTTPConnection.connect(self)
        assert self.sock is not None
        self.sock = self._ssl_context.wrap_socket(
            self.sock,
            server_hostname=self._server_hostname,
        )


def _open_connection(
    scheme: str,
    hostname: str,
    timeout_sec: float,
    *,
    port: int | None = None,
    resolved_ip: str = "",
) -> HTTPConn:
    """Open a new HTTP/HTTPS connection; pin to ``resolved_ip`` when given."""
    connect_host = bracket_ipv6(resolved_ip or hostname)
    if scheme == "https":
        ctx = ssl.create_default_context()
        if resolved_ip:
            return _ValidatedHTTPSConnection(
                connect_host,
                port=port,
                server_hostname=hostname,
                timeout=timeout_sec,
                context=ctx,
            )
        return http.client.HTTPSConnection(
            connect_host,
            port=port,
            timeout=timeout_sec,
            context=ctx,
        )
    return http.client.HTTPConnection(connect_host, port=port, timeout=timeout_sec)


def fetch_stdlib(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout_sec: float,
    max_redirects: int,
    impersonate: str,
    on_redirect: Callable[[str], None] | None,
    on_response: Observer | None,
    validated_hosts: ValidatedHosts | None,
    session: object | None = None,
) -> bytes:
    """Stdlib transport: http.client with manual redirect following.

    A drop-in peer of :func:`fetch_curl` with the identical signature, so the
    core dispatcher can select either transport. This backend has no TLS
    impersonation and no pooled connection, so ``impersonate`` and ``session``
    are accepted for interface parity and ignored; the coherent Chrome header set
    is instead hand-built upstream in :func:`_build_headers`.
    """
    del impersonate, session  # No impersonation or connection pooling here.
    # The connection is owned entirely here: opened locally and closed in the
    # finally on every exit (success, HTTP error, redirect/decompress failure),
    # so no socket leaks. Nothing escapes to the caller.
    parsed = urlparse(url)
    scheme = parsed.scheme
    hostname = parsed.hostname or parsed.netloc
    port = parsed.port
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    validated = validated_hosts(hostname) if validated_hosts is not None else None
    connect_host = validated.ip if validated is not None else ""
    request_headers = headers
    if validated is not None:
        # Host first: real browsers and http.client's own auto-generated
        # Host header both place it before User-Agent/Accept/etc. Servers
        # that observe header order return 403 when Host is trailing. The
        # resolver returns the bare host (its contract), so host_header
        # re-appends any non-default port.
        request_headers = {
            "Host": host_header(validated.host, port, scheme),
            **headers,
        }

    raw_conn = _open_connection(
        scheme, hostname, timeout_sec, port=port, resolved_ip=connect_host
    )
    try:
        current_url = url
        remaining = max_redirects
        while True:
            raw_conn.request(method, path, body=body, headers=request_headers)
            response = raw_conn.getresponse()
            resp_headers = join_headers(response.getheaders())
            if on_response is not None:
                on_response(response.status, resp_headers, current_url)

            is_redirect = response.status in (301, 302, 303, 307, 308)
            if is_redirect and remaining > 0:
                remaining -= 1
                location = resp_headers.get("location")
                response.read()
                if not location:
                    raise FetchError(
                        current_url,
                        response.status,
                        resp_headers,
                        b"Redirect with no Location header",
                    )
                # RFC 3986 relative resolution against the current URL: handles
                # absolute, scheme-relative (//host/p), and path-relative (both
                # "/p" and bare "p") Locations without corrupting the host.
                redirect_url = urljoin(current_url, location)
                redir = urlparse(redirect_url)
                redir_scheme = redir.scheme or scheme
                if on_redirect is not None:
                    on_redirect(redirect_url)
                redir_parsed = urlparse(redirect_url)
                redir_hostname = redir_parsed.hostname or hostname
                redir_port = redir_parsed.port
                # Per-hop transform (Origin rewrite, 303 -> bodyless GET) via the
                # ONE shared helper, so the stdlib and curl paths cannot drift.
                # Applied to request_headers (which carries any validated Host).
                request_headers, method, body = apply_redirect(
                    current_url,
                    request_headers,
                    method,
                    body,
                    response.status,
                    redirect_url,
                )
                if (
                    redir_hostname != hostname
                    or redir_port != port
                    or redir_scheme != scheme
                ):
                    raw_conn.close()
                    scheme = redir_scheme
                    hostname = redir_hostname
                    port = redir_port
                    validated = (
                        validated_hosts(hostname)
                        if validated_hosts is not None
                        else None
                    )
                    connect_host = validated.ip if validated is not None else ""
                    # New host: replace the Host header (drop any prior first).
                    # HTTP field names are case-insensitive, so drop any casing.
                    request_headers = {
                        k: v for k, v in request_headers.items() if k.lower() != "host"
                    }
                    if validated is not None:
                        request_headers = {
                            "Host": host_header(validated.host, port, scheme),
                            **request_headers,
                        }
                    raw_conn = _open_connection(
                        scheme,
                        hostname,
                        timeout_sec,
                        port=port,
                        resolved_ip=connect_host,
                    )
                path = redir.path or "/"
                if redir.query:
                    path = f"{path}?{redir.query}"
                current_url = redirect_url
                continue

            raw_body = response.read()
            if response.status >= 400:
                raise classify_http_error(
                    current_url,
                    response.status,
                    resp_headers,
                    decompress_error_body(raw_body, resp_headers),
                )
            encoding = resp_headers.get("content-encoding", "identity")
            return decompress(raw_body, encoding)
    finally:
        raw_conn.close()
