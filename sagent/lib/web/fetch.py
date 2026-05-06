"""Unified HTTP fetch for ``sagent.lib.web``.

All HTTP in ``sagent.lib.web`` flows through :func:`fetch`. Sync,
urllib-based, with transparent decompression and optional retry.

Usage::

    from sagent.lib.web import fetch

    # Simple (99% case):
    body = fetch(url)
    html = fetch(url).decode("utf-8")

    # Track final URL via callback:
    final_url = url
    def track(redirect_url: str) -> None:
        nonlocal final_url
        final_url = redirect_url
    body = fetch(url, on_redirect=track)

    # Connection reuse:
    http_conn = None
    for url in urls:
        body, http_conn = fetch(
            url, return_connection=True, http_conn=http_conn,
        )
    http_conn.close()
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, overload, override
from urllib.parse import urlencode, urlparse

import gzip
import http.client
import io
import json as json_mod
import logging
import random
import ssl
import time
import urllib.error
import urllib.request
import zlib

import brotli
import zstandard

from sagent.lib.json import JSONValue


__all__ = [
    "FetchError",
    "fetch",
]

logger = logging.getLogger(__name__)

HTTPConn = http.client.HTTPConnection | http.client.HTTPSConnection


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatedHost:
    host: str
    ip: str


ValidatedHosts = Callable[[str], ValidatedHost]

# Chrome request signature -- (major version, sec-ch-ua brand list,
# platform). UA, sec-ch-ua, and sec-ch-ua-platform must agree; servers
# that observe these headers flag drift between them. Hardcoding
# "Linux" (rather than platform.system()) keeps the signature
# reproducible across machines and matches the UA token.
_CHROME_SIGNATURE: tuple[str, str, str] = (
    "125",
    '"Chromium";v="125", "Not.A/Brand";v="24", "Google Chrome";v="125"',
    '"Linux"',
)
_CHROME_UA = (
    f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    f"(KHTML, like Gecko) Chrome/{_CHROME_SIGNATURE[0]}.0.0.0 Safari/537.36"
)
_NAV_ACCEPT = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,"
    "image/avif,image/webp,image/apng,*/*;q=0.8,"
    "application/signed-exchange;v=b3;q=0.7"
)


def _build_headers(
    *,
    method: str,
    url: str,
    content_type: str | None,
    extra: dict[str, str] | None,
) -> dict[str, str]:
    """Build canonical-order Chrome request headers.

    Header order, names, and the presence of ``sec-ch-ua`` /
    ``Sec-Fetch-*`` are observable to servers; a minimal HTTP/1.1
    request that omits modern browser headers is materially different
    from typical user traffic, and many web gateways return 403 to
    such requests. The layout below mirrors a real Chrome 125 wire
    shape so that legitimate fetches are not mistaken for malformed
    clients:
      - GET/HEAD: top-level navigation (Sec-Fetch-Mode=navigate,
        Upgrade-Insecure-Requests, Accept=text/html...).
      - POST: fetch/XHR (Sec-Fetch-Mode=cors, Origin set, Accept=*/*,
        Content-Type spliced between Accept and Sec-Fetch-*).

    Host is omitted -- http.client.putrequest auto-adds it first on the
    wire, and the connection path overrides explicitly when
    validated_hosts forces an SNI/connect-IP split.

    Content-Length is also omitted -- http.client._send_request adds it
    automatically right after Host when body is present, which matches
    Chrome's POST wire order.
    """
    h: dict[str, str] = {
        "Connection": "keep-alive",
        "sec-ch-ua": _CHROME_SIGNATURE[1],
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": _CHROME_SIGNATURE[2],
    }
    if method in ("GET", "HEAD"):
        h["Upgrade-Insecure-Requests"] = "1"
        h["User-Agent"] = _CHROME_UA
        h["Accept"] = _NAV_ACCEPT
        h["Sec-Fetch-Site"] = "none"
        h["Sec-Fetch-Mode"] = "navigate"
        h["Sec-Fetch-User"] = "?1"
        h["Sec-Fetch-Dest"] = "document"
    else:
        h["User-Agent"] = _CHROME_UA
        h["Accept"] = "*/*"
        if content_type:
            h["Content-Type"] = content_type
        parsed = urlparse(url)
        h["Origin"] = f"{parsed.scheme}://{parsed.netloc}"
        h["Sec-Fetch-Site"] = "cross-site"
        h["Sec-Fetch-Mode"] = "cors"
        h["Sec-Fetch-Dest"] = "empty"
    h["Accept-Encoding"] = "gzip, deflate, br, zstd"
    h["Accept-Language"] = "en-US,en;q=0.9"
    if extra:
        # Caller wins; dict.update preserves slot for existing keys and
        # appends new ones at the end.
        h.update(extra)
    return h


_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
_DEFAULT_MAX_REDIRECTS = 10


class FetchError(Exception):
    """HTTP request returned a non-success status code.

    Attributes:
      url: Requested URL.
      status: HTTP status code.
      headers: Response headers.
      body: Response body bytes (for debugging error pages).

    """

    def __init__(
        self,
        url: str,
        status: int,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        self.url = url
        self.status = status
        self.headers = headers
        self.body = body
        super().__init__(f"HTTP {status}: {url}")


@overload
def fetch(
    url: str,
    *,
    method: str = ...,
    params: dict[str, str | int] | None = ...,
    data: dict[str, str] | None = ...,
    json: JSONValue = ...,
    headers: dict[str, str] | None = ...,
    cookies: dict[str, str] | None = ...,
    retries: int = ...,
    timeout_sec: float = ...,
    max_redirects: int = ...,
    on_redirect: Callable[[str], None] | None = ...,
    validated_hosts: ValidatedHosts | None = ...,
    return_connection: Literal[False] = ...,
) -> bytes: ...


@overload
def fetch(
    url: str,
    *,
    method: str = ...,
    params: dict[str, str | int] | None = ...,
    data: dict[str, str] | None = ...,
    json: JSONValue = ...,
    headers: dict[str, str] | None = ...,
    cookies: dict[str, str] | None = ...,
    retries: int = ...,
    timeout_sec: float = ...,
    max_redirects: int = ...,
    on_redirect: Callable[[str], None] | None = ...,
    validated_hosts: ValidatedHosts | None = ...,
    return_connection: Literal[True],
    http_conn: HTTPConn | None = ...,
) -> tuple[bytes, HTTPConn]: ...


def fetch(
    url: str,
    *,
    method: str = "GET",
    params: dict[str, str | int] | None = None,
    data: dict[str, str] | None = None,
    json: JSONValue = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    retries: int = 0,
    timeout_sec: float = 30,
    max_redirects: int = _DEFAULT_MAX_REDIRECTS,
    on_redirect: Callable[[str], None] | None = None,
    validated_hosts: ValidatedHosts | None = None,
    return_connection: bool = False,
    http_conn: HTTPConn | None = None,
) -> bytes | tuple[bytes, HTTPConn]:
    """Fetch a URL and return the response body.

    Args:
      url: Fully-qualified URL (http or https).
      method: HTTP method (GET, POST, etc.).
      params: Query parameters appended to the URL.
      data: Form data dict, sent as application/x-www-form-urlencoded.
        Mutually exclusive with ``json``.
      json: JSON-serializable body. Sent as application/json.
        Mutually exclusive with ``data``.
      headers: Extra headers; merged with defaults (caller wins).
      cookies: Cookie name-value pairs, serialized to a Cookie header.
      retries: Number of retry attempts for transient failures.
      timeout_sec: Socket timeout in seconds.
      max_redirects: Maximum redirects to follow. 0 to disable.
      on_redirect: Called with the redirect target URL before following.
        Raise to abort the redirect.
      validated_hosts: Optional resolver that returns a validated IP for each host.
      return_connection: If True, return ``(bytes, HTTPConn)`` for reuse.
      http_conn: Existing connection to reuse (same-host only).

    Returns:
      body: Response bytes (default).
      body_conn: ``(bytes, HTTPConn)`` when ``return_connection=True``.

    Raises:
      FetchError: On non-success HTTP status after exhausting retries.
      ValueError: On unsupported or corrupt Content-Encoding.

    """
    if data is not None and json is not None:
        raise ValueError("'data' and 'json' are mutually exclusive.")
    if params:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{urlencode(params)}"
    body_bytes: bytes | None = None
    body_content_type: str | None = None
    if data is not None:
        body_bytes = urlencode(data).encode()
        body_content_type = "application/x-www-form-urlencoded"
    elif json is not None:
        body_bytes = json_mod.dumps(json).encode()
        body_content_type = "application/json"
    merged = _build_headers(
        method=method,
        url=url,
        content_type=body_content_type,
        extra=headers,
    )
    if cookies:
        merged["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())

    use_conn_path = (
        return_connection
        or http_conn is not None
        or max_redirects != _DEFAULT_MAX_REDIRECTS
        or on_redirect is not None
        or validated_hosts is not None
    )
    last_err: Exception | None = None
    for attempt in range(1 + retries):
        try:
            if use_conn_path:
                result, result_conn = _fetch_connection(
                    url,
                    method=method,
                    headers=merged,
                    body=body_bytes,
                    timeout_sec=timeout_sec,
                    max_redirects=max_redirects,
                    on_redirect=on_redirect,
                    validated_hosts=validated_hosts,
                    http_conn=http_conn,
                )
                if return_connection:
                    return result, result_conn
                result_conn.close()
                return result
            return _fetch_simple(
                url,
                method=method,
                headers=merged,
                body=body_bytes,
                timeout_sec=timeout_sec,
            )
        except FetchError as e:
            last_err = e
            http_conn = None
            if e.status not in _RETRYABLE_STATUSES or attempt == retries:
                raise
            delay_sec = _backoff_delay(attempt, e.headers)
            logger.debug(
                "fetch %s → %d, retry in %.1fs",
                url,
                e.status,
                delay_sec,
            )
            time.sleep(delay_sec)
        except (OSError, TimeoutError) as e:
            last_err = e
            http_conn = None
            if attempt == retries:
                raise
            delay_sec = _backoff_delay(attempt, {})
            logger.debug(
                "fetch %s failed: %s, retry in %.1fs",
                url,
                e,
                delay_sec,
            )
            time.sleep(delay_sec)
    assert last_err is not None
    raise last_err


def _backoff_delay(attempt: int, headers: dict[str, str]) -> float:
    """Exponential backoff with jitter; respects Retry-After."""
    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return min(float(retry_after), 30)
        except ValueError:
            pass
    delay = min(1.0 * (2**attempt), 30)
    return delay + random.uniform(0, delay * 0.5)  # noqa: S311 -- jitter, not security


def _decompress(body: bytes, encoding: str) -> bytes:
    """Decompress response body based on Content-Encoding.

    Raises:
      ValueError: On unknown encoding or decompression failure.

    """
    enc = encoding.strip().lower()
    if enc in ("", "identity"):
        return body
    try:
        if enc == "gzip":
            return gzip.decompress(body)
        if enc == "deflate":
            return zlib.decompress(body)
        if enc == "br":
            return brotli.decompress(body)
        if enc == "zstd":
            # stream_reader handles frames without an embedded size,
            # which `.decompress()` rejects. Servers (e.g. Cloudflare)
            # commonly emit such frames.
            return zstandard.ZstdDecompressor().stream_reader(io.BytesIO(body)).read()
    except (OSError, zlib.error, brotli.error, zstandard.ZstdError) as e:
        raise ValueError(f"Decompression failed ({enc}): {e}") from None
    raise ValueError(f"Unknown Content-Encoding: {enc!r}")


def _join_headers(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Lowercase and merge header pairs.

    Duplicate names are joined with ", " per RFC 9110 SS 5.3.
    """
    out: dict[str, str] = {}
    for k, v in pairs:
        key = k.lower()
        if key in out:
            out[key] = f"{out[key]}, {v}"
        else:
            out[key] = v
    return out


def _fetch_simple(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout_sec: float,
) -> bytes:
    """Simple path: urllib.request.urlopen, no connection reuse."""
    request = urllib.request.Request(  # noqa: S310 -- caller-supplied URL; validation at trust boundary
        url,
        headers=headers,
        method=method,
        data=body,
    )
    try:
        response = urllib.request.urlopen(  # noqa: S310 -- caller-supplied URL; validation at trust boundary
            request,
            timeout=timeout_sec,
        )
    except urllib.error.HTTPError as e:
        resp_body = e.read()
        raise FetchError(
            url,
            e.code,
            _join_headers(e.headers.items()),
            resp_body,
        ) from None
    raw = response.read()
    encoding = response.headers.get("Content-Encoding", "identity")
    return _decompress(raw, encoding)


class _ValidatedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        *,
        server_hostname: str,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        super().__init__(host, timeout=timeout, context=context)
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


def _hostname(netloc: str) -> str:
    parsed = urlparse(f"//{netloc}")
    return parsed.hostname or netloc


def _bracket_ipv6(host: str) -> str:
    """Wrap an IPv6 literal in brackets for http.client compatibility.

    http.client splits the host string on the last ':' to extract a
    port, so a bare IPv6 address like ``2606:4700::6810:7c60`` is
    misparsed as ``host=2606:4700::6810`` + ``port=7c60``. Bracketing
    avoids the heuristic. Hostnames and IPv4 addresses pass through.
    """
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _open_connection(
    scheme: str,
    host: str,
    timeout_sec: float,
    resolved_ip: str = "",
) -> HTTPConn:
    """Open a new HTTP or HTTPS connection."""
    connect_host = _bracket_ipv6(resolved_ip or host)
    if scheme == "https":
        ctx = ssl.create_default_context()
        if resolved_ip:
            return _ValidatedHTTPSConnection(
                connect_host,
                server_hostname=_hostname(host),
                timeout=timeout_sec,
                context=ctx,
            )
        return http.client.HTTPSConnection(
            connect_host,
            timeout=timeout_sec,
            context=ctx,
        )
    return http.client.HTTPConnection(connect_host, timeout=timeout_sec)


def _fetch_connection(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout_sec: float,
    max_redirects: int,
    on_redirect: Callable[[str], None] | None,
    validated_hosts: ValidatedHosts | None,
    http_conn: HTTPConn | None,
) -> tuple[bytes, HTTPConn]:
    """Connection path: http.client with manual redirect following."""
    parsed = urlparse(url)
    scheme = parsed.scheme
    host = parsed.netloc
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    validated = validated_hosts(host) if validated_hosts is not None else None
    connect_host = validated.ip if validated is not None else ""
    request_headers = headers
    if validated is not None:
        # Host first: real browsers and http.client's own auto-generated
        # Host header both place it before User-Agent/Accept/etc. Servers
        # that observe header order return 403 when Host is trailing.
        request_headers = {"Host": validated.host, **headers}

    if (
        http_conn is not None
        and http_conn.host == (connect_host or host)
        and isinstance(http_conn, http.client.HTTPSConnection) == (scheme == "https")
    ):
        raw_conn = http_conn
    else:
        raw_conn = _open_connection(scheme, host, timeout_sec, resolved_ip=connect_host)

    current_url = url
    remaining = max_redirects
    while True:
        raw_conn.request(method, path, body=body, headers=request_headers)
        response = raw_conn.getresponse()
        resp_headers = _join_headers(response.getheaders())

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
            redir = urlparse(location)
            redir_scheme = redir.scheme or scheme
            redir_host = redir.netloc or host
            redirect_url = (
                location
                if redir.netloc
                else (
                    f"{redir_scheme}://{redir_host}{redir.path or '/'}"
                    + (f"?{redir.query}" if redir.query else "")
                )
            )
            if on_redirect is not None:
                on_redirect(redirect_url)
            if redir_host != host or redir_scheme != scheme:
                raw_conn.close()
                scheme = redir_scheme
                host = redir_host
                validated = (
                    validated_hosts(host) if validated_hosts is not None else None
                )
                connect_host = validated.ip if validated is not None else ""
                request_headers = headers
                if validated is not None:
                    request_headers = {"Host": validated.host, **headers}
                raw_conn = _open_connection(
                    scheme,
                    host,
                    timeout_sec,
                    resolved_ip=connect_host,
                )
            path = redir.path or "/"
            if redir.query:
                path = f"{path}?{redir.query}"
            current_url = redirect_url
            if response.status == 303:
                method = "GET"
                body = None
            continue

        raw_body = response.read()
        if response.status >= 400:
            raise FetchError(
                current_url,
                response.status,
                resp_headers,
                raw_body,
            )
        encoding = resp_headers.get("content-encoding", "identity")
        return _decompress(raw_body, encoding), raw_conn
