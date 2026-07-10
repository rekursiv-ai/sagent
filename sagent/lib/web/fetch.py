"""Unified HTTP fetch for ``sagent.lib.web``.

All HTTP in ``sagent.lib.web`` flows through :func:`fetch`. Sync, with
transparent decompression and optional retry. The transport is curl_cffi
(a Chrome-compatible TLS/HTTP-2 profile) when installed, else stdlib
``http.client``.

Usage::

    from sagent.lib.web.fetch import fetch

    # Simple (99% case):
    body = fetch(url)
    html = fetch(url).decode("utf-8")

    # Track final URL via callback:
    final_url = url
    def track(redirect_url: str) -> None:
        nonlocal final_url
        final_url = redirect_url
    body = fetch(url, on_redirect=track)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, override
from urllib.parse import unquote, urlencode, urljoin, urlparse

import base64
import gzip
import http.client
import io
import json as json_lib
import logging
import random
import ssl
import time
import zlib

import brotli
import zstandard

from sagent.lib.custom_json import JSONValue
from sagent.lib.web.errors import FetchError, classify_http_error


# curl_cffi presents a Chrome-compatible TLS/HTTP-2 profile; when installed it is
# the preferred transport for compatibility -- modern CDNs and servers negotiate
# TLS/HTTP-2 the way current browsers do, and a plain stdlib client (older TLS
# extension ordering, HTTP/1.1) is often served a 403 or a degraded response.
# Matching a mainstream browser's profile is what a normal client looks like. It
# is an OPTIONAL
# dependency, imported once at module load (guarded) rather than inline per
# function: inline needs a PLC0415 suppression, and ``wrapt.lazy_import`` is
# unusable here -- ``CurlError`` is caught in ``except`` clauses and a
# lazy-import proxy is not a real exception class. Under TYPE_CHECKING the
# symbols import unconditionally so the checkers
# see the REAL curl types (no ``None`` poisoning downstream); at runtime the
# guarded import binds them only when present and ``_HAVE_CURL`` gates every use.
if TYPE_CHECKING:
    from curl_cffi import (
        Curl,
        CurlError,
        CurlInfo,
        CurlOpt,
        requests as cc_requests,
    )
    from curl_cffi.requests.session import HttpMethod

    _HAVE_CURL = True
else:
    try:
        from curl_cffi import (
            Curl,
            CurlError,
            CurlInfo,
            CurlOpt,
            requests as cc_requests,
        )

        _HAVE_CURL = True
    except ImportError:  # pragma: no cover -- exercised via _HAVE_CURL=False path
        _HAVE_CURL = False


__all__ = [
    "fetch",
]

logger = logging.getLogger(__name__)


HTTPConn = http.client.HTTPConnection | http.client.HTTPSConnection


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatedHost:
    host: str
    ip: str


ValidatedHosts = Callable[[str], ValidatedHost]

# Chrome request signature. UA, sec-ch-ua, and sec-ch-ua-platform must agree;
# servers that observe these headers may reject requests where they disagree.
# Everything is derived from a single major version so a bump touches one
# constant, and "Linux" is hardcoded (rather than platform.system()) to keep
# the signature reproducible across machines and matching the UA token.
_CHROME_MAJOR = "125"  # config-globals: ignore -- browser version literal.
_CHROME_PLATFORM = '"Linux"'  # config-globals: ignore -- browser platform literal.
_CHROME_SEC_CH_UA = (
    f'"Chromium";v="{_CHROME_MAJOR}", "Not.A/Brand";v="24", '
    f'"Google Chrome";v="{_CHROME_MAJOR}"'
)
_CHROME_UA = (
    f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    f"(KHTML, like Gecko) Chrome/{_CHROME_MAJOR}.0.0.0 Safari/537.36"
)
_NAV_ACCEPT = (  # config-globals: ignore -- HTTP Accept header literal.
    "text/html,application/xhtml+xml,application/xml;q=0.9,"
    "image/avif,image/webp,image/apng,*/*;q=0.8,"
    "application/signed-exchange;v=b3;q=0.7"
)


_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


def fetch(
    url: str,
    *,
    method: str = "GET",
    params: dict[str, str | int] | None = None,
    data: dict[str, str] | None = None,
    json: JSONValue = None,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    raw_headers: bool = False,
    retries: int = 0,
    timeout_sec: float = 30,
    max_redirects: int = 10,
    on_redirect: Callable[[str], None] | None = None,
    validated_hosts: ValidatedHosts | None = None,
) -> bytes:
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
      raw_headers: Send exactly ``headers`` plus cookies and auth; skip defaults.
      retries: Number of retry attempts for transient failures.
      timeout_sec: Socket timeout in seconds.
      max_redirects: Maximum redirects to follow. 0 to disable.
      on_redirect: Called with the redirect target URL before following.
        Raise to abort the redirect.
      validated_hosts: Optional resolver returning a validated IP per hostname.
        It receives the bare hostname (never ``host:port``) and must be pure:
        a given hostname always resolves to the same IP for the lifetime of a
        call. A redirect that stays on the same hostname, port, and scheme
        reuses the prior resolution without re-invoking the resolver.

    Returns:
      body: Response bytes.

    Raises:
      FetchError: On non-success HTTP status after exhausting retries.
      ValueError: On unsupported or corrupt Content-Encoding.

    Note:
      ``*_PROXY`` environment variables are intentionally ignored. Egress is
      controlled explicitly via ``validated_hosts`` (connect-IP pinning for
      SSRF); an implicit env proxy would fight that pin, and no caller in this
      package wants one.

    """
    if data is not None and json is not None:
        raise ValueError("'data' and 'json' are mutually exclusive.")
    if retries < 0:
        raise ValueError(f"'retries' must be >= 0, got {retries}.")
    if max_redirects < 0:
        raise ValueError(f"'max_redirects' must be >= 0, got {max_redirects}.")
    if timeout_sec <= 0:
        raise ValueError(f"'timeout_sec' must be > 0, got {timeout_sec}.")
    url, basic_auth = _split_userinfo(url)
    if params:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{urlencode(params)}"
    body_bytes: bytes | None = None
    body_content_type: str | None = None
    if data is not None:
        body_bytes = urlencode(data).encode()
        body_content_type = "application/x-www-form-urlencoded"
    elif json is not None:
        body_bytes = json_lib.dumps(json).encode()
        body_content_type = "application/json"
    merged = _build_headers(
        method=method,
        url=url,
        content_type=body_content_type,
        extra=headers,
        raw_headers=raw_headers,
    )
    if basic_auth is not None:
        merged.setdefault("Authorization", basic_auth)
    if cookies:
        merged["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())

    # Prefer curl when installed (Chrome-compatible TLS/HTTP-2), else the
    # stdlib http.client path. Both transports own their own connection and
    # return bytes; no connection escapes to the caller, so no lifecycle is
    # split across functions (the class of leak that a returnable conn invited).
    use_curl = _HAVE_CURL
    for attempt in range(1 + retries):
        try:
            if use_curl:
                return _fetch_curl(
                    url,
                    method=method,
                    headers=merged,
                    body=body_bytes,
                    timeout_sec=timeout_sec,
                    max_redirects=max_redirects,
                    on_redirect=on_redirect,
                    validated_hosts=validated_hosts,
                )
            return _fetch_connection(
                url,
                method=method,
                headers=merged,
                body=body_bytes,
                timeout_sec=timeout_sec,
                max_redirects=max_redirects,
                on_redirect=on_redirect,
                validated_hosts=validated_hosts,
            )
        except FetchError as e:
            # status 0 is the transport-failure sentinel (a curl CurlError, or a
            # connection/TLS failure wrapped by a transport) -- retryable like the
            # OSError below, which the stdlib path raises for the same class of
            # failure. Without this the two transports disagree on `retries=`.
            retryable = e.status in _RETRYABLE_STATUSES or e.status == 0
            if not retryable or attempt == retries:
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
    # The loop returns on success and re-raises on the final attempt, so this
    # is unreachable; it exists only to satisfy the type checker.
    raise AssertionError("retry loop exited without returning or raising")


def _build_headers(
    *,
    method: str,
    url: str,
    content_type: str | None,
    extra: dict[str, str] | None,
    raw_headers: bool,
) -> dict[str, str]:
    """Build canonical-order Chrome request headers."""
    # Header order/names and sec-ch-ua / Sec-Fetch-* are observable, and a
    # minimal HTTP/1.1 request that omits the headers a modern browser sends is
    # often served a 403 or a degraded response. Sending the same header set a
    # current browser does keeps a legitimate request compatible. The layout
    # mirrors a real Chrome wire shape -- GET/HEAD as top-level navigation, POST
    # as fetch/XHR. Host and
    # Content-Length are omitted: http.client auto-adds both first on the wire
    # (the connection path overrides Host when validated_hosts splits SNI/IP).
    if raw_headers:
        return dict(extra or {})
    h: dict[str, str] = {
        "Connection": "keep-alive",
        "sec-ch-ua": _CHROME_SEC_CH_UA,
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": _CHROME_PLATFORM,
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


def _backoff_delay(attempt: int, headers: dict[str, str]) -> float:
    """Return the backoff delay in seconds, honoring any ``Retry-After``."""
    # Only the HTTP-status retry path supplies headers; network-error retries
    # pass an empty mapping and always fall through to the computed backoff.
    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return min(float(retry_after), 30)
        except ValueError:
            pass
    delay = min(1.0 * (2**attempt), 30)
    return delay + random.uniform(0, delay * 0.5)  # noqa: S311 -- jitter, not security


def _decompress(body: bytes, encoding: str) -> bytes:
    """Decompress a response body per Content-Encoding; raise ValueError if bad."""
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


def _decompress_error_body(body: bytes, headers: dict[str, str]) -> bytes:
    """Decompress an ERROR response body best-effort, never raising."""
    # Error pages are compressed like any success body (Cloudflare serves its
    # challenge pages zstd/br), so a raw FetchError.body is undecodable garbage
    # and a caller cannot tell a challenge from a genuine 404. This must NOT
    # raise: an undecodable body must still surface the original HTTP error, so
    # a decompression failure returns the raw bytes rather than mask the status.
    encoding = headers.get("content-encoding", "identity")
    try:
        return _decompress(body, encoding)
    except ValueError:
        return body


def _split_userinfo(url: str) -> tuple[str, str | None]:
    """Strip ``user:pass@`` from *url*; return cleaned URL plus Basic auth."""
    # http.client would getaddrinfo() the whole "u:p@host" as a hostname
    # (Errno -2), so pre-strip userinfo into an Authorization: Basic header.
    # The split keys off the final "@" (a literal "@" in userinfo is invalid
    # per RFC 3986), so a bracketed "u:p@[::1]:8443" cleanly yields "[::1]:8443".
    parsed = urlparse(url)
    if not (parsed.username or parsed.password):
        return url, None
    user = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    creds = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
    new_netloc = parsed.netloc[parsed.netloc.rfind("@") + 1 :]
    return parsed._replace(netloc=new_netloc).geturl(), f"Basic {creds}"


def _join_headers(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Lowercase header pairs, joining duplicates with ", " (RFC 9110 SS 5.3)."""
    out: dict[str, str] = {}
    for k, v in pairs:
        key = k.lower()
        if key in out:
            out[key] = f"{out[key]}, {v}"
        else:
            out[key] = v
    return out


def _fetch_curl(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout_sec: float,
    max_redirects: int,
    on_redirect: Callable[[str], None] | None,
    validated_hosts: ValidatedHosts | None,
) -> bytes:
    """Dispatch to the SSRF-pinned curl handle, or the plain one if unvalidated."""
    if validated_hosts is not None:
        return _fetch_curl_pinned(
            url,
            method=method,
            headers=headers,
            body=body,
            timeout_sec=timeout_sec,
            max_redirects=max_redirects,
            on_redirect=on_redirect,
            validated_hosts=validated_hosts,
        )
    return _fetch_curl_simple(
        url,
        method=method,
        headers=headers,
        body=body,
        timeout_sec=timeout_sec,
        max_redirects=max_redirects,
        on_redirect=on_redirect,
    )


def _fetch_curl_simple(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout_sec: float,
    max_redirects: int,
    on_redirect: Callable[[str], None] | None,
) -> bytes:
    """High-level curl path: ``requests.request`` with manual redirects."""
    # requests auto-decompresses .content, so no _decompress call is needed.
    # Cookies are already in headers["Cookie"], so NO cookies= kwarg is passed
    # (curl would emit a second Cookie source -- verified both are sent).
    current_url = url
    request_headers = headers
    remaining = max_redirects
    while True:
        try:
            resp = cc_requests.request(  # pyright: ignore[reportUnknownMemberType] -- curl_cffi types request() with **Unpack[RequestParams]; its TypedDict members are unstubbed, so pyright reports the callable as partially unknown
                # curl_cffi types the verb as a Literal; ``method`` originates
                # from fetch()'s ``str`` param and cannot be narrowed there.
                cast("HttpMethod", method),
                current_url,
                headers=request_headers,
                data=body,
                # "chrome" is curl_cffi's forward-tracking alias, bumped to the
                # newest Chrome profile each release -- no hardcoded version.
                impersonate="chrome",
                timeout=timeout_sec,
                allow_redirects=False,
            )
        except CurlError as e:
            raise FetchError(current_url, 0, {}, str(e).encode()) from e
        status = int(resp.status_code)
        resp_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
        content = bytes(resp.content or b"")
        # A redirect is followed only while the budget allows; at 0 the contract
        # is "do not follow, return the 3xx body" (matching the stdlib path).
        if status in _REDIRECT_STATUSES and remaining > 0:
            remaining -= 1
            redirect_url = _redirect_target(current_url, status, resp_headers)
            if on_redirect is not None:
                on_redirect(redirect_url)
            request_headers, method, body = _apply_redirect(
                request_headers, method, body, status, redirect_url
            )
            current_url = redirect_url
            continue
        if status >= 400:
            raise classify_http_error(current_url, status, resp_headers, content)
        return content


def _fetch_curl_pinned(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout_sec: float,
    max_redirects: int,
    on_redirect: Callable[[str], None] | None,
    validated_hosts: ValidatedHosts,
) -> bytes:
    """Low-level curl path: SSRF-pinned ``Curl`` handle, manual redirects."""
    # The connect IP is pinned to validated_hosts(host).ip via CurlOpt.RESOLVE
    # ("host:port:ip") so the socket hits exactly the validated address
    # regardless of DNS, re-pinned on a cross-host redirect. Bodies arrive raw
    # (no auto-decompression at this layer), so they are decompressed here.
    handle = Curl()
    try:
        current_url = url
        request_headers = headers
        remaining = max_redirects
        # Cache the last resolution: a same-origin redirect must reuse it without
        # re-invoking the resolver (the resolver contract, honored by the stdlib
        # path). Keyed on (hostname, port) so only an origin change re-resolves.
        resolved_key: tuple[str, int] | None = None
        validated: ValidatedHost | None = None
        while True:
            parsed = urlparse(current_url)
            hostname = parsed.hostname or parsed.netloc
            port = parsed.port or _default_port(parsed.scheme)
            if resolved_key != (hostname, port):
                validated = validated_hosts(hostname)
                resolved_key = (hostname, port)
            assert validated is not None
            write_buf = io.BytesIO()
            header_buf = io.BytesIO()
            handle.reset()
            handle.setopt(CurlOpt.URL, current_url.encode())
            handle.setopt(CurlOpt.CUSTOMREQUEST, method.encode())
            handle.setopt(CurlOpt.TIMEOUT_MS, int(timeout_sec * 1000))
            # Bracket a v6 pin: curl's RESOLVE is "host:port:ip" and an
            # unbracketed IPv6 collides with those colon delimiters.
            handle.setopt(
                CurlOpt.RESOLVE,
                [f"{hostname}:{port}:{_bracket_ipv6(validated.ip)}"],
            )
            handle.setopt(
                CurlOpt.HTTPHEADER,
                [f"{k}: {v}".encode() for k, v in request_headers.items()],
            )
            if body is not None:
                handle.setopt(CurlOpt.POSTFIELDS, body)
                handle.setopt(CurlOpt.POSTFIELDSIZE, len(body))
            handle.setopt(CurlOpt.WRITEDATA, write_buf)
            handle.setopt(CurlOpt.HEADERDATA, header_buf)
            handle.impersonate("chrome")
            try:
                handle.perform()
            except CurlError as e:
                raise FetchError(current_url, 0, {}, str(e).encode()) from e
            status = int(_curl_response_code(handle, CurlInfo.RESPONSE_CODE))
            resp_headers = _parse_raw_headers(header_buf.getvalue())
            raw_body = write_buf.getvalue()
            # Follow only while the budget allows; at 0 return the 3xx body.
            if status in _REDIRECT_STATUSES and remaining > 0:
                remaining -= 1
                redirect_url = _redirect_target(current_url, status, resp_headers)
                if on_redirect is not None:
                    on_redirect(redirect_url)
                request_headers, method, body = _apply_redirect(
                    request_headers, method, body, status, redirect_url
                )
                current_url = redirect_url
                continue
            if status >= 400:
                raise classify_http_error(
                    current_url,
                    status,
                    resp_headers,
                    _decompress_error_body(raw_body, resp_headers),
                )
            return _decompress(
                raw_body, resp_headers.get("content-encoding", "identity")
            )
    finally:
        handle.close()


def _curl_response_code(handle: object, info: object) -> int:
    """Read an integer ``CurlInfo`` (e.g. response code) off a ``Curl`` handle."""
    assert isinstance(handle, Curl)
    assert isinstance(info, CurlInfo)
    value = handle.getinfo(info)
    assert isinstance(value, int)
    return value


def _redirect_target(current_url: str, status: int, headers: dict[str, str]) -> str:
    """Resolve a redirect ``Location`` against *current_url*; raise if absent."""
    location = headers.get("location")
    if not location:
        raise FetchError(
            current_url, status, headers, b"Redirect with no Location header"
        )
    # RFC 3986 relative resolution: handles absolute, scheme-relative, and
    # path-relative Locations without corrupting the host.
    return urljoin(current_url, location)


def _parse_raw_headers(block: bytes) -> dict[str, str]:
    """Parse a raw CRLF response-header block into a merged lowercase dict."""
    pairs: list[tuple[str, str]] = []
    for line in block.split(b"\r\n"):
        if not line or b":" not in line:
            continue
        k, _, v = line.partition(b":")
        pairs.append((k.decode("latin-1").strip(), v.decode("latin-1").strip()))
    return _join_headers(pairs)


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


def _default_port(scheme: str) -> int:
    """Return the default TCP port for an HTTP scheme."""
    return 443 if scheme == "https" else 80


def _netloc(hostname: str, port: int | None) -> str:
    """Recombine a hostname and optional port into a netloc."""
    return f"{hostname}:{port}" if port is not None else hostname


def _host_header(host: str, port: int | None, scheme: str) -> str:
    """The ``Host`` header value: bare host, plus a non-default port."""
    # RFC 9110 requires the port in Host only when it is not the scheme default;
    # a real browser omits :443/:80. The ONE place this rule lives, so the
    # initial hop and the cross-host-redirect rebuild cannot disagree.
    if port is not None and port != _default_port(scheme):
        return _netloc(host, port)
    return host


def _bracket_ipv6(host: str) -> str:
    """Wrap an IPv6 literal in brackets; pass hostnames and IPv4 through."""
    # http.client splits host on the last ':' for a port, misparsing a bare
    # "2606:4700::6810:7c60" as host+port. Bracketing avoids that heuristic.
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _rewrite_origin(headers: dict[str, str], target_url: str) -> dict[str, str]:
    """Return ``headers`` with ``Origin`` reset to ``target_url``'s origin."""
    # A cross-origin redirect must NOT leak the source Origin to the new host (a
    # real browser sets it to the new origin, never the old). A GET carries no
    # Origin and passes through. Field names are case-insensitive, so a
    # caller-supplied "origin" in any case is matched and rewritten in place.
    origin_key = next((k for k in headers if k.lower() == "origin"), None)
    if origin_key is None:
        return headers
    parsed = urlparse(target_url)
    scheme = parsed.scheme or "https"
    # Bracket a v6 host: an unbracketed IPv6 literal is not a valid Origin
    # (the colons collide with the scheme/port delimiters).
    netloc = _netloc(_bracket_ipv6(parsed.hostname or ""), parsed.port)
    return {**headers, origin_key: f"{scheme}://{netloc}"}


def _apply_redirect(
    headers: dict[str, str],
    method: str,
    body: bytes | None,
    status: int,
    redirect_url: str,
) -> tuple[dict[str, str], str, bytes | None]:
    """Compute the (headers, method, body) for the next hop of a redirect."""
    # The ONE place the per-hop transform lives, called by every transport's
    # redirect loop so the rules cannot drift: rewrite Origin to the new target,
    # and on a 303 convert to a bodyless GET dropping Content-Type (a real
    # browser sends none on a bodyless GET). Casing-insensitive throughout.
    headers = _rewrite_origin(headers, redirect_url)
    if status == 303 and method != "GET":
        method = "GET"
        body = None
        headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
    return headers, method, body


def _open_connection(
    scheme: str,
    hostname: str,
    timeout_sec: float,
    *,
    port: int | None = None,
    resolved_ip: str = "",
) -> HTTPConn:
    """Open a new HTTP/HTTPS connection; pin to ``resolved_ip`` when given."""
    connect_host = _bracket_ipv6(resolved_ip or hostname)
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
) -> bytes:
    """Connection path: http.client with manual redirect following."""
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
        # resolver returns the bare host (its contract), so _host_header
        # re-appends any non-default port.
        request_headers = {
            "Host": _host_header(validated.host, port, scheme),
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
                request_headers, method, body = _apply_redirect(
                    request_headers, method, body, response.status, redirect_url
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
                            "Host": _host_header(validated.host, port, scheme),
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
                    _decompress_error_body(raw_body, resp_headers),
                )
            encoding = resp_headers.get("content-encoding", "identity")
            return _decompress(raw_body, encoding)
    finally:
        raw_conn.close()
