"""curl_cffi transport for :mod:`sagent.lib.web.fetch`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from urllib.parse import urlparse

import io
import threading

from sagent.lib.web.challenge import classify_http_error
from sagent.lib.web.errors import FetchError
from sagent.lib.web.fetch.common import (
    _REDIRECT_STATUSES,
    Observer,
    ValidatedHost,
    ValidatedHosts,
    apply_redirect,
    bracket_ipv6,
    decompress,
    decompress_error_body,
    default_port,
    join_headers,
    redirect_target,
)


if TYPE_CHECKING:
    from curl_cffi import requests as cc_requests
    from curl_cffi.requests import Response
    from curl_cffi.requests.impersonate import BrowserTypeLiteral
    from curl_cffi.requests.session import HttpMethod

    import curl_cffi
else:
    from wrapt import lazy_import

    curl_cffi = lazy_import("curl_cffi")


__all__ = [
    "close_curl_session",
    "close_curl_sessions_except",
    "curl_session",
    "fetch_curl",
    "seed_session_jar",
    "set_session_cookies",
]

# Live curl_cffi Sessions keyed by identity, so a session reuses one connection
# across requests -- the connection continuity a real browser has, and which a
# per-call request() (fresh TLS each time) lacks. Keyed on impersonate too.
# config-globals: ignore -- live pool of open connections, not a tunable.
_curl_sessions: dict[tuple[str, str, str], cc_requests.Session[Response]] = {}
# Guards every mutation of the live curl session pool.
_curl_lock = threading.Lock()  # config-globals: ignore -- guards live sessions.


def curl_session(
    egress: str, domain: str, impersonate: str
) -> cc_requests.Session[Response]:
    """Return the pooled curl_cffi Session for an identity, creating it once.

    Keyed on the REGISTRABLE domain (eTLD+1), not the exact host, so sibling
    subdomains of one site share a single connection + cookie jar -- the HTTP/2
    connection coalescing a real browser does for hosts on one certificate.
    ``www.google.com`` and ``scholar.google.com`` therefore reuse one session,
    so a warm-up GET to the apex carries its TLS handshake and Set-Cookie into a
    later request to the subdomain (a cold second connection is a bot tell that
    Scholar, in particular, budgets against).
    """
    key = (egress, _registrable_domain(domain), impersonate)
    with _curl_lock:
        session = _curl_sessions.get(key)
        if session is None:
            session = cast(
                "cc_requests.Session[Response]",
                curl_cffi.requests.Session(
                    impersonate=cast("BrowserTypeLiteral", impersonate)
                ),
            )
            _curl_sessions[key] = session
        return session


def seed_session_jar(
    session: cc_requests.Session[Response], domain: str, cookies: dict[str, str]
) -> None:
    """Load stored profile cookies into a curl session jar it does not yet hold.

    Cross-process persistence: the profile store outlives the in-memory session,
    so a fresh process seeds the jar from disk. Only names absent from the jar
    are added, so a live rotating cookie (curl tracking Scholar's NID/GSP) is
    never clobbered by a stale stored copy.
    """
    if not cookies:
        return
    present = {c.name for c in session.cookies.jar}
    for name, value in cookies.items():
        if name not in present:  # never clobber a live jar cookie with a stale copy
            _jar_set(session, domain, name, value)


def set_session_cookies(
    session: cc_requests.Session[Response], domain: str, cookies: dict[str, str]
) -> None:
    """Set caller cookies into a curl session jar, OVERWRITING any prior value.

    Unlike :func:`seed_session_jar` (which preserves live jar cookies), a caller
    cookie is an explicit per-call override and must win, so it replaces a
    same-named jar entry. This keeps the jar the single cookie source on the curl
    path: sending the cookie via a header too would duplicate a name the jar
    already holds.
    """
    for name, value in cookies.items():
        _jar_set(session, domain, name, value)


def _jar_set(
    session: cc_requests.Session[Response], domain: str, name: str, value: str
) -> None:
    """Set one cookie in a curl jar, honoring RFC 6265bis name-prefix rules."""
    # RFC 6265bis 4.1.3 cookie-name prefixes, which curl_cffi enforces (and warns
    # + coerces when violated): a __Secure- cookie must be Secure; a __Host-
    # cookie must additionally be host-only (no Domain) with Path=/. Chrome only
    # ever sends these over https, so set them to match.
    if name.startswith("__Host-"):
        session.cookies.set(name, value, path="/", secure=True)
    elif name.startswith("__Secure-"):
        session.cookies.set(name, value, domain=domain, secure=True)
    else:
        session.cookies.set(name, value, domain=domain)


def _registrable_domain(host: str) -> str:
    """Return the eTLD+1 of a host (``a.b.example.co.uk`` -> ``example.co.uk``).

    A coarse public-suffix approximation: a two-label tail is the registrable
    domain, unless the last label is a 2-letter ccTLD and the second-to-last is
    a short (<=3-char) second-level label (``co.uk``, ``com.au``), in which case
    the tail is three labels. Sufficient for connection coalescing -- an
    over-broad grouping only shares a connection, never crosses a real origin
    boundary for cookies (those stay domain-scoped by the jar).
    """
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    tail = labels[-2:]
    if len(labels[-1]) == 2 and len(labels[-2]) <= 3:
        return ".".join(labels[-3:])
    return ".".join(tail)


def close_curl_session(egress: str, domain: str, impersonate: str) -> None:
    """Close and drop an identity's pooled Session (a burn ends the connection)."""
    key = (egress, _registrable_domain(domain), impersonate)
    with _curl_lock:
        session = _curl_sessions.pop(key, None)
    if session is not None:
        session.close()  # I/O outside the lock; the pop already removed it.


def close_curl_sessions_except(egress: str | None) -> None:
    """Close pooled sessions that belong to a different egress."""
    with _curl_lock:
        sessions = [
            _curl_sessions.pop(key) for key in list(_curl_sessions) if key[0] != egress
        ]
    for session in sessions:
        session.close()


def _curl_set_cookies(resp: Response) -> list[str]:
    """Return the individual ``Set-Cookie`` headers of a curl response."""
    get_list = getattr(resp.headers, "get_list", None)
    if get_list is None:
        value = resp.headers.get("set-cookie")
        return [value] if value else []
    return list(cast("list[str]", get_list("set-cookie")))


@dataclass(slots=True, kw_only=True)
class _CurlLoop:
    """Mutable per-hop state shared by both curl backends' redirect loops.

    Holds the current URL, method, headers, body, and remaining redirect budget.
    :meth:`follow` runs the identical post-response decision both backends make:
    fire ``on_response``, and if the status is a followable redirect within
    budget, advance the state to the next hop (via :func:`apply_redirect`) and
    report ``True``. A ``False`` return means the response is terminal, leaving
    each backend to classify/return its (differently decompressed) body.
    """

    url: str
    method: str
    headers: dict[str, str]
    body: bytes | None
    remaining: int

    def follow(
        self,
        status: int,
        resp_headers: dict[str, str],
        *,
        on_response: Observer | None,
        on_redirect: Callable[[str], None] | None,
    ) -> bool:
        """Fire ``on_response``; advance to the next hop on a redirect within budget."""
        if on_response is not None:
            on_response(status, resp_headers, self.url)
        # A redirect is followed only while the budget allows; at 0 the contract
        # is "do not follow, return the 3xx body" (matching the stdlib path).
        if status not in _REDIRECT_STATUSES or self.remaining <= 0:
            return False
        self.remaining -= 1
        redirect_url = redirect_target(self.url, status, resp_headers)
        if on_redirect is not None:
            on_redirect(redirect_url)
        self.headers, self.method, self.body = apply_redirect(
            self.url, self.headers, self.method, self.body, status, redirect_url
        )
        self.url = redirect_url
        return True


def fetch_curl(
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
    session: cc_requests.Session[Response] | None = None,
) -> bytes:
    """Dispatch to the SSRF-pinned curl handle, or the plain one if unvalidated."""
    if validated_hosts is not None:
        # The pinned path owns a raw Curl handle for SSRF; no Session reuse.
        return _fetch_curl_pinned(
            url,
            method=method,
            headers=headers,
            body=body,
            timeout_sec=timeout_sec,
            max_redirects=max_redirects,
            impersonate=impersonate,
            on_redirect=on_redirect,
            on_response=on_response,
            validated_hosts=validated_hosts,
        )
    return _fetch_curl_simple(
        url,
        method=method,
        headers=headers,
        body=body,
        timeout_sec=timeout_sec,
        max_redirects=max_redirects,
        impersonate=impersonate,
        on_redirect=on_redirect,
        on_response=on_response,
        session=session,
    )


def _fetch_curl_simple(
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
    session: cc_requests.Session[Response] | None = None,
) -> bytes:
    """High-level curl path: ``requests.request`` with manual redirects."""
    # requests auto-decompresses .content, so no decompress call is needed.
    # Cookies are already in headers["Cookie"], so NO cookies= kwarg is passed
    # (curl would emit a second Cookie source -- verified both are sent).
    loop = _CurlLoop(
        url=url, method=method, headers=headers, body=body, remaining=max_redirects
    )
    impers = cast("BrowserTypeLiteral", impersonate)
    while True:
        try:
            verb = cast("HttpMethod", loop.method)  # curl types verb as a Literal.
            resp = (
                session.request(  # pyright: ignore[reportUnknownMemberType] -- curl_cffi's **Unpack[RequestParams] TypedDict is unstubbed
                    verb,
                    loop.url,
                    headers=loop.headers,
                    data=loop.body,
                    impersonate=impers,
                    timeout=timeout_sec,
                    allow_redirects=False,
                )
                if session is not None
                else curl_cffi.requests.request(  # pyright: ignore[reportUnknownMemberType] -- curl_cffi's **Unpack[RequestParams] TypedDict is unstubbed
                    verb,
                    loop.url,
                    headers=loop.headers,
                    data=loop.body,
                    impersonate=impers,
                    timeout=timeout_sec,
                    allow_redirects=False,
                )
            )
        except curl_cffi.CurlError as e:
            raise FetchError(loop.url, 0, {}, str(e).encode()) from e
        # curl_cffi's request/Session.request type a None return for the
        # thread/stream overloads; the sync call here always yields a Response.
        assert resp is not None
        status = int(resp.status_code)
        resp_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
        # curl_cffi's Headers.items() lossily folds duplicate Set-Cookie with
        # ", " (and Set-Cookie values may contain ", "); get_list preserves the
        # individual headers, newline-joined to match join_headers so
        # parse_set_cookie splits them back exactly.
        cookies_list = _curl_set_cookies(resp)
        if cookies_list:
            resp_headers["set-cookie"] = "\n".join(cookies_list)
        content = bytes(resp.content or b"")
        current_url = loop.url
        if loop.follow(
            status, resp_headers, on_response=on_response, on_redirect=on_redirect
        ):
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
    impersonate: str,
    on_redirect: Callable[[str], None] | None,
    on_response: Observer | None,
    validated_hosts: ValidatedHosts,
) -> bytes:
    """Low-level curl path: SSRF-pinned ``Curl`` handle, manual redirects."""
    # The connect IP is pinned to validated_hosts(host).ip via CurlOpt.RESOLVE
    # ("host:port:ip") so the socket hits exactly the validated address
    # regardless of DNS, re-pinned on a cross-host redirect. Bodies arrive raw
    # (no auto-decompression at this layer), so they are decompressed here.
    # Bind the lazy curl_cffi symbols once at entry (materializes the module).
    Curl, CurlError, CurlInfo, CurlOpt = (
        curl_cffi.Curl,
        curl_cffi.CurlError,
        curl_cffi.CurlInfo,
        curl_cffi.CurlOpt,
    )
    handle = Curl()
    try:
        loop = _CurlLoop(
            url=url, method=method, headers=headers, body=body, remaining=max_redirects
        )
        # Cache the last resolution: a same-origin redirect must reuse it without
        # re-invoking the resolver (the resolver contract, honored by the stdlib
        # path). Keyed on (hostname, port) so only an origin change re-resolves.
        resolved_key: tuple[str, int] | None = None
        validated: ValidatedHost | None = None
        while True:
            parsed = urlparse(loop.url)
            hostname = parsed.hostname or parsed.netloc
            port = parsed.port or default_port(parsed.scheme)
            if resolved_key != (hostname, port):
                validated = validated_hosts(hostname)
                resolved_key = (hostname, port)
            assert validated is not None
            write_buf = io.BytesIO()
            header_buf = io.BytesIO()
            handle.reset()
            handle.setopt(CurlOpt.URL, loop.url.encode())
            handle.setopt(CurlOpt.CUSTOMREQUEST, loop.method.encode())
            handle.setopt(CurlOpt.TIMEOUT_MS, int(timeout_sec * 1000))
            # Bracket a v6 pin: curl's RESOLVE is "host:port:ip" and an
            # unbracketed IPv6 collides with those colon delimiters.
            handle.setopt(
                CurlOpt.RESOLVE,
                [f"{hostname}:{port}:{bracket_ipv6(validated.ip)}"],
            )
            handle.setopt(
                CurlOpt.HTTPHEADER,
                [f"{k}: {v}".encode() for k, v in loop.headers.items()],
            )
            if loop.body is not None:
                handle.setopt(CurlOpt.POSTFIELDS, loop.body)
                handle.setopt(CurlOpt.POSTFIELDSIZE, len(loop.body))
            handle.setopt(CurlOpt.WRITEDATA, write_buf)
            handle.setopt(CurlOpt.HEADERDATA, header_buf)
            handle.impersonate(impersonate)
            try:
                handle.perform()
            except CurlError as e:
                raise FetchError(loop.url, 0, {}, str(e).encode()) from e
            status = int(_curl_response_code(handle, CurlInfo.RESPONSE_CODE))
            resp_headers = _parse_raw_headers(header_buf.getvalue())
            raw_body = write_buf.getvalue()
            current_url = loop.url
            if loop.follow(
                status, resp_headers, on_response=on_response, on_redirect=on_redirect
            ):
                continue
            if status >= 400:
                raise classify_http_error(
                    current_url,
                    status,
                    resp_headers,
                    decompress_error_body(raw_body, resp_headers),
                )
            return decompress(
                raw_body, resp_headers.get("content-encoding", "identity")
            )
    finally:
        handle.close()


def _curl_response_code(handle: object, info: object) -> int:
    """Read an integer ``CurlInfo`` (e.g. response code) off a ``Curl`` handle."""
    assert isinstance(handle, curl_cffi.Curl)
    assert isinstance(info, curl_cffi.CurlInfo)
    value = handle.getinfo(info)
    assert isinstance(value, int)
    return value


def _parse_raw_headers(block: bytes) -> dict[str, str]:
    """Parse a raw CRLF response-header block into a merged lowercase dict."""
    pairs: list[tuple[str, str]] = []
    for line in block.split(b"\r\n"):
        if not line or b":" not in line:
            continue
        k, _, v = line.partition(b":")
        pairs.append((k.decode("latin-1").strip(), v.decode("latin-1").strip()))
    return join_headers(pairs)
