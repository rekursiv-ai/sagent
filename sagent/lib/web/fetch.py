"""Unified HTTP fetch for ``sagent.lib.web``.

All HTTP in ``sagent.lib.web`` flows through :func:`fetch`. Sync, with
transparent decompression and optional retry. The transport is curl_cffi
(a Chrome-compatible TLS/HTTP-2 profile). A stdlib ``http.client`` path is
retained as a reference implementation for validating the curl path, not as a
runtime fallback -- curl_cffi is a hard dependency. Every call returns
``(body, session)``; the returned :class:`FetchSession` is a browsing identity
you thread into the next call.

Usage::

    from sagent.lib.web.fetch import fetch

    # Simple (99% case): ignore the returned session.
    body, _ = fetch(url)
    html = fetch(url)[0].decode("utf-8")

    # Repeated calls: thread the session so each request builds on the last
    # (cookies set, Accept-CH opt-ins), the way a real browser's do.
    body, session = fetch(url)
    body, session = fetch(next_url, session=session)
    body, session = fetch(next_url, session=session)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast, override
from urllib.parse import unquote, urlencode, urljoin, urlparse

import base64
import gzip
import http.client
import io
import ipaddress
import json as json_lib
import logging
import random
import ssl
import threading
import time
import zlib

import brotli
import zstandard

from sagent.lib.custom_json import JSONValue
from sagent.lib.web.challenge import classify_http_error
from sagent.lib.web.chrome_headers import (
    chrome_client_hints,
    chrome_navigation_headers,
    impersonate_version_platform,
)
from sagent.lib.web.errors import BotDetectionError, FetchError
from sagent.lib.web.fetch_zendriver import (
    default_profile_dir,
    fetch_zendriver,
    shutdown_browsers,
)
from sagent.lib.web.profile import (
    Profile,
    ProfileStore,
    parse_set_cookie,
    parsedate_to_datetime_or_none,
)
from sagent.lib.web.useragents import draw_user_agent, kind_for_impersonate


if TYPE_CHECKING:
    from curl_cffi import requests as cc_requests
    from curl_cffi.requests import Response
    from curl_cffi.requests.impersonate import BrowserTypeLiteral
    from curl_cffi.requests.session import HttpMethod

    import curl_cffi
else:
    from wrapt import lazy_import

    # ~150ms import (importlib.metadata + asyncio); paid on first fetch, not at
    # import. Runtime code reaches every symbol through this one module proxy
    # (curl_cffi.Curl / .CurlError / .requests.request); a per-symbol lazy proxy
    # cannot be used in ``except``/``isinstance`` (not seen as a real class).
    curl_cffi = lazy_import("curl_cffi")


__all__ = [
    "FetchSession",
    "RequestParams",
    "Transport",
    "egress_ip",
    "fetch",
    "last_known_egress_ip",
    "on_egress_rotation",
    "set_last_egress_ip",
]

logger = logging.getLogger(__name__)

Transport = Literal["auto", "curl", "curl-then-zendriver", "zendriver", "stdlib"]


def resolve_transport(
    url: str,
    transport: Transport,
    *,
    method: HttpMethod = "GET",
    raw_headers: bool = False,
    has_body: bool = False,
    zendriver_domains: tuple[str, ...] = ("google.com",),
) -> Transport:
    """Resolve ``auto`` to a concrete transport for this request."""
    if transport != "auto":
        return transport
    if method != "GET" or raw_headers or has_body:
        return "curl"
    host = (urlparse(url).hostname or "").lower()
    if any(
        host == domain or host.endswith(f".{domain}") for domain in zendriver_domains
    ):
        return "zendriver"
    return "curl-then-zendriver"


HTTPConn = http.client.HTTPConnection | http.client.HTTPSConnection


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatedHost:
    host: str
    ip: str


ValidatedHosts = Callable[[str], ValidatedHost]

# Internal per-hop response sink: (status, response headers, responding URL).
# The URL lets a sink scope what it learns (cookies, hints) to the hop's origin,
# so a cross-origin redirect never mis-attributes the target's Set-Cookie to the
# source. The public ``RequestParams.on_response`` stays (status, headers).
_Observer = Callable[[int, dict[str, str], str], None]

_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


@dataclass(frozen=True, slots=True, kw_only=True)
class FetchSession:
    """Frozen browsing state threaded through :func:`fetch` calls.

    Returned by every :func:`fetch` call carrying what the response set; pass it
    back as the ``session`` argument of the next call. Serializable via
    :meth:`serialize` / :meth:`deserialize`.

    Attributes:
      impersonate: The curl_cffi TLS-impersonation target; the User-Agent and
        client hints are derived from it.
      egress_ip: The public egress the session is keyed to; empty until observed.
      cookies: The cookie jar (``name -> value``), updated from ``Set-Cookie``.
      accept_ch: Per-origin (``scheme://host``) sets of extended client-hint
        header names the origin requested via ``Accept-CH``.

    """

    impersonate: str = "chrome"
    egress_ip: str = ""
    cookies: Mapping[str, str] = field(default_factory=dict[str, str])
    accept_ch: Mapping[str, frozenset[str]] = field(
        default_factory=dict[str, frozenset[str]]
    )

    def with_cookies(self, updates: Mapping[str, str]) -> FetchSession:
        """Return a copy whose jar is merged with ``updates`` (new values win)."""
        if not updates:
            return self
        return replace(self, cookies={**self.cookies, **updates})

    def with_accept_ch(self, origin: str, hints: frozenset[str]) -> FetchSession:
        """Return a copy recording ``origin``'s ``hints`` (unchanged if same)."""
        if not hints or self.accept_ch.get(origin) == hints:
            return self
        return replace(self, accept_ch={**self.accept_ch, origin: hints})

    def with_egress(self, ip: str) -> FetchSession:
        """Return a copy pinned to egress ``ip`` (unchanged if already pinned)."""
        return self if ip == self.egress_ip else replace(self, egress_ip=ip)

    def serialize(self) -> dict[str, object]:
        """Return a JSON-serializable dict of this session's state.

        Returns:
          data: A dict with ``impersonate``, ``egress_ip``, ``cookies``, and
            ``accept_ch``. Each hint set is emitted as a sorted list purely for a
            deterministic serialized form; the hints are an unordered set (the
            outgoing request emits them in Chrome's own client-hint order, not
            this one), and :meth:`deserialize` rebuilds a ``frozenset``.

        """
        return {
            "impersonate": self.impersonate,
            "egress_ip": self.egress_ip,
            "cookies": dict(self.cookies),
            "accept_ch": {
                origin: sorted(hints) for origin, hints in self.accept_ch.items()
            },
        }

    @classmethod
    def deserialize(cls, data: Mapping[str, object]) -> FetchSession:
        """Reconstruct a session from :meth:`serialize` output.

        Args:
          data: A dict as produced by :meth:`serialize`.

        Returns:
          session: The reconstructed :class:`FetchSession`.

        """
        accept_ch = cast("Mapping[str, list[str]]", data.get("accept_ch", {}))
        return cls(
            impersonate=cast("str", data.get("impersonate", "chrome")),
            egress_ip=cast("str", data.get("egress_ip", "")),
            cookies=dict(cast("Mapping[str, str]", data.get("cookies", {}))),
            accept_ch={origin: frozenset(hints) for origin, hints in accept_ch.items()},
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestParams:
    """Per-call request parameters for :func:`fetch`.

    Attributes:
      method: HTTP method.
      params: Query parameters appended to the URL.
      data: Form data, sent as application/x-www-form-urlencoded. Mutually
        exclusive with ``json``.
      json: JSON-serializable body, sent as application/json. Mutually exclusive
        with ``data``.
      headers: Extra headers, merged over the session identity (these win).
      cookies: Cookies to send, merged over the session jar (these win).
      raw_headers: Send exactly ``headers`` plus cookies and auth; skip the
        Chrome identity and the session jar.
      retries: Retry attempts for transient failures.
      timeout_sec: Socket timeout in seconds.
      max_redirects: Maximum redirects to follow; 0 disables.
      on_redirect: Called with the redirect target URL before following; raise to
        abort.
      on_response: Called with ``(status, headers)`` for every received response.
        Observational; must not raise.
      validated_hosts: Resolver returning a validated IP per hostname; receives
        the bare hostname and must resolve it to the same IP for the call.
      transport: Retrieval transport. ``"auto"`` (default) selects Zendriver
        for ``google.com`` and its subdomains, curl-then-Zendriver for other GETs,
        and curl for requests a browser cannot replay. ``"curl"`` is the
        curl_cffi impersonated path; ``"stdlib"`` is the http.client reference path;
        ``"zendriver"`` drives a real headless Chrome via
        :mod:`sagent.lib.web.fetch_zendriver` (opt-in, for JS/challenge-walled
        pages); ``"curl-then-zendriver"`` tries curl first and falls back to zendriver ONLY when
        curl is bot-blocked (a :class:`BotDetectionError`) -- a non-block failure
        propagates unchanged. ``"zendriver"`` and ``"curl-then-zendriver"`` are
        GET-only and reject raw-header mode. A filtering proxy preserves
        ``validated_hosts`` DNS pinning for the browser leg.

    """

    method: HttpMethod = "GET"
    params: dict[str, str | int] | None = None
    data: dict[str, str] | None = None
    json: JSONValue = None
    headers: dict[str, str] | None = None
    cookies: dict[str, str] | None = None
    raw_headers: bool = False
    retries: int = 0
    timeout_sec: float = 30
    max_redirects: int = 10
    on_redirect: Callable[[str], None] | None = None
    on_response: Callable[[int, dict[str, str]], None] | None = None
    validated_hosts: ValidatedHosts | None = None
    transport: Transport = "auto"

    def __post_init__(self) -> None:
        """Reject contradictory or out-of-range parameters at construction."""
        if self.data is not None and self.json is not None:
            raise ValueError("'data' and 'json' are mutually exclusive.")
        if self.retries < 0:
            raise ValueError(f"'retries' must be >= 0, got {self.retries}.")
        if self.max_redirects < 0:
            raise ValueError(f"'max_redirects' must be >= 0, got {self.max_redirects}.")
        if self.timeout_sec <= 0:
            raise ValueError(f"'timeout_sec' must be > 0, got {self.timeout_sec}.")
        if self.transport in ("zendriver", "curl-then-zendriver"):
            # The browser leg can replay GET navigation, headers, and cookies,
            # but not a request body or byte-exact raw-header mode.
            if self.method != "GET":
                raise ValueError(
                    f"The {self.transport} backend supports only GET requests."
                )
            if self.data is not None or self.json is not None:
                raise ValueError(
                    f"The {self.transport} backend cannot send a request body."
                )
            if self.raw_headers:
                raise ValueError(
                    f"The {self.transport} transport cannot honor 'raw_headers'."
                )

    def backoff_delay(self, attempt: int, headers: dict[str, str]) -> float:
        """Retry backoff in seconds for ``attempt``, honoring any ``Retry-After``.

        ``Retry-After`` is delta-seconds OR an HTTP-date (RFC 9110 SS 10.2.3);
        both forms are honored, capped at 30s. A malformed value falls through to
        the computed exponential backoff.
        """
        # Only the HTTP-status retry path supplies headers; network-error retries
        # pass an empty mapping and always fall through to the computed backoff.
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                return min(float(retry_after), 30)
            except ValueError:
                pass
            when = parsedate_to_datetime_or_none(retry_after.strip())
            if when is not None:
                return min(max((when - datetime.now(UTC)).total_seconds(), 0.0), 30)
        delay = min(1.0 * (2**attempt), 30)
        return delay + random.uniform(0, delay * 0.5)  # noqa: S311 -- jitter


def fetch(
    url: str,
    *,
    session: FetchSession | None = None,
    request: RequestParams | None = None,
) -> tuple[bytes, FetchSession]:
    """Fetch a URL; return the body and the updated browsing session.

    Args:
      url: Fully-qualified URL (http or https).
      session: A prior returned session to reuse, or ``None`` for a fresh one.
      request: Per-call parameters, or ``None`` for the defaults.

    Returns:
      body: The response bytes.
      session: A :class:`FetchSession` carrying what the response set; pass it to
        the next call.

    Raises:
      FetchError: On non-success HTTP status after exhausting retries.
      ValueError: On unsupported or corrupt Content-Encoding.

    """
    return _Request(
        url=url,
        session=FetchSession() if session is None else session,
        params=RequestParams() if request is None else request,
    ).fetch()


class _ResponseLearner:
    """Accumulates what responses teach a session, then folds it in.

    Its :meth:`observe` is the ``on_response`` sink for a :func:`fetch_session`
    call: it records ``Set-Cookie`` and ``Accept-CH`` from every response (each
    redirect hop and the final one), passing the status/headers through to the
    caller's own callback. :meth:`merge_into` returns the session updated with
    everything observed.
    """

    def __init__(
        self, *, url: str, caller: Callable[[int, dict[str, str]], None] | None
    ) -> None:
        self._url = url
        self._caller = caller
        self._cookies: dict[str, str] = {}
        self._accept_ch: dict[str, frozenset[str]] = {}

    def observe(self, status: int, resp_headers: dict[str, str], url: str) -> None:
        """Record cookies + Accept-CH from a hop; forward to the caller.

        Cookies are absorbed into the session jar only when the responding
        ``url`` shares the request's origin -- a redirect target on a foreign
        origin sets ITS cookies, which must not be attributed to this session's
        identity. Accept-CH opt-ins are keyed by the responding origin, so a
        cross-origin hop's hints attach to that origin, never this one.
        """
        if _origin(url) == _origin(self._url):
            set_cookie = resp_headers.get("set-cookie")
            if set_cookie:
                self._cookies.update(parse_set_cookie(set_cookie))
        hints = _accept_ch_hints(resp_headers)
        if hints:
            self._accept_ch[_origin(url)] = hints
        if self._caller is not None:
            self._caller(status, resp_headers)

    def merge_into(self, session: FetchSession) -> FetchSession:
        """Return ``session`` updated with the observed cookies + opt-ins."""
        updated = session.with_cookies(self._cookies)
        for origin, hints in self._accept_ch.items():
            updated = updated.with_accept_ch(origin, hints)
        return updated


@dataclass(frozen=True, slots=True, kw_only=True)
class _Request:
    """One :func:`fetch` invocation: its URL, session, and per-call params.

    ``observer`` is the internal per-hop sink (:class:`_ResponseLearner`, which
    also forwards to the caller's public 2-arg ``params.on_response``). It is
    kept OFF ``params`` so the public 2-arg contract and this 3-arg internal
    contract don't conflate.
    """

    url: str
    session: FetchSession
    params: RequestParams
    observer: _Observer | None = None

    @property
    def domain(self) -> str:
        """The hostname the session identity is keyed on (``""`` if hostless)."""
        return urlparse(self.url).hostname or ""

    def fetch(self) -> tuple[bytes, FetchSession]:
        """Perform the request; return the body and the updated session."""
        p = self.params
        resolved = resolve_transport(
            self.url,
            p.transport,
            method=p.method,
            raw_headers=p.raw_headers,
            has_body=p.data is not None or p.json is not None,
        )
        if resolved != p.transport:
            p = replace(p, transport=resolved)
        learner = _ResponseLearner(url=self.url, caller=p.on_response)
        request = replace(self, params=p, observer=learner.observe)
        seeded_cookies = {**self.session.cookies, **(p.cookies or {})}
        if p.raw_headers:
            body = request.send(
                headers=p.headers, cookies=seeded_cookies or None, raw_headers=True
            )
        else:
            body = _fetch_with_identity(
                request,
                caller_headers=p.headers,
                caller_cookies=seeded_cookies or None,
            )
        return body, learner.merge_into(self.session)

    def send(
        self,
        *,
        headers: dict[str, str] | None,
        cookies: dict[str, str] | None,
        raw_headers: bool,
        on_response: _Observer | None = None,
        curl: cc_requests.Session[Response] | None = None,
    ) -> bytes:
        """Perform the request once (with retries) via the raw transport."""
        return _fetch_once(
            self.url,
            self.params,
            headers=headers,
            cookies=cookies,
            raw_headers=raw_headers,
            impersonate=self.session.impersonate,
            accept_ch=self.session.accept_ch,
            on_response=self.observer if on_response is None else on_response,
            session=curl,
        )


def _fetch_with_identity(
    request: _Request,
    *,
    caller_headers: dict[str, str] | None,
    caller_cookies: dict[str, str] | None,
) -> bytes:
    """Send ``request`` under the stored per-``(egress, domain)`` identity."""
    if request.params.transport == "zendriver":
        return _send_via_zendriver(
            request,
            headers=caller_headers,
            cookies=caller_cookies,
        )
    if request.params.transport == "curl-then-zendriver":
        # Curl first (fast, cheap); fall back to the real browser ONLY when curl
        # is bot-blocked -- the one case the browser can clear that curl cannot.
        # A non-block failure (404, timeout) propagates: the browser would not
        # help and must not silently pay Chrome's launch cost.
        try:
            return _fetch_with_identity(
                replace(request, params=replace(request.params, transport="curl")),
                caller_headers=caller_headers,
                caller_cookies=caller_cookies,
            )
        except BotDetectionError:
            return _send_via_zendriver(
                request,
                headers=caller_headers,
                cookies=caller_cookies,
            )
    domain = request.domain
    if not domain:
        return _send_as(request, None, None, caller_headers, caller_cookies)

    store = ProfileStore.shared()
    # A cheap last-known egress finds an existing session; a NEW session must pin
    # to the live egress so a stale last-known never mis-keys the identity.
    egress = egress_ip(cache=True)
    profile = store.load(egress, domain) if egress is not None else None
    if profile is None:
        egress = egress_ip(cache=False)

    try:
        return _send_as(request, profile, egress, caller_headers, caller_cookies)
    except BotDetectionError:
        if profile is None:
            raise  # First contact burned: no known identity to discard or retry.
        assert egress is not None  # A profile only loads once egress resolved.
        store.discard(egress, domain)
        close_curl_session(egress, domain, request.session.impersonate)
        # The burn may be a VPN rotation: re-resolve live before the fresh retry.
        return _send_as(
            request, None, egress_ip(cache=False), caller_headers, caller_cookies
        )


def _send_as(
    request: _Request,
    profile: Profile | None,
    egress: str | None,
    caller_headers: dict[str, str] | None,
    caller_cookies: dict[str, str] | None,
) -> bytes:
    """Send seeded with ``profile``'s UA + cookies (caller wins), save Set-Cookie."""
    # egress=None => keyless: draw a UA, persist nothing. A drawn UA matches the
    # request's impersonated browser so the UA and TLS fingerprint agree.
    impersonate = request.session.impersonate
    ua = (
        profile.ua
        if profile is not None
        else draw_user_agent(kind_for_impersonate(impersonate))
    )
    jar = dict(profile.cookies) if profile is not None else {}
    # On the curl path, curl_cffi's impersonate emits a COHERENT User-Agent that
    # matches its TLS/HTTP-2 fingerprint; seeding a stored/drawn UA here would
    # override it and make the two disagree (a bot tell). Let impersonate own the
    # UA on curl; only the stdlib reference path (no impersonation) needs one.
    seeded_headers: dict[str, str] = {**(caller_headers or {})}
    if request.params.transport == "stdlib":
        seeded_headers = {"User-Agent": ua, **seeded_headers}
    captured: dict[str, str] = {}

    request_origin = _origin(request.url)

    def capture(status: int, resp_headers: dict[str, str], url: str) -> None:
        # Persist only cookies the request's OWN origin set; a cross-origin
        # redirect target's Set-Cookie belongs to that origin's profile, not this
        # one, so it must not pollute (egress, request.domain).
        if _origin(url) == request_origin:
            set_cookie = resp_headers.get("set-cookie")
            if set_cookie:
                captured.update(parse_set_cookie(set_cookie))
        if request.observer is not None:
            request.observer(status, resp_headers, url)

    curl = (
        curl_session(egress, request.domain, impersonate)
        if egress is not None and request.params.transport == "curl"
        else None
    )
    # Single cookie source to avoid a duplicated Cookie header: when a curl
    # session drives the request, ITS jar persists and resends cookies across the
    # coalesced connection. Both the stored profile cookies AND the caller
    # cookies are loaded INTO that jar (caller value overwriting any jar entry of
    # the same name), so exactly one value per cookie goes on the wire -- sending
    # a caller cookie via the Cookie header too would duplicate a name the jar
    # already holds (a bot tell). On the stdlib path (no jar) both are seeded
    # into the header.
    if curl is not None:
        _seed_session_jar(curl, request.domain, jar)
        _set_session_cookies(curl, request.domain, caller_cookies or {})
        seeded_cookies = None
    else:
        seeded_cookies = {**jar, **(caller_cookies or {})}
    body = request.send(
        headers=seeded_headers,
        cookies=seeded_cookies or None,
        raw_headers=False,
        on_response=capture,
        curl=curl,
    )
    if egress is None:
        return body  # Keyless: nothing to persist.
    store = ProfileStore.shared()
    if profile is None:
        store.save(egress, request.domain, Profile(ua=ua, cookies=captured))
    elif captured:
        store.update_cookies(egress, request.domain, captured)
    return body


def _send_via_zendriver(
    request: _Request,
    *,
    headers: dict[str, str] | None,
    cookies: dict[str, str] | None,
) -> bytes:
    """Fetch ``request`` through the headless-Chrome backend, warming the session.

    Reuses the identity layer's egress resolution and ProfileStore so a browser
    fetch and a curl fetch on the same ``(egress, domain)`` share cookies. The
    cookies the browser acquired are folded back through the per-hop observer, so
    the :class:`FetchSession` the caller receives is warm and a following curl
    fetch reuses them.
    """
    egress = egress_ip(cache=True) or egress_ip(cache=False)
    browser_url = _url_with_params(request.url, request.params.params)
    validated_hosts = request.params.validated_hosts

    def resolve_host(hostname: str) -> str:
        assert validated_hosts is not None
        return validated_hosts(hostname).ip

    result = fetch_zendriver(
        browser_url,
        profile_dir=default_profile_dir(),
        egress=egress or "",
        timeout_sec=request.params.timeout_sec,
        headers=headers,
        cookies=cookies,
        resolve_host=None if validated_hosts is None else resolve_host,
        on_redirect=request.params.on_redirect,
    )
    if result.cookies and request.observer is not None:
        # Fold the harvested jar into the returned session (and the caller's
        # on_response) as a synthesized Set-Cookie for this origin, so the
        # session warms exactly as a header-level backend's would.
        synthesized = "\n".join(f"{k}={v}" for k, v in result.cookies.items())
        request.observer(200, {"set-cookie": synthesized}, request.url)
    if egress is not None and request.domain and result.cookies:
        store = ProfileStore.shared()
        if store.load(egress, request.domain) is None:
            store.save(
                egress,
                request.domain,
                Profile(
                    ua=draw_user_agent(
                        kind_for_impersonate(request.session.impersonate)
                    ),
                    cookies=dict(result.cookies),
                ),
            )
        else:
            store.update_cookies(egress, request.domain, result.cookies)
    return result.body


# Live curl_cffi Sessions keyed by identity, so a session reuses one connection
# across requests -- the connection continuity a real browser has, and which a
# per-call request() (fresh TLS each time) lacks. Keyed on impersonate too.
# config-globals: ignore -- live pool of open connections, not a tunable.
_curl_sessions: dict[tuple[str, str, str], cc_requests.Session[Response]] = {}
# Guards EVERY mutation of _curl_sessions AND the _last_egress_ip global as one
# unit, so a VPN-roll close-sweep never races a pooled-session insert or drop.
_egress_lock = threading.Lock()  # config-globals: ignore -- shared observed state.


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
    with _egress_lock:
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


def _seed_session_jar(
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


def _set_session_cookies(
    session: cc_requests.Session[Response], domain: str, cookies: dict[str, str]
) -> None:
    """Set caller cookies into a curl session jar, OVERWRITING any prior value.

    Unlike :func:`_seed_session_jar` (which preserves live jar cookies), a caller
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
    with _egress_lock:
        session = _curl_sessions.pop(key, None)
    if session is not None:
        session.close()  # I/O outside the lock; the pop already removed it.


# Memoized from the most recent successful probe by any :func:`egress_ip` call,
# so the whole process shares one observed egress. ``None`` until the first
# successful probe; a failed probe leaves the prior value intact. A deliberate
# module global: shared *observed* state (a last-seen cache), not a tunable.
_last_egress_ip: str | None = None  # config-globals: ignore -- shared observed state.


def last_known_egress_ip() -> str | None:
    """Return the last-known egress IP without any network, or ``None`` if unseen."""
    with _egress_lock:
        return _last_egress_ip


# Callbacks fired when the egress rolls, so any transport that pools live,
# egress-keyed resources (open connections, running browsers) can invalidate
# them WITHOUT fetch.py naming that transport. fetch.py tears down its OWN curl
# pool inline; other backends (e.g. the zendriver browser pool) register here.
# config-globals: ignore -- observer registry, not a tunable.
_on_egress_rotation: list[Callable[[str | None], None]] = []


def on_egress_rotation(callback: Callable[[str | None], None]) -> None:
    """Register ``callback`` to run when the egress IP rolls (a VPN change).

    Called with the new IP whenever :func:`set_last_egress_ip` observes a change.
    A backend that pools egress-keyed live resources registers its teardown here,
    so this module never has to know that backend exists to invalidate it.
    """
    _on_egress_rotation.append(callback)


def set_last_egress_ip(ip: str | None) -> None:
    """Set the process-wide last-known egress IP (e.g. after a known VPN roll).

    A changed IP means the exit rolled, so every pooled Session for a DIFFERENT
    egress is now dead (its connection went out the old exit) -- close and drop
    them, leaving only the new egress's sessions. Registered rotation callbacks
    (:func:`on_egress_rotation`) then fire so other backends invalidate their own
    egress-keyed pools.
    """
    global _last_egress_ip  # noqa: PLW0603 -- memoize shared observed state.
    with _egress_lock:
        rolled = ip != _last_egress_ip
        if rolled:
            for key in [k for k in _curl_sessions if k[0] != ip]:
                _curl_sessions.pop(key).close()
        _last_egress_ip = ip
    if rolled:
        for callback in _on_egress_rotation:
            callback(ip)


# The zendriver browser pool is egress-keyed live state (running Chromes), so a
# roll invalidates it exactly like the curl pool. shutdown_browsers is a no-op
# when no pool exists, so this subscription is free until a browser fetch runs.
# Registered via the rotation hook so the teardown mechanism is uniform, not a
# special case wired into set_last_egress_ip.
on_egress_rotation(lambda _ip: shutdown_browsers())


def egress_ip(
    *,
    cache: bool = True,
    ipv6: bool = False,
    v4_echoes: Sequence[str] = (
        "https://ipv4.icanhazip.com",
        "https://api.ipify.org",
    ),
    v6_echoes: Sequence[str] = (
        "https://ipv6.icanhazip.com",
        "https://api64.ipify.org",
    ),
    timeout_sec: float = 5.0,
) -> str | None:
    """Return this host's public egress IP, or ``None`` if none resolves.

    Args:
      cache: When true (default), return the last-known value if set, probing
        only to fill it. False always probes live.
      ipv6: Resolve the IPv6 egress instead of IPv4.
      v4_echoes: v4-only echo hosts tried in order.
      v6_echoes: v6-only echo hosts tried in order.
      timeout_sec: Per-request HTTP timeout.

    Returns:
      ip: The public address of the requested family, or ``None`` when none
        resolves (offline, or no egress of that family).

    """
    if cache and (cached := last_known_egress_ip()) is not None:
        return cached
    echoes = v6_echoes if ipv6 else v4_echoes
    for url in echoes:
        try:
            # raw_headers bypasses the identity layer: the profile is keyed by
            # egress IP and resolving it is THIS call, so a profiled echo would
            # recurse. A bare GET is all an echo service needs.
            body, _ = fetch(
                url,
                request=RequestParams(
                    headers={}, raw_headers=True, timeout_sec=timeout_sec
                ),
            )
            ip = body.decode().strip()
        except (FetchError, OSError, ValueError):
            continue
        if _is_valid_ip_address(ip, ipv6=ipv6):
            set_last_egress_ip(ip)
            return ip
    return None


def _is_valid_ip_address(text: str, *, ipv6: bool) -> bool:
    """Whether *text* is a valid IP address of the requested family."""
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        return False
    return addr.version == (6 if ipv6 else 4)


def _url_with_params(
    url: str,
    params: Mapping[str, str | int] | None,
) -> str:
    """Return ``url`` with encoded query parameters appended."""
    if not params:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{urlencode(params)}"


def _fetch_once(
    url: str,
    params: RequestParams,
    *,
    headers: dict[str, str] | None,
    cookies: dict[str, str] | None,
    raw_headers: bool,
    impersonate: str,
    accept_ch: Mapping[str, frozenset[str]],
    on_response: _Observer | None,
    session: cc_requests.Session[Response] | None,
) -> bytes:
    """Build and send one request (with retries), no profile layer.

    The raw transport core: encodes query/body/headers and dispatches to the
    curl or stdlib path with the shared retry loop. The profile-aware
    :func:`fetch` wraps this; ``raw_headers`` callers reach it directly (no
    cookie jar). ``params`` supplies the validated per-call policy; ``headers`` /
    ``cookies`` / ``impersonate`` are resolved by the identity layer above;
    ``accept_ch`` is the session's per-origin extended-hint opt-ins;
    ``on_response`` may wrap ``params.on_response`` to capture cookies; and
    ``session`` is a pooled curl_cffi Session to reuse (the identity's persistent
    connection), or ``None`` to open a throwaway one.
    """
    url, basic_auth = _split_userinfo(url)
    url = _url_with_params(url, params.params)
    body_bytes: bytes | None = None
    body_content_type: str | None = None
    if params.data is not None:
        body_bytes = urlencode(params.data).encode()
        body_content_type = "application/x-www-form-urlencoded"
    elif params.json is not None:
        body_bytes = json_lib.dumps(params.json).encode()
        body_content_type = "application/json"
    merged = _build_headers(
        method=params.method,
        url=url,
        content_type=body_content_type,
        extra=headers,
        raw_headers=raw_headers,
        impersonate=impersonate,
        use_curl=params.transport == "curl",
        accept_ch=accept_ch,
    )
    if basic_auth is not None:
        merged.setdefault("Authorization", basic_auth)
    # HTTP header names are case-insensitive: collapse any caller-supplied
    # case-variant "cookie" header and the cookies= param into ONE Cookie key.
    # Two dict keys ("cookie" + "Cookie") would emit two Cookie lines on the wire
    # -- a bot tell. The param values follow the caller's header pairs.
    cookie_parts = [
        merged.pop(key) for key in [k for k in merged if k.lower() == "cookie"]
    ]
    if cookies:
        cookie_parts.append("; ".join(f"{k}={v}" for k, v in cookies.items()))
    if cookie_parts:
        merged["Cookie"] = "; ".join(cookie_parts)
    method = params.method
    backend = _fetch_curl if params.transport == "curl" else _fetch_stdlib
    for attempt in range(1 + params.retries):
        try:
            return backend(
                url,
                method=method,
                headers=merged,
                body=body_bytes,
                timeout_sec=params.timeout_sec,
                max_redirects=params.max_redirects,
                impersonate=impersonate,
                on_redirect=params.on_redirect,
                on_response=on_response,
                validated_hosts=params.validated_hosts,
                session=session,
            )
        except FetchError as e:
            # status 0 is the transport-failure sentinel (a curl CurlError, or a
            # connection/TLS failure wrapped by a transport) -- retryable like the
            # OSError below, which the stdlib path raises for the same class of
            # failure. Without this the two transports disagree on `retries=`.
            retryable = e.status in _RETRYABLE_STATUSES or e.status == 0
            if not retryable or attempt == params.retries:
                raise
            delay_sec = params.backoff_delay(attempt, e.headers)
            logger.debug(
                "fetch %s → %d, retry in %.1fs",
                url,
                e.status,
                delay_sec,
            )
            time.sleep(delay_sec)
        except (OSError, TimeoutError) as e:
            if attempt == params.retries:
                raise
            delay_sec = params.backoff_delay(attempt, {})
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
    extra: Mapping[str, str] | None,
    raw_headers: bool,
    impersonate: str,
    use_curl: bool,
    accept_ch: Mapping[str, frozenset[str]],
) -> dict[str, str]:
    """Build canonical-order Chrome request headers.

    On the curl transport, curl_cffi's ``impersonate`` supplies the coherent
    Chrome fingerprint (User-Agent, ``sec-ch-ua`` hints, Accept, Sec-Fetch-*,
    Priority) matching its TLS/HTTP-2 profile exactly. Overriding those with
    hand-built values makes the identities disagree -- a bot tell -- so the curl
    path emits ONLY the structural headers curl does not set (Origin/Content-Type
    on a POST), the extended client hints an origin opted into via ``Accept-CH``,
    and caller extras. The stdlib path has no impersonation, so it reproduces the
    full hand-built Chrome set to look like a browser at all.
    """
    # Host and Content-Length are omitted: http.client auto-adds both first on
    # the wire (the connection path overrides Host when validated_hosts splits
    # SNI/IP); curl adds them itself.
    if raw_headers:
        return dict(extra or {})
    if use_curl:
        return _curl_structural_headers(
            method=method,
            url=url,
            content_type=content_type,
            extra=extra,
            impersonate=impersonate,
            accept_ch=accept_ch,
        )
    # The stdlib path has no impersonation, so it reproduces the full Chrome
    # header set by hand -- from the SAME source (chrome_navigation_headers,
    # matched to the impersonate target) the curl path's fingerprint uses, so
    # the two transports present one coherent identity, not two drifting ones.
    parsed = urlparse(url)
    major, platform = impersonate_version_platform(impersonate)
    h = chrome_navigation_headers(
        major=major,
        platform=platform,
        method=method,
        content_type=content_type or "",
        origin=f"{parsed.scheme}://{parsed.netloc}",
    )
    if extra:
        # Caller wins; dict.update preserves slot for existing keys and
        # appends new ones at the end.
        h.update(extra)
    return h


def _origin(url: str) -> str:
    """The scheme://host[:port] origin of a URL (the Accept-CH opt-in key)."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _accept_ch_hints(resp_headers: dict[str, str]) -> frozenset[str]:
    """The extended client-hint names an ``Accept-CH`` response requested."""
    accept_ch = resp_headers.get("accept-ch")
    if accept_ch is None:
        return frozenset()
    wanted = {tok.strip().lower() for tok in accept_ch.split(",") if tok.strip()}
    return frozenset(name for name in chrome_client_hints(major=1) if name in wanted)


def _curl_structural_headers(
    *,
    method: str,
    url: str,
    content_type: str | None,
    extra: Mapping[str, str] | None,
    impersonate: str,
    accept_ch: Mapping[str, frozenset[str]],
) -> dict[str, str]:
    """Headers for the curl path: what impersonate omits + Accept-CH opt-ins.

    Verified on Linux against real Chrome (146, the impersonate target): the
    FIRST request to an
    origin sends only the core set curl_cffi's impersonate reproduces (UA, the
    three basic ``sec-ch-ua`` hints, Accept, Sec-Fetch-*, Priority,
    Accept-Encoding/Language). It adds the EXTENDED client hints
    (``sec-ch-ua-arch`` etc.) only AFTER the server opts in via ``Accept-CH``,
    on subsequent same-origin requests. We mirror that exactly: a cold origin
    gets nothing extra (adding hints an unrequesting site never asked for is
    itself a tell), and an origin that has sent Accept-CH gets precisely the
    hints it requested, version-matched to the impersonate target. Only a POST's
    Origin/Content-Type and caller extras follow (Cookie is merged by caller).
    """
    h: dict[str, str] = {}
    wanted = accept_ch.get(_origin(url))
    if wanted:
        major, platform = impersonate_version_platform(impersonate)
        hints = chrome_client_hints(major=major, platform=platform)
        h.update({name: value for name, value in hints.items() if name in wanted})
    if method not in ("GET", "HEAD"):
        if content_type:
            h["Content-Type"] = content_type
        parsed = urlparse(url)
        h["Origin"] = f"{parsed.scheme}://{parsed.netloc}"
    if extra:
        h.update(extra)
    return h


def _decompress(body: bytes, encoding: str) -> bytes:
    """Decompress a response body per Content-Encoding; raise ValueError if bad.

    ``Content-Encoding`` may chain several codings (RFC 9110 SS 8.4.1, e.g.
    ``gzip, br``); they are applied left-to-right on encode, so decode
    right-to-left. Each token is one coding.
    """
    for enc in reversed([tok.strip().lower() for tok in encoding.split(",")]):
        body = _decompress_one(body, enc)
    return body


def _decompress_one(body: bytes, enc: str) -> bytes:
    """Decompress ``body`` under a SINGLE Content-Encoding token."""
    if enc in ("", "identity"):
        return body
    try:
        if enc == "gzip":
            return gzip.decompress(body)
        if enc == "deflate":
            # RFC 7230 says zlib-wrapped, but some servers emit RAW DEFLATE (no
            # header); browsers retry with a negative window. Try zlib first,
            # fall back to raw so a header-less stream still decodes.
            try:
                return zlib.decompress(body)
            except zlib.error:
                return zlib.decompress(body, -zlib.MAX_WBITS)
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


def _curl_set_cookies(resp: Response) -> list[str]:
    """The individual ``Set-Cookie`` headers of a curl response, unfolded.

    ``resp.headers.items()`` folds duplicates with ``", "`` (lossy for cookies);
    ``get_list`` returns each header separately. Returns ``[]`` when the response
    set no cookie.
    """
    get_list = getattr(resp.headers, "get_list", None)
    if get_list is None:
        value = resp.headers.get("set-cookie")
        return [value] if value else []
    return list(cast("list[str]", get_list("set-cookie")))


def _join_headers(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Lowercase header pairs, folding duplicates into one value.

    Duplicates join with ``", "`` per RFC 9110 SS 5.3 -- EXCEPT ``set-cookie``,
    which that RFC explicitly exempts (a cookie value may itself contain ``", "``,
    so folding then re-splitting mis-parses it). Multiple ``set-cookie`` headers
    join with a newline instead -- a byte that never appears in a header value --
    so :func:`sagent.lib.web.profile.parse_set_cookie` can split them back exactly.
    """
    out: dict[str, str] = {}
    for k, v in pairs:
        key = k.lower()
        if key not in out:
            out[key] = v
        elif key == "set-cookie":
            out[key] = f"{out[key]}\n{v}"
        else:
            out[key] = f"{out[key]}, {v}"
    return out


@dataclass(slots=True, kw_only=True)
class _CurlLoop:
    """Mutable per-hop state shared by both curl backends' redirect loops.

    Holds the current URL, method, headers, body, and remaining redirect budget.
    :meth:`follow` runs the identical post-response decision both backends make:
    fire ``on_response``, and if the status is a followable redirect within
    budget, advance the state to the next hop (via :func:`_apply_redirect`) and
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
        on_response: _Observer | None,
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
        redirect_url = _redirect_target(self.url, status, resp_headers)
        if on_redirect is not None:
            on_redirect(redirect_url)
        self.headers, self.method, self.body = _apply_redirect(
            self.url, self.headers, self.method, self.body, status, redirect_url
        )
        self.url = redirect_url
        return True


def _fetch_curl(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout_sec: float,
    max_redirects: int,
    impersonate: str,
    on_redirect: Callable[[str], None] | None,
    on_response: _Observer | None,
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
    on_response: _Observer | None,
    session: cc_requests.Session[Response] | None = None,
) -> bytes:
    """High-level curl path: ``requests.request`` with manual redirects."""
    # requests auto-decompresses .content, so no _decompress call is needed.
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
        # individual headers, newline-joined to match _join_headers so
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
    on_response: _Observer | None,
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
            port = parsed.port or _default_port(parsed.scheme)
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
                [f"{hostname}:{port}:{_bracket_ipv6(validated.ip)}"],
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
                    _decompress_error_body(raw_body, resp_headers),
                )
            return _decompress(
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
    current_url: str,
    headers: dict[str, str],
    method: str,
    body: bytes | None,
    status: int,
    redirect_url: str,
) -> tuple[dict[str, str], str, bytes | None]:
    """Compute the (headers, method, body) for the next hop of a redirect.

    The ONE place the per-hop transform lives, called by every transport's
    redirect loop so the rules cannot drift, mirroring what a real browser does:

    - Rewrite ``Origin`` to the new target.
    - On 301/302/303 of a non-GET, convert to a bodyless GET dropping
      Content-Type (browsers downgrade all three; only 307/308 preserve the
      method -- that is why 307/308 exist).
    - On a CROSS-ORIGIN hop, drop every origin-bound header (``Cookie`` and the
      extended client hints), since those belong to the source origin and must
      not leak to the target. Same-origin hops keep them.

    Casing-insensitive throughout.
    """
    headers = _rewrite_origin(headers, redirect_url)
    if status in (301, 302, 303) and method != "GET":
        method = "GET"
        body = None
        headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
    if _origin(current_url) != _origin(redirect_url):
        headers = {k: v for k, v in headers.items() if k.lower() not in _ORIGIN_BOUND}
    return headers, method, body


# Headers scoped to the origin that set/opted-into them; dropped on a
# cross-origin redirect so the source origin's Cookie and extended client hints
# never leak to the target (a real browser scopes both per origin).
_ORIGIN_BOUND: frozenset[str] = frozenset(
    {"cookie"} | {name.lower() for name in chrome_client_hints(major=1)}
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


def _fetch_stdlib(
    url: str,
    *,
    method: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout_sec: float,
    max_redirects: int,
    impersonate: str,
    on_redirect: Callable[[str], None] | None,
    on_response: _Observer | None,
    validated_hosts: ValidatedHosts | None,
    session: cc_requests.Session[Response] | None = None,
) -> bytes:
    """Stdlib transport: http.client with manual redirect following.

    A drop-in peer of :func:`_fetch_curl` with the identical signature, so
    :func:`_fetch_once` dispatches to either by name. This backend has no TLS
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
                request_headers, method, body = _apply_redirect(
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
