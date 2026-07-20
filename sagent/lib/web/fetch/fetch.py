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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import unquote, urlencode, urlparse

import base64
import ipaddress
import json as json_lib
import logging
import random
import threading
import time

from sagent.lib.custom_json import JSONValue
from sagent.lib.web.chrome.headers import (
    chrome_client_hints,
    chrome_navigation_headers,
    impersonate_version_platform,
)
from sagent.lib.web.chrome.useragents import draw_user_agent, kind_for_impersonate
from sagent.lib.web.errors import BotDetectionError, FetchError
from sagent.lib.web.fetch import transport_routing
from sagent.lib.web.fetch.common import Observer, ValidatedHosts
from sagent.lib.web.fetch.curl import (
    close_curl_session,
    close_curl_sessions_except,
    curl_session,
    fetch_curl,
    seed_session_jar,
    set_session_cookies,
)
from sagent.lib.web.fetch.stdlib import fetch_stdlib
from sagent.lib.web.profile import (
    Profile,
    ProfileStore,
    parse_set_cookie,
    parsedate_to_datetime_or_none,
)


if TYPE_CHECKING:
    from curl_cffi import requests as cc_requests
    from curl_cffi.requests import Response
    from curl_cffi.requests.session import HttpMethod

    import sagent.lib.web.fetch.zendriver as zendriver_backend
else:
    from wrapt import lazy_import

    zendriver_backend = lazy_import("sagent.lib.web.fetch.zendriver")


__all__ = [
    "FetchSession",
    "RequestParams",
    "Transport",
    "egress_ip",
    "fetch",
    "last_known_egress_ip",
    "on_egress_rotation",
    "resolve_transport",
    "set_last_egress_ip",
]

logger = logging.getLogger(__name__)

Transport = Literal["auto", "curl", "curl-then-zendriver", "zendriver", "stdlib"]


def resolve_transport(
    transport: Transport,
    *,
    method: HttpMethod = "GET",
    raw_headers: bool = False,
    has_body: bool = False,
) -> Transport:
    """Resolve ``auto`` to a concrete transport for this request."""
    if transport != "auto":
        return transport
    if method != "GET" or raw_headers or has_body:
        return "curl"
    return "curl-then-zendriver"


_RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


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
      body_validator: Called with every final response body before it is accepted.
        Raise :class:`BotDetectionError` when a provider-specific success body
        proves browser or human interaction is required; automatic transport
        fallback then learns the domain.
      validated_hosts: Resolver returning a validated IP per hostname; receives
        the bare hostname and must resolve it to the same IP for the call. Browser
        transports reject this option because Chrome owns its DNS connections.
      transport: Retrieval transport. ``"auto"`` (default) selects
        curl-then-Zendriver for eligible GETs and curl for requests a browser
        cannot replay or requests using ``validated_hosts``. ``"curl"`` is the
        curl_cffi impersonated path; ``"stdlib"`` is the http.client reference path;
        ``"zendriver"`` drives a real headless Chrome via
        :mod:`sagent.lib.web.fetch.zendriver` (opt-in, for JS/challenge-walled
        pages); ``"curl-then-zendriver"`` tries curl first and falls back to zendriver ONLY when
        curl is bot-blocked (a :class:`BotDetectionError`) -- a non-block failure
        propagates unchanged. ``"zendriver"`` and ``"curl-then-zendriver"`` are
        GET-only and reject raw-header mode.

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
    body_validator: Callable[[bytes], None] | None = None
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
    observer: Observer | None = None

    @property
    def domain(self) -> str:
        """The hostname the session identity is keyed on (``""`` if hostless)."""
        return urlparse(self.url).hostname or ""

    def fetch(self) -> tuple[bytes, FetchSession]:
        """Perform the request; return the body and the updated session."""
        p = self.params
        if p.transport == "auto" and p.validated_hosts is not None:
            resolved: Transport = "curl"
        elif (
            p.transport == "auto"
            and self.domain in transport_routing.zendriver_domains()
        ):
            resolved = "zendriver"
        else:
            resolved = resolve_transport(
                p.transport,
                method=p.method,
                raw_headers=p.raw_headers,
                has_body=p.data is not None or p.json is not None,
            )
        if resolved in ("zendriver", "curl-then-zendriver") and (
            p.validated_hosts is not None
        ):
            raise ValueError(
                f"The {resolved} transport cannot honor 'validated_hosts'."
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
        on_response: Observer | None = None,
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


def _validated_body(request: _Request, body: bytes) -> bytes:
    """Validate and return ``body`` using the caller's provider-specific rule."""
    if request.params.body_validator is not None:
        request.params.body_validator(body)
    return body


def _fetch_with_identity(
    request: _Request,
    *,
    caller_headers: dict[str, str] | None,
    caller_cookies: dict[str, str] | None,
) -> bytes:
    """Send ``request`` under the stored per-``(egress, domain)`` identity."""
    if request.params.transport == "zendriver":
        return _validated_body(
            request,
            _send_via_zendriver(
                request,
                headers=caller_headers,
                cookies=caller_cookies,
            ),
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
            if request.domain:
                transport_routing.remember_zendriver_domain(request.domain)
            return _validated_body(
                request,
                _send_via_zendriver(
                    request,
                    headers=caller_headers,
                    cookies=caller_cookies,
                ),
            )
    domain = request.domain
    if not domain:
        return _validated_body(
            request,
            _send_as(request, None, None, caller_headers, caller_cookies),
        )

    store = ProfileStore.shared()
    # A cheap last-known egress finds an existing session; a NEW session must pin
    # to the live egress so a stale last-known never mis-keys the identity.
    egress = egress_ip(cache=True)
    profile = store.load(egress, domain) if egress is not None else None
    if profile is None:
        egress = egress_ip(cache=False)

    try:
        return _validated_body(
            request,
            _send_as(request, profile, egress, caller_headers, caller_cookies),
        )
    except BotDetectionError:
        if profile is None:
            raise  # First contact burned: no known identity to discard or retry.
        assert egress is not None  # A profile only loads once egress resolved.
        store.discard(egress, domain)
        close_curl_session(egress, domain, request.session.impersonate)
        # The burn may be a VPN rotation: re-resolve live before the fresh retry.
        return _validated_body(
            request,
            _send_as(
                request, None, egress_ip(cache=False), caller_headers, caller_cookies
            ),
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
        seed_session_jar(curl, request.domain, jar)
        set_session_cookies(curl, request.domain, caller_cookies or {})
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
    result = zendriver_backend.fetch_zendriver(
        browser_url,
        profile_dir=zendriver_backend.default_profile_dir(),
        egress=egress or "",
        timeout_sec=request.params.timeout_sec,
        headers=headers,
        cookies=cookies,
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


_egress_lock = threading.Lock()  # config-globals: ignore -- guards egress state.

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
        _last_egress_ip = ip
    if rolled:
        close_curl_sessions_except(ip)
        for callback in _on_egress_rotation:
            callback(ip)


# The zendriver browser pool is egress-keyed live state (running Chromes), so a
# roll invalidates it exactly like the curl pool. shutdown_browsers is a no-op
# when no pool exists, so this subscription is free until a browser fetch runs.
# Registered via the rotation hook so the teardown mechanism is uniform, not a
# special case wired into set_last_egress_ip.
on_egress_rotation(lambda _ip: zendriver_backend.shutdown_browsers())


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


def _split_userinfo(url: str) -> tuple[str, str | None]:
    """Strip ``user:pass@`` from a URL; return the URL and Basic auth."""
    parsed = urlparse(url)
    if not (parsed.username or parsed.password):
        return url, None
    user = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    credentials = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
    netloc = parsed.netloc[parsed.netloc.rfind("@") + 1 :]
    return parsed._replace(netloc=netloc).geturl(), f"Basic {credentials}"


def _fetch_once(
    url: str,
    params: RequestParams,
    *,
    headers: dict[str, str] | None,
    cookies: dict[str, str] | None,
    raw_headers: bool,
    impersonate: str,
    accept_ch: Mapping[str, frozenset[str]],
    on_response: Observer | None,
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
    backend = fetch_curl if params.transport == "curl" else fetch_stdlib
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
