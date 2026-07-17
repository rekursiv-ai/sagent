"""WebFetch tool: fetch a URL (GET or POST) and extract its content."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, cast
from urllib.parse import quote, urlparse
from xml.etree.ElementTree import Element, ParseError

import asyncio
import html
import ipaddress
import os
import re
import socket

from wrapt import lazy_import

import cachetools
import defusedxml.common
import defusedxml.ElementTree

from sagent.lib.custom_json import JSON, JSONValue, json_freeze, json_unfreeze
from sagent.lib.web.errors import BotDetectionError, FetchError, classify_bot_detection
from sagent.lib.web.fetch import RequestParams, ValidatedHost, fetch
from sagent.tools.core import (
    TOOL_RESULT_MAX_CHARS,
    load_tool_description,
    truncate,
)
from sagent.tools.lib.bash import Node, unwrap_cd_prefix
from sagent.types.runtime import ToolResult


trafilatura = lazy_import("trafilatura")

# The only HTTP methods this tool supports; enforced at the directive boundary.
HttpMethod = Literal["GET", "POST"]

# Response kinds returned by ``_fetch_body``; controls extraction.
_KIND_HTML = "html"  # raw HTML, needs trafilatura
_KIND_MARKDOWN = "markdown"  # already-extracted markdown (reader proxy)
_KIND_RSS = "rss"  # RSS 2.0 / Atom feed XML, needs feed formatter
# HTTP statuses that trigger the bot-wall fallback ladder. Other
# 4xx/5xx (404, 410, 451, 500, ...) are not signs of bot detection and
# surface to the caller immediately.
_FALLBACK_STATUSES: frozenset[int] = frozenset({403, 429, 503})

# Reader-proxy fallback endpoint. Jina AI's free Reader API takes a
# target URL as a path segment and returns clean markdown rendered by
# its own browser stack -- which gets past bot walls without us
# touching TLS-layer details. URL templated so the proxy host can be
# swapped (self-hosted, alternate provider) by overriding this module
# attribute.
_READER_PROXY_TEMPLATE = "https://r.jina.ai/{url}"

# Sentinel embedded in the markdown body when Jina's own backend got
# bot-walled. Returned with HTTP 200, so we have to detect at the
# content level. Matches both ``Target URL returned error 4xx`` and
# ``Target URL returned error 5xx``.
_READER_PROXY_SOFT_FAIL_RE = re.compile(
    rb"Warning:\s*Target URL returned error \d{3}",
    re.IGNORECASE,
)


class WebFetch:
    """Fetch a web page and extract its main content as clean text."""

    name: str = "WebFetch"
    tool_id: str = "application/x-tool-webfetch"
    clearable_results: bool = True
    description: str = load_tool_description("WebFetch")
    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."},
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "description": (
                        "HTTP method. Defaults to GET. Use POST to call"
                        " JSON or form APIs."
                    ),
                },
                "json": {
                    "description": (
                        "JSON-serializable body for POST requests. Sets"
                        " Content-Type: application/json. Mutually"
                        " exclusive with 'form'."
                    ),
                },
                "form": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": (
                        "Form fields for POST requests. Sets Content-Type:"
                        " application/x-www-form-urlencoded. Mutually"
                        " exclusive with 'json'."
                    ),
                },
            },
            "required": ["url"],
        }
    )

    def __init__(self) -> None:
        self._cache = cachetools.TTLCache[str, str](maxsize=128, ttl=15 * 60)

    def bash_match(self, trees: Sequence[Node]) -> str | None:
        """Emit a tool-use nudge for ``curl URL`` / ``wget URL``.

        Accepts ``cd PATH && CMD`` compounds via ``unwrap_cd_prefix``.
        Bails on output redirection (``-o``/``-O``/``--output``),
        ``--data-binary @file`` style file uploads, and any non-http(s)
        URL -- those are cases WebFetch can't cleanly replace. Pipelines
        and stdout redirects are already filtered by ``unwrap_cd_prefix``.

        Args:
          trees: Parsed bashlex command trees from the active Bash call.

        Returns:
          hint: Nudge string redirecting to the WebFetch tool, or ``None``.

        """
        unwrapped = unwrap_cd_prefix(trees)
        if unwrapped is None:
            return None
        _cwd, cmd = unwrapped
        if cmd.exe not in {"curl", "wget"} or cmd.env_prefix:
            return None
        return _match_http_fetch(cmd.exe, cmd.args)

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short display label for this invocation.

        Args:
          args: Directive carrying ``url``.

        Returns:
          label: ``WebFetch <url>`` line shown before invocation.

        """
        url = str(args.get("url", ""))
        if len(url) > 60:
            url = url[:57] + "..."
        return f"WebFetch {url}"

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for WebFetch.

        Args:
          result: Completed ``ToolResult`` (ignored).

        Returns:
          receipt: Always ``None`` (no receipt line).

        """
        del result
        return None

    def prompt(self) -> str:
        """Return no supplemental system-prompt text for WebFetch.

        Returns:
          contribution: Empty string.

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: independent network fetch, no serialization."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Fetch the URL, extract main content, and return as text.

        GET responses are cached per-URL for 15 minutes. The cache key
        is the raw URL alone, so a URL whose server-side extraction
        path changes during the TTL window (e.g. a Reddit page that
        starts returning a different shape and falls into a different
        adapter) will continue to serve the previously-extracted body
        until the entry expires. Callers that need to bypass a stale
        cache entry can switch to ``POST`` or wait out the TTL.

        Args:
          args: Directive with ``url`` and optional ``method`` / ``json``
              / ``form`` keys.

        Returns:
          result: Extracted text body, or a fetch/extraction error.

        """
        raw_url = str(args.get("url", ""))
        raw_method = str(args.get("method", "GET")).upper()
        if raw_method not in ("GET", "POST"):
            return ToolResult(
                call_id="",
                content=(
                    f"Unsupported method {raw_method!r}; only GET and POST allowed."
                ),
                is_error=True,
            )
        # The guard above admits only "GET"/"POST", narrowing raw_method to
        # the HttpMethod literal.
        method: HttpMethod = raw_method

        try:
            json_body, form_body = _request_bodies(method, args)
        except ValueError as e:
            return ToolResult(call_id="", content=str(e), is_error=True)

        cache_key = raw_url if method == "GET" else None
        if cache_key is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return ToolResult(call_id="", content=cached)

        try:
            body, kind = await _fetch_body(
                raw_url,
                method=method,
                json_body=json_body,
                form_body=form_body,
            )
        except BotDetectionError as e:
            # fetch() classified the block at the boundary: surface the SPECIFIC
            # kind (Cloudflare vs puzzle vs Google /sorry), each with its own
            # actionable guidance, rather than a generic "HTTP 403".
            return ToolResult(call_id="", content=e.explain(raw_url), is_error=True)
        except (FetchError, ValueError, OSError) as e:
            return ToolResult(call_id="", content=f"Fetch failed: {e}", is_error=True)

        # A block/challenge page can arrive as apparent success on any rung -- a
        # reader proxy returns Cloudflare's "security check" HTML with HTTP 200,
        # or a site soft-blocks with a 200 body. Detect it on the raw body (the
        # markers survive there even if extraction strips the title) and surface
        # it as an error, so block-page prose is never rendered as the document.
        flag = classify_bot_detection(body, on_success_body=True)
        if flag is not None:
            # Surface the SPECIFIC kind of block (Cloudflare vs puzzle vs Google
            # /sorry), each with its own actionable guidance, rather than a
            # generic "blocked" -- the class knows what it is and how to clear it.
            return ToolResult(call_id="", content=flag.explain(raw_url), is_error=True)

        text = await _extract_text(
            body,
            kind=kind,
            method=method,
        )
        truncated = truncate(text, TOOL_RESULT_MAX_CHARS)
        if cache_key is not None:
            self._cache[cache_key] = truncated
        return ToolResult(call_id="", content=truncated)


def _request_bodies(
    method: HttpMethod,
    args: Mapping[str, object],
) -> tuple[JSONValue, dict[str, str] | None]:
    """Return POST request bodies from a tool directive."""
    if method != "POST":
        return None, None
    raw_json = args.get("json")
    raw_form = args.get("form")
    if raw_json is not None and raw_form is not None:
        raise ValueError("'json' and 'form' are mutually exclusive.")
    if raw_json is not None:
        return json_unfreeze(raw_json), None
    if raw_form is None:
        return None, None
    unfrozen_form = json_unfreeze(raw_form)
    # Schema declares ``form`` as an object, but LLM-supplied
    # directives can violate the schema (``form=[]`` slipped through
    # historically). Reject anything not a mapping so the downstream
    # ``.items()`` call can't AttributeError out of the tool envelope.
    # ``ValueError`` (not ``TypeError``) so it joins the existing
    # caller-side ``except ValueError`` envelope in ``WebFetch.run``.
    if not isinstance(unfrozen_form, dict):
        raise ValueError(  # noqa: TRY004 -- caller catches ValueError uniformly.
            f"'form' must be an object of string fields, got {type(unfrozen_form).__name__}."
        )
    return None, {
        str(k): str(v) for k, v in cast(dict[str, Any], unfrozen_form).items()
    }


async def _extract_text(body: bytes, *, kind: str, method: HttpMethod) -> str:
    """Extract tool result text from a response body.

    ``kind`` selects the post-processing path:
      - ``_KIND_RSS``: parse as RSS 2.0 / Atom XML and format as markdown.
      - ``_KIND_MARKDOWN``: return as-is (the reader-proxy rung already
        rendered to markdown; running trafilatura on it would strip
        structure).
      - ``_KIND_HTML``: trafilatura main-content extraction, with a
        raw-content fallback when extraction returns nothing.
    """
    content = body.decode("utf-8", errors="replace")

    if kind == _KIND_RSS:
        return _format_rss(body)
    if kind == _KIND_MARKDOWN:
        return content[:TOOL_RESULT_MAX_CHARS]
    if method == "POST" or content.lstrip().startswith(("{", "[")):
        return content[:TOOL_RESULT_MAX_CHARS]
    extracted = await asyncio.to_thread(
        trafilatura.extract,
        content,
        include_links=True,
        include_tables=True,
    )
    return extracted or content[:TOOL_RESULT_MAX_CHARS]


async def _fetch_body(
    raw_url: str,
    *,
    method: HttpMethod,
    json_body: JSONValue,
    form_body: dict[str, str] | None,
) -> tuple[bytes, str]:
    """Fetch a URL and classify the response for downstream extraction.

    GET requests are first offered to each ``HostAdapter`` in
    ``_ADAPTERS``; the first ``matches`` returning True takes over the
    fetch and owns its own retry/fallback policy. URLs with no matching
    adapter (and all non-GET requests) fall through to
    ``_fetch_with_fallback``, a multi-rung ladder that handles bot-wall
    403/429/503 responses transparently.

    Args:
      raw_url: Target URL to fetch.
      method: HTTP method (``GET`` or ``POST``).
      json_body: JSON-serializable body for POST requests.
      form_body: Form-encoded body for POST requests.

    Returns:
      body: Raw response bytes.
      kind: One of the ``_KIND_*`` constants; selects the
        post-processing branch in ``_extract_text``.

    """
    if method == "GET":
        for adapter in _ADAPTERS:
            if adapter.matches(raw_url):
                return await adapter.fetch(raw_url)
    return await asyncio.to_thread(
        _fetch_with_fallback,
        raw_url,
        method=method,
        json_body=json_body,
        form_body=form_body,
    )


def _fetch_with_fallback(
    url: str,
    *,
    method: HttpMethod,
    json_body: JSONValue,
    form_body: dict[str, str] | None,
) -> tuple[bytes, str]:
    """Fetch ``url`` through a bot-wall-aware fallback ladder.

    The direct path (``_safe_fetch`` -> :func:`sagent.lib.web.fetch.fetch`,
    Chrome TLS/HTTP-2 impersonation) is always tried first. On a 403/429/503
    response to a GET -- the signature of edge-side bot detection (Fastly,
    Akamai, Cloudflare) -- the ladder falls through to a reader-proxy hop that
    renders the URL with a third-party browser stack (a different egress).
    Non-GET methods, non-fallback statuses (404, 500, ...), and SSRF / DNS
    errors surface immediately; the ladder only engages on the specific
    bot-wall signature.

    Args:
      url: Target URL (SSRF-checked by ``_safe_fetch`` on each rung).
      method: HTTP method; the fallback path is GET-only.
      json_body: POST JSON body (initial-rung only).
      form_body: POST form body (initial-rung only).

    Returns:
      body_kind: ``(bytes, kind)`` where kind is ``_KIND_HTML`` from
        the HTTP rungs and ``_KIND_MARKDOWN`` from the reader proxy.

    Raises:
      FetchError: The original error, if every rung fails.

    """
    try:
        body = _safe_fetch(
            url,
            method=method,
            json_body=json_body,
            form_body=form_body,
        )
        return body, _KIND_HTML
    except FetchError as e:
        if e.status not in _FALLBACK_STATUSES or method != "GET":
            raise
        rung1_err = e

    # Reader-proxy fallback (final rung). The first rung already fetches with
    # Chrome TLS/HTTP-2 impersonation via sagent.lib.web.fetch, so a same-egress
    # curl retry would present an identical fingerprint and hit the same wall;
    # the proxy is the only rung with a genuinely different egress.
    try:
        return _reader_proxy_fetch(url), _KIND_MARKDOWN
    except (FetchError, ValueError, OSError) as e:
        raise rung1_err from e


def _reader_proxy_fetch(url: str) -> bytes:
    """Fetch ``url`` through the r.jina.ai reader proxy.

    The proxy receives the target URL as a path segment, fetches it
    with a full browser stack, and returns clean markdown. The proxy
    URL itself goes through ``_safe_fetch`` so SSRF guards still apply
    to the proxy host. The user URL travels as encoded path data; the
    proxy -- not us -- is the one that contacts the target server.

    Jina returns HTTP 200 even when its own backend was bot-walled,
    embedding the diagnostic as a ``Warning:`` line in the markdown.
    We detect that sentinel and raise ``FetchError`` so the ladder
    treats it as a soft failure instead of handing the agent text
    that looks like an article but is actually a proxy diagnostic.
    """
    proxy_url = _READER_PROXY_TEMPLATE.format(url=quote(url, safe=":/"))
    body = _safe_fetch(proxy_url)
    if _READER_PROXY_SOFT_FAIL_RE.search(body):
        raise FetchError(
            url=url,
            status=502,
            headers={},
            body=body[:200],
        )
    return body


def _safe_fetch(
    url: str,
    *,
    method: HttpMethod = "GET",
    json_body: JSONValue = None,
    form_body: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> bytes:
    """Fetch with SSRF check on the initial URL and every redirect.

    Delegates the transport -- Chrome TLS/HTTP-2 impersonation, redirect
    following, retry, decompression -- to :func:`sagent.lib.web.fetch.fetch`. This
    tool supplies only the app-level SSRF policy (via the ``validated_hosts`` and
    ``on_redirect`` hooks) and, when a caller passes ``headers`` (e.g. Reddit's
    Android app User-Agent + bearer token), the identity to present.
    """
    _check_ssrf(url)
    body, _session = fetch(
        url,
        request=RequestParams(
            method=method,
            json=json_body,
            data=form_body,
            headers=headers,
            on_redirect=_check_ssrf,
            validated_hosts=_validated_host,
            timeout_sec=15,
        ),
    )
    return body


def _check_ssrf(url: str) -> None:
    """Raise if ``url`` resolves to a non-public address."""
    err = _url_is_safe(url)
    if err is not None:
        raise ValueError(err)


def _validated_host(hostname: str) -> ValidatedHost:
    """Return a host/IP pair after SSRF validation.

    ``hostname`` is the bare host the ``validated_hosts`` resolver contract
    passes (never a netloc-with-port). Pins the connect IP to defeat DNS
    rebinding. Prefers an IPv4 address when the host resolves to both families:
    ``getaddrinfo`` commonly lists AAAA (v6) first, but many hosts/networks have
    no working v6 route, and pinning to a single unreachable v6 address turns a
    servable page into a status-0 connection failure (unlike an un-pinned
    client, which Happy-Eyeballs to v4). Falls back to v6 only when that is the
    sole family.
    """
    parsed = urlparse(f"//{hostname}")
    host = parsed.hostname
    if not host:
        raise ValueError("URL has no host.")
    err = _url_is_safe(f"http://{hostname}")
    if err is not None:
        raise ValueError(err)
    infos = socket.getaddrinfo(host, None)
    ips = [str(info[4][0]) for info in infos]
    # Prefer the first IPv4; else the first address of any family.
    ip = next((a for a in ips if ":" not in a), ips[0] if ips else "")
    err = _ip_is_safe(host, ip)
    if err is not None:
        raise ValueError(err)
    # Return the BARE host (never the raw netloc-with-port): the transport
    # re-appends any port via _host_header, so returning a port here would
    # double it on the wire.
    return ValidatedHost(host=host, ip=ip)


def _url_is_safe(url: str) -> str | None:
    """Return an error string if ``url`` is unsafe to fetch, else None."""
    try:
        parsed = urlparse(url)
    except ValueError as e:
        return f"Invalid URL: {e}"
    if parsed.scheme not in ("http", "https"):
        return f"Unsupported scheme {parsed.scheme!r}; only http(s) allowed."
    host = parsed.hostname
    if not host:
        return "URL has no host."
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError) as e:
        return f"DNS resolution failed for {host!r}: {e}"
    for info in infos:
        err = _ip_is_safe(host, str(info[4][0]))
        if err is not None:
            return err
    return None


def _ip_is_safe(host: str, raw_ip: str) -> str | None:
    """Return an error string if ``raw_ip`` is unsafe to fetch, else None."""
    try:
        ip = ipaddress.ip_address(raw_ip)
    except ValueError:
        return None
    if (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return f"Refusing to fetch {host!r} (resolves to non-public address {ip})."
    return None


class HostAdapter(Protocol):
    """Per-host fetch override for sites that need bespoke retrieval.

    The dispatcher in ``_fetch_body`` walks ``_ADAPTERS`` in order and
    delegates to the first adapter whose ``matches`` returns True. The
    adapter then owns the full fetch, including any host-specific
    fallbacks (Reddit's JS-verification → old.reddit hop, X's renderer
    delegation). The dispatcher does not second-guess: an adapter that
    raises propagates its exception up through ``WebFetch.run``.
    """

    def matches(self, url: str) -> bool:
        """Return True iff this adapter handles ``url``."""
        ...

    async def fetch(self, url: str) -> tuple[bytes, str]:
        """Fetch ``url`` through host-specific logic.

        Args:
          url: Target URL (already SSRF-checked by the inner call to
            ``_safe_fetch``).

        Returns:
          body: Raw response bytes.
          kind: One of the ``_KIND_*`` constants identifying which
            ``_extract_text`` branch to use.

        """
        ...


class _RedditAdapter:
    """Reddit via the public RSS/Atom feed.

    Reddit bot-walls every datacenter-IP request to its JSON API
    (``.json``), HTML pages, and ``old.reddit.com`` with a 403 wall;
    reader proxies are blocked too. The ``.rss`` feed still serves
    anonymously: a thread feed carries top-level comments as Atom
    entries, so we keep titles, bodies, and comment text. Every Reddit
    URL is normalized to its ``.rss`` form and parsed by ``_format_rss``.
    """

    _THREAD_RE = re.compile(r"/r/[^/?#]+/comments/\w+")

    def matches(self, url: str) -> bool:
        """Match ``reddit.com`` and any subdomain (``old``, ``np``, ``new``)."""
        hostname = urlparse(url).hostname or ""
        return hostname == "reddit.com" or hostname.endswith(".reddit.com")

    async def fetch(self, url: str) -> tuple[bytes, str]:
        """Fetch the Reddit feed (RSS), or richer JSON when configured."""
        body = await asyncio.to_thread(_safe_fetch, _rss_url(url))
        return body, _KIND_RSS


# Google News front-page paths that route to the top-stories RSS feed.
# Article-detail URLs (``/articles/...``) and topic URLs (``/topics/...``)
# fall through unrewritten -- topic IDs don't map cleanly between the
# SPA and RSS URL spaces, and article pages have no RSS equivalent.
_GOOGLE_NEWS_TOP_PATHS: frozenset[str] = frozenset(
    {"", "/home", "/topstories", "/foryou"},
)


class _GoogleNewsAdapter:
    """Rewrite ``news.google.com`` SPA URLs to their RSS endpoints.

    The SPA's server-side render only contains a sparse "Your briefing"
    block; the bulk of the page is hydrated by JS we don't execute. The
    public RSS feed at the same hostname serves the full set of story
    clusters as structured XML, which ``_format_rss`` renders cleanly.
    """

    def matches(self, url: str) -> bool:
        """Match the exact ``news.google.com`` hostname (no subdomains)."""
        return urlparse(url).hostname == "news.google.com"

    async def fetch(self, url: str) -> tuple[bytes, str]:
        """Fetch via RSS if the path has a known rewrite, else HTML."""
        rewritten = self._rewrite(url)
        target = rewritten if rewritten is not None else url
        body = await asyncio.to_thread(_safe_fetch, target)
        kind = _KIND_RSS if urlparse(target).path.startswith("/rss") else _KIND_HTML
        return body, kind

    @staticmethod
    def _rewrite(url: str) -> str | None:
        """Return the RSS-equivalent URL, or None for paths we leave alone."""
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        if path.startswith("/rss"):
            return None
        if path in _GOOGLE_NEWS_TOP_PATHS:
            return parsed._replace(path="/rss").geturl()
        if path == "/search":
            return parsed._replace(path="/rss/search").geturl()
        return None


_ALLOW_THIRD_PARTY_RENDER_ENV = "SAGENT_ALLOW_THIRD_PARTY_RENDER"


def _third_party_render_allowed() -> bool:
    """Return whether the operator has opted into third-party rendering.

    Default: refuse. X / Twitter content is fetched via ``r.jina.ai``
    (a third-party renderer); a privacy-sensitive caller cannot opt
    out at the URL level, so default to refusing the hop and require
    explicit consent via the environment variable.
    """
    value = os.environ.get(_ALLOW_THIRD_PARTY_RENDER_ENV, "").strip().lower()
    return value in ("1", "true", "yes", "on")


class _XAdapter:
    """X (Twitter) -- full SPA with no useful SSR; route via reader proxy.

    X serves an empty shell to non-JS clients; tweet text only appears
    after a JS hydration step. We delegate the render to the existing
    reader-proxy rung (Jina) and return its markdown. Because every
    fetch egresses to ``r.jina.ai``, this adapter is opt-in: callers
    must set ``SAGENT_ALLOW_THIRD_PARTY_RENDER=1`` or the fetch raises
    ``FetchError``.
    """

    def matches(self, url: str) -> bool:
        """Match ``x.com``, ``twitter.com``, and their subdomains."""
        hostname = urlparse(url).hostname or ""
        if hostname in ("x.com", "twitter.com"):
            return True
        return hostname.endswith((".x.com", ".twitter.com"))

    async def fetch(self, url: str) -> tuple[bytes, str]:
        """Fetch via reader proxy and tag as already-extracted markdown."""
        if not _third_party_render_allowed():
            message = (
                f"X/Twitter fetch requires the third-party reader proxy"
                f" ({_READER_PROXY_TEMPLATE.format(url='...')});"
                f" set {_ALLOW_THIRD_PARTY_RENDER_ENV}=1 to allow."
            )
            raise FetchError(url, 0, {}, message.encode("utf-8"))
        body = await asyncio.to_thread(_reader_proxy_fetch, url)
        return body, _KIND_MARKDOWN


# Registry of host adapters. Order matters: the first matching adapter
# wins. New entries should be added in expected-match-frequency order
# so a popular host doesn't pay for failed matches against niche ones.
_ADAPTERS: tuple[HostAdapter, ...] = (
    _RedditAdapter(),
    _GoogleNewsAdapter(),
    _XAdapter(),
)


def _rss_url(raw_url: str) -> str:
    """Return the ``.rss`` feed URL for any Reddit URL.

    Rewrites the path to end in ``.rss`` (dropping a trailing ``.json``
    if present) while preserving the query string, which carries feed
    options like ``?limit=100`` and ``?sort=top``. URLs whose path
    already ends in ``.rss`` are returned unchanged.
    """
    parsed = urlparse(raw_url)
    if parsed.path.endswith(".rss"):
        return raw_url
    path = re.sub(r"/?\.json$", "", parsed.path).rstrip("/")
    return parsed._replace(path=f"{path}/.rss").geturl()


# Matches one ``<li>`` entry in a Google News RSS cluster description.
# The description body is a small fragment of HTML with the same shape
# every time: ``<ol><li><a href="..">title</a> &nbsp;&nbsp;<font ..>source
# </font></li>...</ol>``. We parse with a regex rather than an HTML
# parser because the fragment is well-formed-by-construction and the
# regex stays under ten lines.
_RSS_CLUSTER_LINK_RE = re.compile(
    r'<li>\s*<a\s+[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
    r"(?:[^<]*<font[^>]*>([^<]+)</font>)?",
    re.IGNORECASE | re.DOTALL,
)


def _format_rss(body: bytes) -> str:
    """Format an RSS or Atom feed as readable markdown.

    Args:
      body: Raw feed XML.

    Returns:
      formatted: Markdown text suitable for direct tool output.

    """
    try:
        root = defusedxml.ElementTree.fromstring(body)
    except (ParseError, defusedxml.common.DefusedXmlException):
        return body.decode("utf-8", errors="replace")[:TOOL_RESULT_MAX_CHARS]
    if _local_name(root.tag) == "feed":
        return _format_atom(root).rstrip()
    channel = root.find("channel") if _local_name(root.tag) == "rss" else root
    if channel is None:
        return body.decode("utf-8", errors="replace")[:TOOL_RESULT_MAX_CHARS]
    lines: list[str] = []
    feed_title = (_child_text(channel, "title") or "").strip()
    if feed_title:
        lines.append(f"# {feed_title}\n")
    for item in _children(channel, "item"):
        _append_rss_item(item, lines)
    return "\n".join(lines).rstrip()


def _format_atom(feed: Element[str]) -> str:
    """Format an Atom feed as readable markdown."""
    lines: list[str] = []
    feed_title = (_child_text(feed, "title") or "").strip()
    if feed_title:
        lines.append(f"# {feed_title}\n")
    for entry in _children(feed, "entry"):
        _append_atom_entry(entry, lines)
    return "\n".join(lines)


def _append_atom_entry(entry: Element[str], lines: list[str]) -> None:
    """Append one Atom entry to ``lines``."""
    title = (_child_text(entry, "title") or "").strip()
    author = _atom_author(entry)
    updated = (
        _child_text(entry, "updated") or _child_text(entry, "published") or ""
    ).strip()
    link = _atom_link(entry)
    content = _atom_content(entry)
    if title:
        lines.append(f"## {title}")
    meta_parts = [p for p in (author, updated) if p]
    if meta_parts:
        lines.append(" -- ".join(meta_parts))
    if link:
        lines.append(link)
    if content:
        lines.append(content[:500])
    lines.append("")


def _atom_author(entry: Element[str]) -> str:
    """Return the Atom author name, if present."""
    author = _child(entry, "author")
    if author is None:
        return ""
    return (_child_text(author, "name") or "").strip()


def _atom_link(entry: Element[str]) -> str:
    """Return the first Atom link href, if present."""
    for link in _children(entry, "link"):
        href = link.attrib.get("href")
        if href:
            return href.strip()
    return ""


def _atom_content(entry: Element[str]) -> str:
    """Return cleaned Atom content or summary text."""
    raw = _child_text(entry, "content") or _child_text(entry, "summary") or ""
    return " ".join(re.sub(r"<[^>]+>", " ", html.unescape(raw)).split())


def _append_rss_item(item: Element[str], lines: list[str]) -> None:
    """Append one feed item (heading + meta + cluster bullets) to ``lines``."""
    title = (_child_text(item, "title") or "").strip()
    link = (_child_text(item, "link") or "").strip()
    source_elem = _child(item, "source")
    source = (source_elem.text or "").strip() if source_elem is not None else ""
    pub_date = (_child_text(item, "pubDate") or "").strip()
    if title:
        lines.append(f"## {title}")
    meta_parts = [p for p in (source, pub_date) if p]
    if meta_parts:
        lines.append(" -- ".join(meta_parts))
    if link:
        lines.append(link)
    # The first cluster entry duplicates the item title; siblings follow.
    cluster = _parse_rss_cluster(_child_text(item, "description") or "")
    for sibling_title, sibling_link, sibling_source in cluster[1:]:
        suffix = f" -- {sibling_source}" if sibling_source else ""
        lines.append(f"- [{sibling_title}]({sibling_link}){suffix}")
    lines.append("")


def _local_name(tag: str) -> str:
    """Return an XML tag without its namespace."""
    return tag.rsplit("}", 1)[-1]


def _children(parent: Element[str], name: str) -> list[Element[str]]:
    """Return direct children with local tag name ``name``."""
    return [child for child in list(parent) if _local_name(child.tag) == name]


def _child(parent: Element[str], name: str) -> Element[str] | None:
    """Return the first direct child with local tag name ``name``."""
    for child in list(parent):
        if _local_name(child.tag) == name:
            return child
    return None


def _child_text(parent: Element[str], name: str) -> str | None:
    """Return text for the first direct child with local tag name ``name``."""
    child = _child(parent, name)
    return child.text if child is not None else None


def _parse_rss_cluster(description_html: str) -> list[tuple[str, str, str]]:
    """Parse the ``<ol>`` of sibling stories embedded in a feed item description.

    Args:
      description_html: HTML fragment from an ``<item><description>``.

    Returns:
      entries: ``(title, link, source)`` tuples, in document order. Empty
        list when the fragment lacks Google-News-style cluster markup.

    """
    return [
        (
            html.unescape(match.group(2)).strip(),
            match.group(1),
            html.unescape(match.group(3) or "").strip(),
        )
        for match in _RSS_CLUSTER_LINK_RE.finditer(description_html)
    ]


_NUDGE = "curl/wget via Bash is a bad UX. Use the WebFetch tool."
_HTTP_FETCH_BAIL_FLAGS: frozenset[str] = frozenset(
    {
        "-o",
        "-O",
        "--output",
        "--output-document",
        "--data-binary",
        "--upload-file",
        "-T",
        "-F",
        "--form",
    }
)


def _match_http_fetch(exe: str, args: tuple[str, ...]) -> str | None:
    """Return a nudge when a shell command is a simple HTTP fetch."""
    del exe
    url_count = 0
    for arg in args:
        if arg in _HTTP_FETCH_BAIL_FLAGS:
            return None
        if arg.startswith(("http://", "https://")):
            url_count += 1
    if url_count != 1:
        return None
    return _NUDGE
