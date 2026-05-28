"""WebFetch tool: fetch a URL (GET or POST) and extract its content."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, cast
from urllib.parse import quote, urlparse
from xml.etree.ElementTree import Element, ParseError

import asyncio
import html
import ipaddress
import json
import re
import socket

import cachetools
import defusedxml.common
import defusedxml.ElementTree

from sagent.lib.json import JSON, JSONValue, json_freeze, json_unfreeze
from sagent.lib.lazy_import import lazy_import
from sagent.lib.web.fetch import FetchError, ValidatedHost, fetch
from sagent.tools.core import (
    TOOL_RESULT_MAX_CHARS,
    load_tool_description,
    truncate,
)
from sagent.tools.lib.bash import Node, unwrap_cd_prefix
from sagent.types.runtime import ToolResult


trafilatura = lazy_import("trafilatura")

# Response kinds returned by ``_fetch_body``; controls extraction.
_KIND_HTML = "html"  # raw HTML, needs trafilatura
_KIND_REDDIT = "reddit_thread"  # Reddit JSON, needs comment formatter
_KIND_REDDIT_LISTING = "reddit_listing"  # Reddit listing JSON, needs post formatter
_KIND_MARKDOWN = "markdown"  # already-extracted markdown (reader proxy)
_KIND_RSS = "rss"  # RSS 2.0 XML, needs feed formatter

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

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Fetch the URL, extract main content, and return as text.

        Args:
          args: Directive with ``url`` and optional ``method`` / ``json``
              / ``form`` keys.

        Returns:
          result: Extracted text body, or a fetch/extraction error.

        """
        raw_url = str(args.get("url", ""))
        method = str(args.get("method", "GET")).upper()
        if method not in ("GET", "POST"):
            return ToolResult(
                call_id="",
                content=f"Unsupported method {method!r}; only GET and POST allowed.",
                is_error=True,
            )

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
        except (FetchError, ValueError, OSError) as e:
            return ToolResult(call_id="", content=f"Fetch failed: {e}", is_error=True)

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
    method: str,
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
    return None, {
        str(k): str(v) for k, v in cast(dict[str, Any], json_unfreeze(raw_form)).items()
    }


async def _extract_text(body: bytes, *, kind: str, method: str) -> str:
    """Extract tool result text from a response body.

    ``kind`` selects the post-processing path:
      - ``_KIND_REDDIT``: parse as Reddit thread JSON.
      - ``_KIND_REDDIT_LISTING``: parse as Reddit post-listing JSON.
      - ``_KIND_RSS``: parse as RSS 2.0 XML and format as markdown.
      - ``_KIND_MARKDOWN``: return as-is (the reader-proxy rung already
        rendered to markdown; running trafilatura on it would strip
        structure).
      - ``_KIND_HTML``: trafilatura main-content extraction, with a
        raw-content fallback when extraction returns nothing.
    """
    content = body.decode("utf-8", errors="replace")
    if kind == _KIND_REDDIT:
        try:
            data = json.loads(content)
        except (ValueError, TypeError):
            return content[:TOOL_RESULT_MAX_CHARS]
        return _format_reddit_json(data)
    if kind == _KIND_REDDIT_LISTING:
        try:
            data = json.loads(content)
        except (ValueError, TypeError):
            return content[:TOOL_RESULT_MAX_CHARS]
        return _format_reddit_listing(data)
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
    method: str,
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
    method: str,
    json_body: JSONValue,
    form_body: dict[str, str] | None,
) -> tuple[bytes, str]:
    """Fetch ``url`` through a bot-wall-aware fallback ladder.

    The stdlib HTTP path (via ``_safe_fetch``) is always tried first.
    On a 403/429/503 response to a GET -- the signature of edge-side
    bot detection (Fastly, Akamai, Cloudflare) -- the ladder falls
    through to additional fetch strategies and finally to a reader-
    proxy hop that renders the URL with a third-party browser stack.
    Non-GET methods, non-fallback statuses (404, 500, ...), and SSRF /
    DNS errors surface immediately; the ladder only engages on the
    specific bot-wall signature.

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

    # Reader-proxy fallback (final rung).
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
    method: str = "GET",
    json_body: JSONValue = None,
    form_body: dict[str, str] | None = None,
) -> bytes:
    """Fetch with SSRF check on the initial URL and every redirect."""
    _check_ssrf(url)
    return fetch(
        url,
        method=method,
        json=json_body,
        data=form_body,
        on_redirect=_check_ssrf,
        validated_hosts=_validated_host,
        timeout_sec=15,
    )


def _check_ssrf(url: str) -> None:
    """Raise if ``url`` resolves to a non-public address."""
    err = _url_is_safe(url)
    if err is not None:
        raise ValueError(err)


def _validated_host(netloc: str) -> ValidatedHost:
    """Return a host/IP pair after SSRF validation."""
    parsed = urlparse(f"//{netloc}")
    host = parsed.hostname
    if not host:
        raise ValueError("URL has no host.")
    err = _url_is_safe(f"http://{netloc}")
    if err is not None:
        raise ValueError(err)
    infos = socket.getaddrinfo(host, None)
    ip = str(infos[0][4][0])
    err = _ip_is_safe(host, ip)
    if err is not None:
        raise ValueError(err)
    return ValidatedHost(host=netloc, ip=ip)


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
    """Reddit threads via the JSON API; non-thread pages with old.reddit fallback."""

    _THREAD_RE = re.compile(
        r"^https?://(?:\w+\.)?reddit\.com/r/\w+/comments/\w+",
    )
    _LISTING_RE = re.compile(
        r"^https?://(?:\w+\.)?reddit\.com/r/[^/?#]+/(?:new|hot|top|rising|controversial)?/?\.json(?:[?#].*)?$",
    )

    def matches(self, url: str) -> bool:
        """Match ``reddit.com`` and any subdomain (``old``, ``np``, ``new``)."""
        hostname = urlparse(url).hostname or ""
        return hostname == "reddit.com" or hostname.endswith(".reddit.com")

    async def fetch(self, url: str) -> tuple[bytes, str]:
        """Fetch a Reddit URL via JSON view or HTML with verification fallback."""
        if self._THREAD_RE.search(url):
            json_url = re.sub(
                r"^(https?://)(?:\w+\.)?reddit\.com/",
                r"\1www.reddit.com/",
                url.rstrip("/"),
            )
            if not json_url.endswith(".json"):
                json_url += ".json"
            body = await asyncio.to_thread(_safe_fetch, json_url)
            return body, _KIND_REDDIT
        if self._LISTING_RE.search(url):
            body = await asyncio.to_thread(_safe_fetch, url)
            return body, _KIND_REDDIT_LISTING

        try:
            body = await asyncio.to_thread(_safe_fetch, url)
        except FetchError as e:
            if not _is_reddit_verification_page(e.body):
                raise
            return await self._fetch_old_reddit(url), _KIND_HTML
        if _is_reddit_verification_page(body):
            return await self._fetch_old_reddit(url), _KIND_HTML
        return body, _KIND_HTML

    @staticmethod
    async def _fetch_old_reddit(raw_url: str) -> bytes:
        """Fetch a Reddit HTML page through the legacy host."""
        url = re.sub(
            r"^(https?://)(?:www\.)?reddit\.com/",
            r"\1old.reddit.com/",
            raw_url,
        )
        try:
            return await asyncio.to_thread(_safe_fetch, url)
        except (FetchError, ValueError, OSError) as e:
            raise ValueError(
                "Reddit returned a JavaScript verification page for "
                f"{raw_url}; old Reddit fallback failed: {e}"
            ) from e


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


class _XAdapter:
    """X (Twitter) -- full SPA with no useful SSR; route via reader proxy.

    X serves an empty shell to non-JS clients; tweet text only appears
    after a JS hydration step. We delegate the render to the existing
    reader-proxy rung (Jina) and return its markdown. This is the same
    third-party hop used by the bot-wall fallback; the adapter makes
    that hop the unconditional default for x.com / twitter.com URLs
    rather than a last-resort retry.

    External dependency: every fetch flows through ``r.jina.ai``.
    """

    def matches(self, url: str) -> bool:
        """Match ``x.com``, ``twitter.com``, and their subdomains."""
        hostname = urlparse(url).hostname or ""
        if hostname in ("x.com", "twitter.com"):
            return True
        return hostname.endswith((".x.com", ".twitter.com"))

    async def fetch(self, url: str) -> tuple[bytes, str]:
        """Fetch via reader proxy and tag as already-extracted markdown."""
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


def _is_reddit_verification_page(content: bytes) -> bool:
    """Return whether Reddit returned its JavaScript verification page."""
    text = content.decode("utf-8", errors="replace")
    return "Reddit - Please wait for verification" in text and "js_challenge" in text


def _format_reddit_json(data: list[Any] | dict[str, Any]) -> str:
    """Extract readable text from Reddit's JSON API response."""
    listings = data if isinstance(data, list) else [data]
    lines: list[str] = []
    if listings and listings[0].get("kind") == "Listing":
        posts = listings[0].get("data", {}).get("children", [])
        if posts:
            post = posts[0].get("data", {})
            lines.append(f"# {post.get('title', '')}")
            lines.append(
                f"by u/{post.get('author', '[deleted]')} "
                f"({post.get('score', 0)} points)\n"
            )
            if post.get("selftext", ""):
                lines.append(str(post.get("selftext", "")))
                lines.append("")
    if len(listings) > 1 and listings[1].get("kind") == "Listing":
        comments = listings[1].get("data", {}).get("children", [])
        _format_reddit_comments(comments, lines, depth=0)
    return "\n".join(lines)


def _format_reddit_listing(data: list[Any] | dict[str, Any]) -> str:
    """Extract readable text from Reddit listing JSON."""
    listing_obj: object = data[0] if isinstance(data, list) and data else data
    if not isinstance(listing_obj, dict):
        return ""
    listing = cast(dict[str, object], listing_obj)
    listing_data_obj = listing.get("data", {})
    if not isinstance(listing_data_obj, dict):
        return ""
    listing_data = cast(dict[str, object], listing_data_obj)
    children_obj = listing_data.get("children", [])
    if not isinstance(children_obj, list):
        return ""
    children = cast(list[object], children_obj)
    lines = ["# Reddit listing", ""]
    for idx, child_obj in enumerate(children, start=1):
        if not isinstance(child_obj, dict):
            continue
        child = cast(dict[str, object], child_obj)
        if child.get("kind") != "t3":
            continue
        post_obj = child.get("data", {})
        if not isinstance(post_obj, dict):
            continue
        post = cast(dict[str, object], post_obj)
        created_raw = post.get("created_utc", 0)
        if not isinstance(created_raw, (int, float, str)):
            created_raw = 0
        created = datetime.fromtimestamp(float(created_raw), UTC)
        permalink = str(post.get("permalink", ""))
        reddit_url = f"https://www.reddit.com{permalink}" if permalink else ""
        lines.append(f"{idx}. {post.get('title', '')}")
        lines.append(
            f"   - u/{post.get('author', '[deleted]')} | "
            f"{post.get('score', 0)} points | "
            f"{post.get('num_comments', 0)} comments | {created.isoformat()}"
        )
        flair = post.get("link_flair_text")
        if flair:
            lines.append(f"   - flair: {flair}")
        if reddit_url:
            lines.append(f"   - reddit: {reddit_url}")
        outbound = str(post.get("url", ""))
        if outbound and outbound != reddit_url:
            lines.append(f"   - link: {outbound}")
        selftext = " ".join(str(post.get("selftext", "")).split())
        if selftext:
            lines.append(f"   - excerpt: {selftext[:500]}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _format_reddit_comments(children: list[Any], lines: list[str], depth: int) -> None:
    """Append formatted Reddit comment JSON to ``lines``."""
    indent = "  " * depth
    for child in children:
        if child.get("kind") != "t1":
            continue
        comment = child.get("data", {})
        lines.append(
            f"{indent}**u/{comment.get('author', '[deleted]')}** "
            f"({comment.get('score', 0)} pts):"
        )
        lines.extend(
            f"{indent}  {body_line}"
            for body_line in comment.get("body", "").splitlines()
        )
        lines.append("")
        replies = comment.get("replies")
        if isinstance(replies, dict):
            replies_dict = cast(dict[str, Any], replies)
            reply_children = replies_dict.get("data", {}).get("children", [])
            _format_reddit_comments(reply_children, lines, depth + 1)


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
    """Format an RSS 2.0 feed as readable markdown.

    Each ``<item>`` becomes a section: the lead headline as an ``##``
    heading with source and pub date, followed by bullet-listed sibling
    stories parsed from the (Google-News-style) ``<ol>`` embedded in
    the item's ``<description>``. Feeds without cluster descriptions
    degrade to one heading per item.

    Args:
      body: Raw feed XML.

    Returns:
      formatted: Markdown text suitable for direct tool output.

    """
    try:
        root = defusedxml.ElementTree.fromstring(body)
    except (ParseError, defusedxml.common.DefusedXmlException):
        return body.decode("utf-8", errors="replace")[:TOOL_RESULT_MAX_CHARS]
    channel = root.find("channel") if root.tag == "rss" else root
    if channel is None:
        return body.decode("utf-8", errors="replace")[:TOOL_RESULT_MAX_CHARS]
    lines: list[str] = []
    feed_title = (channel.findtext("title") or "").strip()
    if feed_title:
        lines.append(f"# {feed_title}\n")
    for item in channel.findall("item"):
        _append_rss_item(item, lines)
    return "\n".join(lines).rstrip()


def _append_rss_item(item: Element, lines: list[str]) -> None:
    """Append one feed item (heading + meta + cluster bullets) to ``lines``."""
    title = (item.findtext("title") or "").strip()
    link = (item.findtext("link") or "").strip()
    source_elem = item.find("source")
    source = (source_elem.text or "").strip() if source_elem is not None else ""
    pub_date = (item.findtext("pubDate") or "").strip()
    if title:
        lines.append(f"## {title}")
    meta_parts = [p for p in (source, pub_date) if p]
    if meta_parts:
        lines.append(" -- ".join(meta_parts))
    if link:
        lines.append(link)
    # The first cluster entry duplicates the item title; siblings follow.
    cluster = _parse_rss_cluster(item.findtext("description") or "")
    for sibling_title, sibling_link, sibling_source in cluster[1:]:
        suffix = f" -- {sibling_source}" if sibling_source else ""
        lines.append(f"- [{sibling_title}]({sibling_link}){suffix}")
    lines.append("")


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
