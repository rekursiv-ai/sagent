"""Web search backends with a shared synchronous API.

DuckDuckGo is always available. Source-only builds include additional
configured and scraped backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias
from urllib.parse import parse_qs, urlencode, urlparse

import hashlib
import json
import logging
import os
import re
import urllib.error

from sagent.lib.lazy_import import lazy_import
from sagent.lib.web.fetch import FetchError, fetch


if TYPE_CHECKING:
    import bs4
else:
    bs4 = lazy_import("bs4")


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


# Internal builds keep extra backends; public exports keep only DuckDuckGo.
_BACKEND_NAMES = Literal["duckduckgo", "searxng"]
DEFAULT_SEARCH_BACKEND: SearchBackends = "duckduckgo"
SearchBackends: TypeAlias = _BACKEND_NAMES  # noqa: UP040 -- type keyword breaks get_args() at runtime


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchResult:
    """A single search result."""

    url: str
    title: str
    snippet: str


class CaptchaError(Exception):
    """Raised when a backend returns a CAPTCHA/sorry page."""


class SearchError(RuntimeError):
    """Raised when a search backend fails before returning results."""


_CLEAN_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


def _strip_scripts(tag: bs4.Tag | bs4.BeautifulSoup) -> None:
    """Remove all ``<script>`` elements from the tree in place."""
    for script in tag.find_all("script"):
        script.decompose()


def _clean_text(text: str) -> str:
    """Collapse whitespace runs and drop spaces before punctuation."""
    return _CLEAN_SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(text.split()))


_GSA_USERAGENTS_PATH = Path(__file__).with_name("gsa_useragents.txt")


@cache
def _get_gsa_useragents() -> tuple[str, ...]:
    """Return cached GSA user-agent strings, loading on first call."""
    return tuple(line for line in _GSA_USERAGENTS_PATH.read_text().splitlines() if line)


def _gsa_headers_for_query(query: str) -> dict[str, str]:
    """Build request headers with a query-stable GSA mobile UA."""
    useragents = _get_gsa_useragents()
    idx = int.from_bytes(hashlib.sha256(query.encode()).digest()[:8]) % len(useragents)
    return {"User-Agent": f"{useragents[idx]} NSTNWV"}


# ---------------------------------------------------------------------------
# SearXNG
# ---------------------------------------------------------------------------

_SEARXNG_URL_ENV = "SEARXNG_URL"


def searxng(
    query: str,
    num_results: int = 10,
    headers: dict[str, str] | None = None,
) -> list[SearchResult]:
    """Query a SearXNG instance and return parsed JSON results.

    Args:
      query: Search query string.
      num_results: Maximum results to return.
      headers: Optional override headers forwarded to fetch.

    Returns:
      results: Parsed search results.

    """
    base_url = _searxng_url()
    params = urlencode({"q": query, "format": "json", "pageno": "1"})
    url = f"{base_url}/search?{params}"
    body = fetch(url, headers=headers, timeout_sec=10)
    items = json.loads(body).get("results", [])
    return [
        SearchResult(
            url=item.get("url", ""),
            title=_clean_text(item.get("title", "")),
            snippet=_clean_text(item.get("content", "")),
        )
        for item in items[:num_results]
    ]


def _searxng_url() -> str:
    """Return the configured SearXNG base URL without a trailing slash."""
    url = os.environ.get(_SEARXNG_URL_ENV, "").rstrip("/")
    if not url:
        raise RuntimeError(
            f"{_SEARXNG_URL_ENV} must be set to use SearXNG search",
        )
    return url


# ---------------------------------------------------------------------------
# DuckDuckGo
# ---------------------------------------------------------------------------

_DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
_DUCKDUCKGO_MAX_QUERY_CHARS = 499


def duckduckgo(
    query: str,
    num_results: int = 10,
    headers: dict[str, str] | None = None,
) -> list[SearchResult]:
    """Scrape DuckDuckGo's HTML-only endpoint.

    More reliable than Google scraping -- DDG doesn't block as
    aggressively.

    Args:
      query: Search query string.
      num_results: Maximum results to return.
      headers: Optional override headers forwarded to fetch.

    Returns:
      results: Parsed search results.

    """
    if len(query) > _DUCKDUCKGO_MAX_QUERY_CHARS:
        raise SearchError(
            f"DuckDuckGo query exceeds {_DUCKDUCKGO_MAX_QUERY_CHARS} characters "
            f"(got {len(query)})."
        )
    request_headers = _gsa_headers_for_query(query) | {
        "Accept": "*/*",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Accept-Language": "all,all-ALL;q=0.7",
        "Referer": _DUCKDUCKGO_URL,
    }
    if headers:
        request_headers.update(headers)
    # Send exactly these headers: the GSA mobile User-Agent must not be paired
    # with fetch's default desktop Chrome ``sec-ch-ua``/``sec-ch-ua-platform``,
    # whose drift from the UA can trip DuckDuckGo's bot check.
    body = fetch(
        _DUCKDUCKGO_URL,
        method="POST",
        data={"q": _duckduckgo_quote_bangs(query), "b": "", "kl": "wt-wt"},
        headers=request_headers,
        raw_headers=True,
        retries=2,
    )
    html = body.decode("utf-8")
    _duckduckgo_check_captcha(html)
    return _duckduckgo_parse(html, num_results)


def _duckduckgo_quote_bangs(query: str) -> str:
    """Quote DDG bang tokens to keep them in ordinary web search."""
    return " ".join(
        f"'{token}'" if token.startswith("!") else token for token in query.split()
    )


def _duckduckgo_check_captcha(page_html: str) -> None:
    """Raise when DDG returns its challenge page."""
    soup = bs4.BeautifulSoup(page_html, "html.parser")
    if soup.select_one("form#challenge-form") is not None:
        raise CaptchaError("DuckDuckGo returned a CAPTCHA challenge.")


def _duckduckgo_extract_url(href: str) -> str | None:
    """Extract a usable URL from DDG result links."""
    if not href:
        return None
    url = f"https:{href}" if href.startswith("//") else href
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    if (
        hostname == "duckduckgo.com" or hostname.endswith(".duckduckgo.com")
    ) and parsed.path == "/l/":
        wrapped = parse_qs(parsed.query).get("uddg", [])
        if wrapped:
            return wrapped[0]
    if parsed.scheme in {"http", "https"}:
        return url
    return None


def _duckduckgo_parse(
    page_html: str,
    max_results: int,
) -> list[SearchResult]:
    """Extract search results from DDG's HTML."""
    soup = bs4.BeautifulSoup(page_html, "html.parser")
    _strip_scripts(soup)
    results: list[SearchResult] = []
    for container in soup.select("div#links > div.web-result"):
        link = container.select_one("h2 a[href]")
        if link is None:
            continue
        href = link.get("href", "")
        if not isinstance(href, str):
            continue
        url = _duckduckgo_extract_url(href)
        if url is None:
            continue
        title = _clean_text(link.get_text(separator=" ", strip=True))
        if not title:
            continue

        snippet_el = container.select_one("a.result__snippet")
        snippet = (
            _clean_text(snippet_el.get_text(separator=" ", strip=True))
            if snippet_el is not None
            else ""
        )
        results.append(SearchResult(url=url, title=title, snippet=snippet))
        if len(results) >= max_results:
            break

    if not results:
        logger.warning(
            "No results parsed -- DDG may have changed markup.",
        )
    return results


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------


def search(
    query: str,
    backend: SearchBackends | None = None,
    num_results: int = 10,
    headers: dict[str, str] | None = None,
) -> list[SearchResult]:
    """Dispatch to the named search backend.

    Args:
      query: Search query string.
      backend: Backend name. Defaults to ``DEFAULT_SEARCH_BACKEND``.
      num_results: Maximum results to return.
      headers: Optional override headers forwarded to the backend.

    """
    if backend is None:
        backend = DEFAULT_SEARCH_BACKEND
    try:
        if backend == "duckduckgo":
            return duckduckgo(query, num_results, headers)
        if backend == "searxng":
            return searxng(query, num_results, headers)

    except (
        FetchError,
        OSError,
        TimeoutError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ) as e:
        raise SearchError(f"{backend} search failed: {e}") from e
    raise ValueError(f"Unknown backend: {backend!r}")  # pyright: ignore[reportUnreachable] -- reachable at runtime
