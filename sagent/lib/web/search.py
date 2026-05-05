"""Web search backends with a shared synchronous API.

DuckDuckGo is always available. Source-only builds include additional
configured and scraped backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

import logging
import re

from sagent.lib.lazy_import import lazy_import
from sagent.lib.web.fetch import fetch


if TYPE_CHECKING:
    import bs4
else:
    bs4 = lazy_import("bs4")


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


# Internal builds keep extra backends; public exports keep only DuckDuckGo.
SearchBackends: TypeAlias = Literal["duckduckgo"]
DEFAULT_SEARCH_BACKEND: SearchBackends = "duckduckgo"


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchResult:
    """A single search result."""

    url: str
    title: str
    snippet: str


class CaptchaError(Exception):
    """Raised when a backend returns a CAPTCHA/sorry page."""


_CLEAN_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")


def _strip_scripts(tag: bs4.Tag | bs4.BeautifulSoup) -> None:
    """Remove all ``<script>`` elements from the tree in place."""
    for script in tag.find_all("script"):
        script.decompose()


def _clean_text(text: str) -> str:
    """Collapse whitespace runs and drop spaces before punctuation."""
    return _CLEAN_SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(text.split()))


# ---------------------------------------------------------------------------
# DuckDuckGo
# ---------------------------------------------------------------------------

_DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"


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
    body = fetch(
        _DUCKDUCKGO_URL,
        method="POST",
        data={"q": query, "b": ""},
        headers=headers,
    )
    return _duckduckgo_parse(body.decode("utf-8"), num_results)


def _duckduckgo_parse(
    page_html: str,
    max_results: int,
) -> list[SearchResult]:
    """Extract search results from DDG's HTML."""
    soup = bs4.BeautifulSoup(page_html, "html.parser")
    _strip_scripts(soup)
    results: list[SearchResult] = []

    for div in soup.select("div.result"):
        a = div.select_one("a.result__a")
        if a is None:
            continue
        href = a.get("href", "")
        if not isinstance(href, str) or not href.startswith("http"):
            continue
        title = _clean_text(a.get_text(separator=" ", strip=True))
        if not title:
            continue

        snippet_el = div.select_one("a.result__snippet")
        snippet = (
            _clean_text(snippet_el.get_text(separator=" ", strip=True))
            if snippet_el
            else ""
        )

        results.append(
            SearchResult(url=href, title=title, snippet=snippet),
        )
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
    if backend == "duckduckgo":
        return duckduckgo(query, num_results, headers)

    raise ValueError(f"Unknown backend: {backend!r}")  # pyright: ignore[reportUnreachable] -- reachable in OSS build after google branch is stripped
