"""Web search backends with a shared synchronous API.

DuckDuckGo is always available. Source-only builds include additional
configured and scraped backends.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias, cast, overload
from urllib.parse import parse_qs, urlencode, urlparse

import gzip
import hashlib
import json
import logging
import os
import re
import urllib.error

from sagent.lib.custom_json import (
    datetime_val,
    float_val,
    int_val,
    str_list_val,
    str_map_val,
    str_val,
)
from sagent.lib.web.fetch import FetchError, fetch


if TYPE_CHECKING:
    import bs4
else:
    from wrapt import lazy_import

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
    """A single web search result."""

    url: str
    title: str
    snippet: str


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperResult(SearchResult):
    """A scholarly result from SearXNG's ``science`` category (``paper.html``).

    Subclasses :class:`SearchResult` so any web-result consumer works
    unchanged: ``url``/``title`` carry through and ``snippet`` holds the
    abstract. The added fields surface the structured bibliographic metadata
    SearXNG's science engines (Semantic Scholar, OpenAlex, PubMed, arXiv,
    Crossref, ...) emit. Fields absent from a given engine's response default
    to empty.

    Attributes:
      citations: Citation count when the engine reported one. SearXNG renders
        it as a humanized string in the ``comments`` field (e.g. ``"42
        citations"``); the integer is recovered when parseable, else ``None``.

    """

    authors: tuple[str, ...] = ()
    journal: str = ""
    doi: str = ""
    pdf_url: str = ""
    published: datetime | None = None
    tags: tuple[str, ...] = ()
    citations: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageResult(SearchResult):
    """An image result from SearXNG's ``images`` category (``images.html``).

    ``url`` is the source page; ``image_url`` the full image. ``snippet`` holds
    any caption. Fields absent from a given engine default to empty.
    """

    image_url: str = ""
    thumbnail_url: str = ""
    resolution: str = ""
    img_format: str = ""
    source: str = ""
    filesize: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class MediaResult(SearchResult):
    """A time-based media result -- the ``news`` and ``music`` categories.

    Base for any result with a temporal/playable payload: a publish date, an
    embed or audio URL, a duration, a thumbnail. The ``news`` and ``music``
    tabs emit the generic ``default.html`` template but carry these fields,
    which a bare :class:`SearchResult` drops. :class:`VideoResult` extends this
    with view count and channel. Fields absent from a given engine default to
    empty.
    """

    published: datetime | None = None
    audio_url: str = ""
    iframe_url: str = ""
    length: str = ""
    thumbnail_url: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoResult(MediaResult):
    """A video result from SearXNG's ``videos`` category (``videos.html``).

    Extends :class:`MediaResult` (shared publish date, embed URL, duration,
    thumbnail) with the video-specific view count and channel. The
    ``videos.html`` template adds no fields beyond SearXNG's result base, but
    those base fields are real structure a web result discards. Fields absent
    from a given engine default to empty.
    """

    views: str = ""
    author: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class MapResult(SearchResult):
    """A place result from SearXNG's ``map`` category (``map.html``).

    ``snippet`` holds any description. Coordinates and the structured address
    surface the geographic payload. Fields absent from a given engine default
    to empty / ``None``.

    Attributes:
      address: Structured address components keyed by SearXNG's field names
        (``name``, ``road``, ``house_number``, ``locality``, ``postcode``,
        ``country``); empty when the engine returned none.

    """

    latitude: float | None = None
    longitude: float | None = None
    address: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True, slots=True, kw_only=True)
class PackageResult(SearchResult):
    """A software-package result from the ``it`` category (``packages.html``).

    ``snippet`` holds the package description. Fields absent from a given
    engine default to empty.
    """

    package_name: str = ""
    version: str = ""
    maintainer: str = ""
    license_name: str = ""
    homepage: str = ""
    source_code_url: str = ""
    popularity: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class CodeResult(SearchResult):
    """A source-code result from the ``it`` category (``code.html``).

    ``snippet`` holds the matched code or description. Fields absent from a
    given engine default to empty.
    """

    repository: str = ""
    filename: str = ""
    code_language: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class FileResult(SearchResult):
    """A file result from SearXNG's ``files`` category (``file.html``).

    ``snippet`` holds any abstract. Fields absent from a given engine default
    to empty.
    """

    filename: str = ""
    size: str = ""
    mimetype: str = ""
    author: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class TorrentResult(SearchResult):
    """A torrent result from SearXNG's ``files`` category (``torrent.html``).

    Fields absent from a given engine default to empty / ``None``.
    """

    magnet_url: str = ""
    torrent_url: str = ""
    seed: int | None = None
    leech: int | None = None
    filesize: str = ""


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

# SearXNG is a metasearch aggregator: one query fans out to several upstream
# engines and the response returns only once they finish or hit SearXNG's own
# per-engine timeouts (``outgoing.request_timeout`` defaults to 3s, but heavy
# science engines like Crossref are configured up to ~30s). The client ceiling
# must therefore clear SearXNG's internal aggregation tail, not a single
# engine's latency: at 10s the multi-engine ``it``/``science`` tabs hit a
# premature client-side timeout mid-aggregation (observed live). 15s clears the
# common tail while still bounding an interactive turn.
_SEARXNG_TIMEOUT_SEC = 15.0

# SearXNG result categories (tabs) -- the full set from ``categories_as_tabs``
# in SearXNG's ``settings.yml``. Each maps to one or more result-template
# shapes; ``science`` yields ``paper.html`` (structured ``PaperResult``). The
# categories whose engines emit additional structured templates gain their own
# typed reader and overload over time; until then they return ``SearchResult``
# parsed from the common ``url``/``title``/``content`` fields. ``"social media"``
# carries a space -- that is SearXNG's exact wire value for the tab.
type SearxngCategory = Literal[
    "general",
    "images",
    "videos",
    "news",
    "map",
    "music",
    "it",
    "science",
    "files",
    "social media",
]

# Union of every result shape a SearXNG query can return -- the implementation
# return type behind the per-category overloads. ``VideoResult`` is omitted as
# a ``MediaResult`` subclass and the leaf subclasses cover the rest.
type SearxngResult = (
    PaperResult
    | ImageResult
    | MediaResult
    | MapResult
    | PackageResult
    | CodeResult
    | FileResult
    | TorrentResult
    | SearchResult
)


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["science"],
) -> Sequence[PaperResult]: ...


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["images"],
) -> Sequence[ImageResult]: ...


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["videos"],
) -> Sequence[VideoResult]: ...


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["news", "music"],
) -> Sequence[MediaResult]: ...


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["map"],
) -> Sequence[MapResult]: ...


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["it"],
) -> Sequence[PackageResult | CodeResult | SearchResult]: ...


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["files"],
) -> Sequence[FileResult | TorrentResult | SearchResult]: ...


@overload
def searxng(
    query: str,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: SearxngCategory = ...,
) -> Sequence[SearchResult]: ...


def searxng(
    query: str,
    num_results: int = 10,
    headers: dict[str, str] | None = None,
    *,
    categories: SearxngCategory = "general",
) -> Sequence[SearxngResult]:
    """Query a SearXNG instance and return parsed, typed JSON results.

    The return shape follows ``categories``: each tab whose engines emit a
    structured result template gets a richer :class:`SearchResult` subclass
    (``science`` -> :class:`PaperResult`, ``images`` -> :class:`ImageResult`,
    ``videos`` -> :class:`VideoResult`, ``news``/``music`` -> :class:`MediaResult`,
    ``map`` -> :class:`MapResult`, ``it`` -> :class:`PackageResult` /
    :class:`CodeResult`, ``files`` -> :class:`FileResult` /
    :class:`TorrentResult`). Categories with no extra structure return plain
    :class:`SearchResult`. Every subclass keeps ``url``/``title``/``snippet``
    populated, so a web-result consumer works on any return unchanged. The
    discriminating SearXNG ``template`` field is consumed here per result.

    Args:
      query: Search query string.
      num_results: Maximum results to return.
      headers: Optional override headers forwarded to fetch.
      categories: SearXNG result category (tab) to query.

    Returns:
      results: One typed record per hit -- a :class:`SearchResult` or a
        category-specific subclass of it.

    """
    base_url = _searxng_url()
    params = urlencode(
        {"q": query, "format": "json", "pageno": "1", "categories": categories}
    )
    body = fetch(
        f"{base_url}/search?{params}",
        headers=headers,
        timeout_sec=_SEARXNG_TIMEOUT_SEC,
    )
    payload = cast("object", json.loads(body))
    raw = (
        cast("dict[str, object]", payload).get("results")
        if isinstance(payload, dict)
        else None
    )
    items = cast("list[object]", raw) if isinstance(raw, list) else []
    parse = _SEARXNG_PARSERS.get(categories, _searxng_web)
    return [
        parse(cast("dict[str, object]", item))
        for item in items[:num_results]
        if isinstance(item, dict)
    ]


def _searxng_web(item: dict[str, object]) -> SearchResult:
    """Parse a SearXNG ``default.html`` item into a :class:`SearchResult`."""
    return SearchResult(
        url=str_val(item.get("url")),
        title=_clean_text(str_val(item.get("title"))),
        snippet=_clean_text(str_val(item.get("content"))),
    )


def _searxng_image(item: dict[str, object]) -> ImageResult:
    """Parse a SearXNG ``images.html`` item into an :class:`ImageResult`."""
    return ImageResult(
        url=str_val(item.get("url")),
        title=_clean_text(str_val(item.get("title"))),
        snippet=_clean_text(str_val(item.get("content"))),
        image_url=str_val(item.get("img_src")),
        thumbnail_url=str_val(item.get("thumbnail_src")),
        resolution=str_val(item.get("resolution")),
        img_format=str_val(item.get("img_format")),
        source=str_val(item.get("source")),
        filesize=str_val(item.get("filesize")),
    )


def _searxng_video(item: dict[str, object]) -> VideoResult:
    """Parse a SearXNG ``videos.html`` item into a :class:`VideoResult`."""
    return VideoResult(
        url=str_val(item.get("url")),
        title=_clean_text(str_val(item.get("title"))),
        snippet=_clean_text(str_val(item.get("content"))),
        published=datetime_val(item.get("publishedDate")),
        iframe_url=str_val(item.get("iframe_src")),
        length=str_val(item.get("length")),
        thumbnail_url=str_val(item.get("thumbnail")),
        views=str_val(item.get("views")),
        author=str_val(item.get("author")),
    )


def _searxng_media(item: dict[str, object]) -> MediaResult:
    """Parse a ``news``/``music`` ``default.html`` item into a :class:`MediaResult`."""
    return MediaResult(
        url=str_val(item.get("url")),
        title=_clean_text(str_val(item.get("title"))),
        snippet=_clean_text(str_val(item.get("content"))),
        published=datetime_val(item.get("publishedDate")),
        audio_url=str_val(item.get("audio_src")),
        iframe_url=str_val(item.get("iframe_src")),
        length=str_val(item.get("length")),
        thumbnail_url=str_val(item.get("thumbnail")),
    )


def _searxng_map(item: dict[str, object]) -> MapResult:
    """Parse a SearXNG ``map.html`` item into a :class:`MapResult`."""
    return MapResult(
        url=str_val(item.get("url")),
        title=_clean_text(str_val(item.get("title"))),
        snippet=_clean_text(str_val(item.get("content"))),
        latitude=float_val(item.get("latitude")) if "latitude" in item else None,
        longitude=float_val(item.get("longitude")) if "longitude" in item else None,
        address=str_map_val(item.get("address")),
    )


def _searxng_it(item: dict[str, object]) -> PackageResult | CodeResult | SearchResult:
    """Dispatch an ``it`` item by ``template`` to its package/code/web reader."""
    template = str_val(item.get("template"))
    if template == "packages.html":
        return _searxng_package(item)
    if template == "code.html":
        return _searxng_code(item)
    return _searxng_web(item)


def _searxng_package(item: dict[str, object]) -> PackageResult:
    """Parse a SearXNG ``packages.html`` item into a :class:`PackageResult`."""
    return PackageResult(
        url=str_val(item.get("url")),
        title=_clean_text(str_val(item.get("title"))),
        snippet=_clean_text(str_val(item.get("content"))),
        package_name=str_val(item.get("package_name")),
        version=str_val(item.get("version")),
        maintainer=str_val(item.get("maintainer")),
        license_name=str_val(item.get("license_name")),
        homepage=str_val(item.get("homepage")),
        source_code_url=str_val(item.get("source_code_url")),
        popularity=str_val(item.get("popularity")),
        tags=str_list_val(item.get("tags")),
    )


def _searxng_code(item: dict[str, object]) -> CodeResult:
    """Parse a SearXNG ``code.html`` item into a :class:`CodeResult`."""
    return CodeResult(
        url=str_val(item.get("url")),
        title=_clean_text(str_val(item.get("title"))),
        snippet=_clean_text(str_val(item.get("content"))),
        repository=str_val(item.get("repository")),
        filename=str_val(item.get("filename")),
        code_language=str_val(item.get("code_language")),
    )


def _searxng_files(
    item: dict[str, object],
) -> FileResult | TorrentResult | SearchResult:
    """Dispatch a ``files`` item by ``template`` to its file/torrent/web reader."""
    template = str_val(item.get("template"))
    if template == "torrent.html":
        return _searxng_torrent(item)
    if template == "file.html":
        return _searxng_file(item)
    return _searxng_web(item)


def _searxng_file(item: dict[str, object]) -> FileResult:
    """Parse a SearXNG ``file.html`` item into a :class:`FileResult`."""
    return FileResult(
        url=str_val(item.get("url")),
        title=_clean_text(str_val(item.get("title"))),
        snippet=_clean_text(
            str_val(item.get("abstract")) or str_val(item.get("content"))
        ),
        filename=str_val(item.get("filename")),
        size=str_val(item.get("size")),
        mimetype=str_val(item.get("mimetype")),
        author=str_val(item.get("author")),
    )


def _searxng_torrent(item: dict[str, object]) -> TorrentResult:
    """Parse a SearXNG ``torrent.html`` item into a :class:`TorrentResult`."""
    return TorrentResult(
        url=str_val(item.get("url")),
        title=_clean_text(str_val(item.get("title"))),
        snippet=_clean_text(str_val(item.get("content"))),
        magnet_url=str_val(item.get("magnetlink")),
        torrent_url=str_val(item.get("torrentfile")),
        seed=int_val(item.get("seed"), 0) if "seed" in item else None,
        leech=int_val(item.get("leech"), 0) if "leech" in item else None,
        filesize=str_val(item.get("filesize")),
    )


# Leading integer of SearXNG's humanized citation ``comments`` (e.g. "42
# citations from the year 2019 to 2024" -> 42). No engine reports the count as a
# structured integer, so this is the only recovery path; an unparseable comment
# yields ``None`` rather than a fabricated zero.
_CITATIONS_RE = re.compile(r"^\s*([\d,]+)")


def _searxng_paper(item: dict[str, object]) -> PaperResult:
    """Parse a SearXNG ``paper.html`` item into a :class:`PaperResult`."""
    cites = _CITATIONS_RE.match(str_val(item.get("comments")))
    return PaperResult(
        url=str_val(item.get("url")),
        title=_clean_text(str_val(item.get("title"))),
        snippet=_clean_text(str_val(item.get("content"))),
        authors=str_list_val(item.get("authors")),
        journal=_clean_text(str_val(item.get("journal"))),
        doi=str_val(item.get("doi")),
        pdf_url=str_val(item.get("pdf_url")),
        published=datetime_val(item.get("publishedDate")),
        tags=str_list_val(item.get("tags")),
        citations=int(cites.group(1).replace(",", "")) if cites else None,
    )


# Per-category result parser. A category absent here falls back to the generic
# web reader (``general`` and the structure-free ``social media`` tab). Each
# parser returns a ``SearchResult`` or a subclass of it, so the table's value
# type stays uniform while ``searxng``'s overloads narrow the element type.
_SEARXNG_PARSERS: Mapping[
    SearxngCategory, Callable[[dict[str, object]], SearxngResult]
] = {
    "science": _searxng_paper,
    "images": _searxng_image,
    "videos": _searxng_video,
    "news": _searxng_media,
    "music": _searxng_media,
    "map": _searxng_map,
    "it": _searxng_it,
    "files": _searxng_files,
}


def _searxng_url() -> str:
    """Return the configured SearXNG base URL without a trailing slash."""
    url = os.environ.get(_SEARXNG_URL_ENV, "").rstrip("/")
    if not url:
        raise SearchError(
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


@overload
def search(
    query: str,
    backend: Literal["searxng"],
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["science"],
) -> Sequence[PaperResult]: ...


@overload
def search(
    query: str,
    backend: Literal["searxng"],
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["images"],
) -> Sequence[ImageResult]: ...


@overload
def search(
    query: str,
    backend: Literal["searxng"],
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["videos"],
) -> Sequence[VideoResult]: ...


@overload
def search(
    query: str,
    backend: Literal["searxng"],
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["news", "music"],
) -> Sequence[MediaResult]: ...


@overload
def search(
    query: str,
    backend: Literal["searxng"],
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["map"],
) -> Sequence[MapResult]: ...


@overload
def search(
    query: str,
    backend: Literal["searxng"],
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["it"],
) -> Sequence[PackageResult | CodeResult | SearchResult]: ...


@overload
def search(
    query: str,
    backend: Literal["searxng"],
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: Literal["files"],
) -> Sequence[FileResult | TorrentResult | SearchResult]: ...


@overload
def search(
    query: str,
    backend: SearchBackends | None = ...,
    num_results: int = ...,
    headers: dict[str, str] | None = ...,
    *,
    categories: SearxngCategory = ...,
) -> Sequence[SearchResult]: ...


def search(
    query: str,
    backend: SearchBackends | None = None,
    num_results: int = 10,
    headers: dict[str, str] | None = None,
    *,
    categories: SearxngCategory = "general",
) -> Sequence[SearxngResult]:
    """Dispatch to the named search backend.

    Args:
      query: Search query string.
      backend: Backend name. Defaults to ``DEFAULT_SEARCH_BACKEND``.
      num_results: Maximum results to return.
      headers: Optional override headers forwarded to the backend.
      categories: SearXNG result category; only the ``"searxng"`` backend
        honors a non-default value (the HTML-scraping backends serve general
        web results only). Defaults to ``"general"``.

    Returns:
      results: One typed record per hit. SearXNG categories with extra
        structure yield a category-specific :class:`SearchResult` subclass;
        all other paths yield plain :class:`SearchResult`.

    """
    if backend is None:
        backend = DEFAULT_SEARCH_BACKEND
    if categories != "general" and backend != "searxng":
        raise ValueError(
            f"'categories' is only supported by the 'searxng' backend, not {backend!r}."
        )
    try:
        if backend == "searxng":
            return searxng(query, num_results, headers, categories=categories)
        if backend == "duckduckgo":
            return duckduckgo(query, num_results, headers)

    except (
        FetchError,
        OSError,
        TimeoutError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ) as e:
        raise SearchError(f"{backend} search failed: {e}") from e
    raise ValueError(f"Unknown backend: {backend!r}")  # pyright: ignore[reportUnreachable] -- reachable at runtime


def _refresh_gsa_useragents() -> None:
    """Regenerate the User-Agent pool from upstream source data.

    Downloads the upstream user-agent dataset (a gzipped JSON array of
    ``{"userAgent": ...}`` records) and rewrites ``gsa_useragents.txt``.

    Run via ``python -m sagent.lib.web.search`` (scheduled in CI).
    """
    url = (
        "https://raw.githubusercontent.com/intoli/user-agents/"
        "main/src/user-agents.json.gz"
    )
    raw = gzip.decompress(fetch(url, timeout_sec=30))
    parsed: object = json.loads(raw)
    if not isinstance(parsed, list):
        # TRY004: upstream-shape failure, not a caller type error; matches the
        # RuntimeError raised below for the same class of contract violation.
        msg = f"expected JSON array from {url}; upstream shape changed?"
        raise RuntimeError(msg)  # noqa: TRY004
    records = cast("list[object]", parsed)
    selected: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        ua = cast("dict[str, object]", record).get("userAgent")
        if not isinstance(ua, str):
            continue
        if (
            "Android" in ua
            and "Chrome" in ua
            and "Samsung" not in ua
            and "Android 10; K" not in ua
        ):
            selected.add(ua)
    if not selected:
        raise RuntimeError(
            f"refresh produced 0 user agents from {url}; upstream shape changed?",
        )
    text = "\n".join(sorted(selected)) + "\n"
    _GSA_USERAGENTS_PATH.write_text(text)
    _get_gsa_useragents.cache_clear()
    logger.info("wrote %d user agents to %s", len(selected), _GSA_USERAGENTS_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _refresh_gsa_useragents()
