"""WebSearch tool: query a pluggable search backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast, get_args

import asyncio
import re

from wesearch.errors import BotDetectionError
from wesearch.fetch import Transport
from wesearch.search import (
    DEFAULT_SEARCH_BACKEND,
    CodeResult,
    FileResult,
    ImageResult,
    MapResult,
    MediaResult,
    PackageResult,
    PaperResult,
    SearchBackends,
    SearchError,
    SearchResult,
    SearxngCategory,
    TorrentResult,
    VideoResult,
    search,
)

from sagent.lib.custom_json import JSON, JSONValue, json_freeze
from sagent.tools.core import (
    TOOL_RESULT_MAX_CHARS,
    load_tool_description,
    truncate,
)
from sagent.types.runtime import ToolResult


class WebSearch:
    """Search the web using a configurable backend."""

    name: str = "WebSearch"
    tool_id: str = "application/x-tool-websearch"
    clearable_results: bool = True

    @property
    def description(self) -> str:
        """Return the tool description, re-evaluating ``{{NOW}}`` each access.

        A long-running process that spans a month boundary must not freeze
        the substitution at import time.
        """
        return load_tool_description("WebSearch")

    directive_schema: JSON = json_freeze(
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string."},
                "allowed_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only include results from these domains.",
                },
                "blocked_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exclude results from these domains.",
                },
                "backend": {
                    "type": "string",
                    "enum": get_args(SearchBackends),
                    "description": (
                        f'Search backend (default: "{DEFAULT_SEARCH_BACKEND}").'
                    ),
                },
                "transport": {
                    "type": "string",
                    "enum": get_args(Transport),
                    "description": (
                        "Retrieval path. 'auto' tries curl and escalates to "
                        "Zendriver when a site bot-blocks it. Set an explicit "
                        "transport to stress a path."
                    ),
                },
                "categories": {
                    "type": "string",
                    "enum": list(get_args(SearxngCategory.__value__)),
                    "description": (
                        "SearXNG result category (tab). Non-default values force "
                        "the SearXNG backend. 'science' returns papers, 'images' "
                        "image results, 'videos' video metadata, 'news'/'music' "
                        "dated/media results, 'map' places with coordinates, 'it' "
                        "packages/code, 'files' files/torrents. Omit for general "
                        "web results."
                    ),
                },
            },
            "required": ["query"],
        }
    )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short display label for this invocation.

        Args:
          args: Directive carrying the ``query`` string.

        Returns:
          label: ``WebSearch '<query>'`` line shown before invocation.

        """
        query = str(args.get("query", ""))
        if len(query) > 50:
            query = query[:47] + "..."
        return f"WebSearch {query!r}"

    def summary_result(self, result: ToolResult) -> str | None:
        """Suppress the per-call receipt for WebSearch.

        Args:
          result: Completed ``ToolResult`` (ignored).

        Returns:
          receipt: Always ``None`` (no receipt line).

        """
        del result
        return None

    def prompt(self) -> str:
        """Return no supplemental system-prompt text for WebSearch.

        Returns:
          contribution: Empty string.

        """
        return ""

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: independent network fetch, no serialization."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Execute the web search and return formatted results.

        Args:
          args: Directive with ``query``, optional ``allowed_domains`` /
              ``blocked_domains`` / ``backend`` / ``categories``.

        Returns:
          result: Top results rendered per category -- a markdown link plus a
              snippet, enriched with the structured fields a SearXNG category
              carries -- or an error when the backend rejects the query.

        """
        query = str(args.get("query", ""))
        backend: SearchBackends = DEFAULT_SEARCH_BACKEND
        backend_val = args.get("backend")
        if isinstance(backend_val, str) and backend_val in get_args(SearchBackends):
            backend = cast(SearchBackends, backend_val)
        elif backend_val is not None:
            return ToolResult(
                call_id="",
                content=(
                    f"Invalid backend {backend_val!r}."
                    f" Valid: {', '.join(get_args(SearchBackends))}."
                ),
                is_error=True,
            )
        transport: Transport = "auto"
        transport_val = args.get("transport")
        if isinstance(transport_val, str) and transport_val in get_args(Transport):
            transport = cast(Transport, transport_val)
        elif transport_val is not None:
            return ToolResult(
                call_id="",
                content=(
                    f"Invalid transport {transport_val!r}."
                    f" Valid: {', '.join(get_args(Transport))}."
                ),
                is_error=True,
            )
        valid_categories = get_args(SearxngCategory.__value__)
        categories: SearxngCategory = "general"
        categories_val = args.get("categories")
        if isinstance(categories_val, str) and categories_val in valid_categories:
            categories = cast(SearxngCategory, categories_val)
        elif categories_val is not None:
            return ToolResult(
                call_id="",
                content=(
                    f"Invalid category {categories_val!r}."
                    f" Valid: {', '.join(valid_categories)}."
                ),
                is_error=True,
            )
        # A non-general category requires SearXNG; force it rather than erroring
        # when the caller left the backend at its default.
        if categories != "general":
            backend = "searxng"
        q = _build_query(
            query,
            args.get("allowed_domains"),
            args.get("blocked_domains"),
        )
        try:
            if transport == "auto":
                results = await asyncio.to_thread(
                    search,
                    q,
                    backend=backend,
                    categories=categories,
                )
            else:
                results = await asyncio.to_thread(
                    search,
                    q,
                    backend=backend,
                    categories=categories,
                    transport=transport,
                )
        except BotDetectionError as err:
            # Surface the class guidance (which captcha / IP-rotation remedy)
            # AND the per-instance reason when it carries extra detail (e.g. the
            # scholar cooldown's "~Nh on this IP"), so that actionable specifics
            # are not discarded. Mirrors web_fetch's specific rendering.
            reason = str(err)
            content = (
                f"{reason} {err.guidance}" if reason != err.guidance else err.guidance
            )
            return ToolResult(call_id="", content=content, is_error=True)
        except (RuntimeError, SearchError, ValueError) as err:
            return ToolResult(call_id="", content=str(err), is_error=True)
        if not results:
            text = "(no results)"
        else:
            text = "\n\n".join(_format_result(r) for r in results[:10])
        return ToolResult(call_id="", content=truncate(text, TOOL_RESULT_MAX_CHARS))


# A bare hostname: dot-separated labels of letters/digits/hyphens, optional
# leading wildcard and trailing port. Anything else (whitespace, a ``site:`` or
# ``-site:`` operator, a query fragment) is NOT a hostname and must not be
# spliced into the query string, or a caller could inject/contradict the scope
# (e.g. ``"x.com -site:trusted.com"`` would un-scope the search).
_HOSTNAME_RE = re.compile(r"^(?:\*\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?::\d+)?$")


def _build_query(
    query: str,
    allowed_domains: object,
    blocked_domains: object,
) -> str:
    """Return *query* with ``site:`` / ``-site:`` filters for valid domains.

    Only tokens matching a bare-hostname shape are appended; a non-hostname
    value (containing whitespace or query operators) is silently dropped rather
    than spliced in, so a domain filter cannot inject extra query syntax.
    """
    allowed = (
        list(cast(list[JSONValue], allowed_domains))
        if isinstance(allowed_domains, (list, tuple))
        else []
    )
    blocked = (
        list(cast(list[JSONValue], blocked_domains))
        if isinstance(blocked_domains, (list, tuple))
        else []
    )
    for domain in allowed:
        if isinstance(domain, str) and _HOSTNAME_RE.match(domain.strip()):
            query += f" site:{domain.strip()}"
    for domain in blocked:
        if isinstance(domain, str) and _HOSTNAME_RE.match(domain.strip()):
            query += f" -site:{domain.strip()}"
    return query


def _format_result(r: SearchResult) -> str:
    """Render one result as a markdown link plus its structured fields.

    Dispatches on the concrete :class:`SearchResult` subclass so a category's
    extra fields (a paper's authors/DOI, an image's source URL, a place's
    coordinates) reach the agent instead of being flattened to title/snippet.
    The base ``[title](url)`` line and snippet are always emitted; subclass
    fields follow on an indented detail line when present.
    """
    head = f"[{r.title}]({r.url})"
    detail = _result_detail(r)
    body = "\n".join(part for part in (r.snippet, detail) if part)
    return f"{head}\n{body}" if body else head


def _result_detail(r: SearchResult) -> str:
    """Return the category-specific detail line for a result, or empty."""
    if isinstance(r, PaperResult):
        parts = [
            ", ".join(r.authors[:3]) + (" +" if len(r.authors) > 3 else ""),
            r.journal,
            str(r.published.year) if r.published else "",
            f"doi:{r.doi}" if r.doi else "",
            f"cites:{r.citations}" if r.citations is not None else "",
            r.pdf_url,
        ]
    elif isinstance(r, ImageResult):
        parts = [r.image_url, r.resolution, r.img_format, r.source]
    elif isinstance(r, VideoResult):
        parts = [
            r.author,
            r.length,
            f"{r.views} views" if r.views else "",
            r.iframe_url,
        ]
    elif isinstance(r, MediaResult):
        parts = [
            str(r.published.date()) if r.published else "",
            r.length,
            r.audio_url or r.iframe_url,
        ]
    elif isinstance(r, MapResult):
        coords = f"{r.latitude},{r.longitude}" if r.latitude is not None else ""
        parts = [coords, ", ".join(r.address.values())]
    elif isinstance(r, PackageResult):
        parts = [
            r.package_name,
            r.version,
            r.license_name,
            r.homepage or r.source_code_url,
        ]
    elif isinstance(r, CodeResult):
        parts = [r.repository, r.filename, r.code_language]
    elif isinstance(r, FileResult):
        parts = [r.filename, r.size, r.mimetype]
    elif isinstance(r, TorrentResult):
        parts = [
            r.filesize,
            f"seed:{r.seed}" if r.seed is not None else "",
            f"leech:{r.leech}" if r.leech is not None else "",
            r.magnet_url,
        ]
    else:
        return ""
    kept = [p for p in parts if p]
    return "  " + " · ".join(kept) if kept else ""
