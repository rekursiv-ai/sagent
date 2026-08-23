"""WebSearch tool: query a pluggable search backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, cast

import asyncio
import re

from wesearch.fetch import Transport
from wesearch.search.custom_types import (
    SearchBackends,
    SearchError,
    SearxngCategory,
)
from wesearch.search.render import format_result
from wesearch.search.search import SearchParamsSchema, search
from wesearch.types.errors import BotDetectionError

from sagent.lib.custom_json import json_freeze
from sagent.tools.core import (
    TOOL_RESULT_MAX_CHARS,
    load_tool_description,
    truncate,
)
from sagent.tools.display import Toggle, Wrap
from sagent.tools.tool_spec import CLI_SETTABLE
from sagent.types.runtime import ToolResult


@dataclass(frozen=True, slots=True, kw_only=True)
class WebSearch:
    """Search the web using a configurable backend."""

    name = "WebSearch"
    tool_id = "application/x-tool-websearch"
    clearable_results = True

    @property
    def description(self) -> str:
        """Return the tool description, re-evaluating ``{{NOW}}`` each access.

        A long-running process that spans a month boundary must not freeze
        the substitution at import time.
        """
        return load_tool_description("WebSearch")

    # Domain filters stay local: they are a sagent query-building convenience
    # (spliced into the query string), not a wesearch search parameter.
    directive_schema = json_freeze(
        {
            **SearchParamsSchema.json_schema(),
            "properties": {
                **cast(
                    dict[str, object], SearchParamsSchema.json_schema()["properties"]
                ),
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
            },
        }
    )

    output: Annotated[Toggle, CLI_SETTABLE] = "off"
    """Whether the result body renders in the pane."""

    output_head_rows: Annotated[int, CLI_SETTABLE] = 2
    """Leading body rows kept."""

    output_tail_rows: Annotated[int, CLI_SETTABLE] = 2
    """Trailing body rows kept, after a ``⋯ N lines ⋯`` marker."""

    output_max_width: Annotated[int, CLI_SETTABLE] = 0
    """Cell width cap; ``0`` uses the pane width."""

    output_wrap: Annotated[Wrap, CLI_SETTABLE] = "wrap"
    """``wrap`` continues an over-wide line, ``chop`` marks the cut."""

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short display label for this invocation.

        Args:
          args: Directive carrying the ``query`` string.

        Returns:
          label: ``WebSearch '<query>'`` line shown before invocation.

        """
        return f"WebSearch {str(args.get('query', ''))!r}"

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
        try:
            # One call replaces a per-parameter ladder that restated every
            # name, type, and default the schema above already declares.
            params = SearchParamsSchema.coerce(args)
        except ValueError as e:
            return ToolResult(call_id="", content=str(e), is_error=True)
        query = str(params["query"])
        # ``backend`` stays None when unnamed: ``search`` resolves it, and a
        # non-general category resolves it to SearXNG. Substituting a constant
        # here would hide that choice and reinstate the per-adapter override
        # this tool used to carry.
        backend = cast("SearchBackends | None", params["backend"])
        transport = cast(Transport, params["transport"])
        categories = cast(SearxngCategory, params["categories"])
        try:
            q = _build_query(
                query,
                args.get("allowed_domains"),
                args.get("blocked_domains"),
            )
        except (TypeError, ValueError) as e:
            return ToolResult(call_id="", content=str(e), is_error=True)
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
            text = "\n\n".join(format_result(r) for r in results)
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

    Only tokens matching a bare-hostname shape are accepted; a non-hostname
    value (containing whitespace or query operators) is REJECTED rather than
    spliced in, so a domain filter cannot inject extra query syntax.

    Raises:
      TypeError: When a filter argument is not a list.
      ValueError: When a list member is not a hostname. Dropping
        the bad value instead would run an UNRESTRICTED search while the caller
        believed it was scoped -- failing open on the one argument whose whole
        purpose is to restrict.

    """
    return (
        query
        + _site_filters(allowed_domains, name="allowed_domains", prefix="site:")
        + _site_filters(blocked_domains, name="blocked_domains", prefix="-site:")
    )


def _site_filters(domains: object, *, name: str, prefix: str) -> str:
    """Render one domain list as ``site:``/``-site:`` terms, rejecting bad input."""
    if domains is None:
        return ""
    if not isinstance(domains, (list, tuple)):
        raise TypeError(
            f"{name!r} must be a list of hostnames, got {type(domains).__name__}."
        )
    terms = ""
    # Iterated as `object`, not cast to a value type: the cast asserted a
    # member type the very next line has to check anyway, and left the sequence
    # itself partially unknown.
    for domain in cast(Sequence[object], domains):
        if not isinstance(domain, str) or not _HOSTNAME_RE.match(domain.strip()):
            raise ValueError(f"{name!r} contains a non-hostname value: {domain!r}.")
        terms += f" {prefix}{domain.strip()}"
    return terms
