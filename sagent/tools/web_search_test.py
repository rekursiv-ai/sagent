"""Tests for WebSearch: domain scoping, results, and errors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import itertools
import re

from wesearch.fetch import FetchSession, PolicyParams
from wesearch.search.types import PaperResult, SearchResult
from wesearch.types.errors import BotDetectionError, CooldownError

import pytest

from sagent.session.types import (
    ToolResult,
    WebSearchHit,
    WebSearchResult,
)
from sagent.tools import web_search
from sagent.tools.types import ToolContext, settled
from sagent.tools.web_search import WebSearch


if TYPE_CHECKING:
    from treekle import PlainTree


@dataclass(slots=True, kw_only=True)
class Searches:
    """Stands in for the search backends: records each query, answers or raises."""

    hits: Sequence[SearchResult] = ()
    error: Exception | None = None
    seen: list[dict[str, object]] = field(default_factory=list)

    def __call__(
        self,
        query: str,
        *,
        backend: str | None,
        categories: str,
        transport: str,
        session: FetchSession,
    ) -> Sequence[SearchResult]:
        self.seen.append(
            {
                "query": query,
                "backend": backend,
                "categories": categories,
                "transport": transport,
                "session": session,
            },
        )
        if self.error is not None:
            raise self.error
        return self.hits


@pytest.fixture
def searches(monkeypatch: pytest.MonkeyPatch) -> Searches:
    fake = Searches(
        hits=(
            SearchResult(url="https://a.example", title="A", snippet="about a"),
            SearchResult(url="https://b.example", title="B", snippet=""),
        ),
    )
    monkeypatch.setattr(web_search, "search", fake)
    # Each search takes 2.5 seconds on this clock.
    clock = SimpleNamespace(monotonic=itertools.count(10.0, 2.5).__next__)
    monkeypatch.setattr(web_search, "time", clock)
    return fake


_CONTEXT = ToolContext(cwd=Path("/"), fetch_session=FetchSession())


def test_websearch_describes_itself() -> None:
    tool = WebSearch()
    assert tool.name == "WebSearch"
    assert tool.description.strip()
    assert tool.schema["required"] == ["query"]
    assert tool.summary({"query": "python 3.14"}) == "WebSearch 'python 3.14'"
    assert tool.summary({}) == "WebSearch ''"
    assert tool.serialize_key(_CONTEXT, {"query": "x"}) is None


@pytest.mark.asyncio
async def test_a_search_returns_each_hit(searches: Searches) -> None:
    tool = WebSearch()
    result = await settled(tool.call(_CONTEXT, {"query": "python"}))
    assert result == ToolResult(
        call_id="",
        state="done",
        stdout=(
            "A\nhttps://a.example\nabout a\n\nB\nhttps://b.example\n",
            WebSearchHit(url="https://a.example", title="A", snippet="about a"),
            WebSearchHit(url="https://b.example", title="B", snippet=""),
        ),
        result=WebSearchResult(query="python", duration_sec=2.5),
    )
    assert searches.seen == [
        {
            "query": "python",
            "backend": None,
            "categories": "general",
            "transport": "auto",
            "session": _CONTEXT.fetch_session,
        },
    ]


@pytest.mark.asyncio
async def test_a_hits_snippet_carries_its_category_fields(searches: Searches) -> None:
    searches.hits = (
        PaperResult(
            url="https://p.example",
            title="P",
            snippet="abstract",
            doi="10.1/x",
        ),
    )
    result = await settled(
        WebSearch().call(
            _CONTEXT,
            {"query": "q", "categories": "science"},
        ),
    )
    assert result.text().startswith("P\nhttps://p.example\nabstract\n")
    assert "doi:10.1/x" in result.text()
    assert searches.seen[0]["categories"] == "science"


@pytest.mark.asyncio
async def test_a_call_may_choose_backend_and_transport(searches: Searches) -> None:
    tool = WebSearch(policy=PolicyParams(transport="curl"))
    _ = await settled(tool.call(_CONTEXT, {"query": "q"}))
    _ = await settled(
        tool.call(
            _CONTEXT,
            {"query": "q", "backend": "searxng", "transport": "stdlib"},
        ),
    )
    assert [(s["backend"], s["transport"]) for s in searches.seen] == [
        (None, "curl"),
        ("searxng", "stdlib"),
    ]


@pytest.mark.asyncio
async def test_domains_scope_the_query(searches: Searches) -> None:
    _ = await settled(
        WebSearch().call(
            _CONTEXT,
            {
                "query": "release notes",
                "allowed_domains": [" Docs.Python.org ", "*.python.org:443"],
                "blocked_domains": ["spam.example"],
            },
        ),
    )
    assert searches.seen[0]["query"] == (
        "release notes site:Docs.Python.org site:*.python.org:443 -site:spam.example"
    )


@pytest.mark.parametrize(
    "domain",
    ["x.com -site:trusted.com", "localhost", "a b.com", "", "http://x.com"],
)
@pytest.mark.asyncio
async def test_a_domain_that_is_not_a_hostname_is_refused(
    searches: Searches,
    domain: str,
) -> None:
    for key in ("allowed_domains", "blocked_domains"):
        result = await settled(
            WebSearch().call(
                _CONTEXT,
                {"query": "q", key: [domain]},
            ),
        )
        assert result == ToolResult(
            call_id="",
            state="done",
            stderr=(f"Not a hostname: {domain!r}.",),
            result=WebSearchResult(exit_code=1),
        )
    assert searches.seen == []


@pytest.mark.asyncio
async def test_no_hits_says_so(searches: Searches) -> None:
    searches.hits = ()
    result = await settled(WebSearch().call(_CONTEXT, {"query": "q"}))
    assert result == ToolResult(
        call_id="",
        state="done",
        stdout=("(no results)",),
        result=WebSearchResult(query="q"),
    )


@pytest.mark.asyncio
async def test_hits_are_kept_while_they_fit(searches: Searches) -> None:
    searches.hits = tuple(
        SearchResult(url=f"https://{i}.example", title=f"T{i}", snippet="s" * 50)
        for i in range(10)
    )
    context = ToolContext(cwd=Path("/"), max_result_chars=200)
    result = await settled(WebSearch().call(context, {"query": "q"}))
    assert re.findall(r"^T\d+$", result.text(), flags=re.MULTILINE) == ["T0", "T1"]
    # Each hit written is one the text names.
    hits = [item for item in result.stdout if isinstance(item, WebSearchHit)]
    assert [hit.title for hit in hits] == ["T0", "T1"]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (BotDetectionError(), BotDetectionError.guidance),
        (
            CooldownError("Cooling down for 40 minutes."),
            f"Cooling down for 40 minutes. {CooldownError.guidance}",
        ),
        (RuntimeError("no backend answered"), "no backend answered"),
        (ValueError("bad category"), "bad category"),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_search_says_why(
    searches: Searches,
    error: Exception,
    message: str,
) -> None:
    searches.error = error
    result = await settled(WebSearch().call(_CONTEXT, {"query": "q"}))
    assert result == ToolResult(
        call_id="",
        state="done",
        stderr=(message,),
        result=WebSearchResult(exit_code=1),
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": 3},
        {"query": "q", "backend": "altavista"},
        {"query": "q", "categories": "x"},
    ],
)
@pytest.mark.asyncio
async def test_malformed_arguments_are_argument_errors(
    searches: Searches,
    arguments: dict[str, PlainTree],
) -> None:
    with pytest.raises(ValueError, match="must be"):
        _ = await WebSearch().call(_CONTEXT, arguments).result
    assert searches.seen == []


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
