"""Tests for PaperSearch: its arguments, rendering, cache, and errors.

``wesearch`` owns every backend; the search it makes is replaced where
the tool imports it, so nothing reaches the network.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import TYPE_CHECKING

from wesearch.fetch import FetchSession
from wesearch.paper.errors import PaperError
from wesearch.paper.render import format_record
from wesearch.paper.search import SearchResult
from wesearch.paper.types import PaperRecord
from wesearch.types.errors import PuzzleChallengeError

import pytest

from sagent.tools import paper_search
from sagent.tools.paper_search import PaperSearch, _named
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from pathlib import Path

    from treekle import PlainTree


_ONE = PaperRecord(title="Attention", year=2017, abstract="All you need, really.")
_TWO = PaperRecord(title="Diffusion", year=2020)


def test_a_search_needs_the_session_it_shares() -> None:
    """A session a tool made for itself is one nobody closes."""
    (session,) = [f for f in fields(PaperSearch) if f.name == "session"]
    assert (session.default, session.default_factory) == (MISSING, MISSING)


def test_summary_quotes_the_query_and_names_a_single_source() -> None:
    tool = _tool()
    assert tool.summary({"query": " diffusion "}) == "PaperSearch 'diffusion'"
    assert tool.summary({"query": "x", "source": "fused"}) == "PaperSearch 'x'"
    assert tool.summary({"query": "x", "source": None}) == "PaperSearch 'x'"
    assert (
        tool.summary({"query": "x", "source": "openalex"})
        == "PaperSearch 'x' (openalex)"
    )
    assert tool.summary({}) == "PaperSearch"


def test_searches_run_beside_anything(tmp_path: Path) -> None:
    assert _tool().serialize_key(ToolContext(cwd=tmp_path), {"query": "x"}) is None


@pytest.mark.asyncio
async def test_the_search_gets_every_filter_and_the_shared_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _searches(monkeypatch, SearchResult(records=[_ONE], total=1, complete=True))
    tool = _tool()
    await tool.call(
        ToolContext(cwd=tmp_path),
        {
            "query": " attention ",
            "source": "s2",
            "limit": 5,
            "year_from": 2015,
            "year_to": 2020,
            "open_access_only": True,
        },
    )
    assert calls == [
        _Call(
            query="attention",
            source="s2",
            limit=5,
            year_from=2015,
            year_to=2020,
            open_access_only=True,
            session=tool.session,
        ),
    ]


@pytest.mark.asyncio
async def test_hits_come_back_one_record_each_with_what_was_held_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _searches(monkeypatch, SearchResult(records=[_ONE, _TWO], total=40, complete=True))
    result = await _tool().call(ToolContext(cwd=tmp_path), {"query": "x"})
    assert result.text() == (
        f"{format_record(_ONE)}\n{format_record(_TWO)}"
        "\n... (showing 2 of 40; tighten filters for more)"
    )
    assert result.is_error is None
    assert result.call_id == ""


@pytest.mark.asyncio
async def test_a_limit_and_an_abstract_cap_shape_the_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _searches(monkeypatch, SearchResult(records=[_ONE, _TWO], total=2, complete=True))
    result = await _tool().call(
        ToolContext(cwd=tmp_path),
        {"query": "x", "limit": 1, "abstract_chars": 5},
    )
    assert result.text() == (
        format_record(_ONE, abstract_chars=5)
        + "\n... (showing 1 of 2; tighten filters for more)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "hint"),
    [
        ("diffusion", ""),
        (
            "diffusion zebra",
            (
                "\nNote: every query term must appear in a paper's title/abstract (terms"
                " are AND-ed). A zero result usually means one specific term has no match"
                " -- drop the most specific term and retry to broaden."
            ),
        ),
    ],
)
async def test_nothing_found_says_why_a_search_may_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    hint: str,
) -> None:
    _searches(monkeypatch, SearchResult(records=[], total=0, complete=True))
    result = await _tool().call(ToolContext(cwd=tmp_path), {"query": query})
    assert result.text() == (
        "(no results)"
        + hint
        + "\nNote: paper search matches title/abstract text, not author names (true"
        " for every source). If your query was an author name, use the PaperAuthor"
        " tool instead."
    )


@pytest.mark.asyncio
async def test_a_complete_result_is_kept_and_a_partial_one_is_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _tool()
    context = ToolContext(cwd=tmp_path)
    calls = _searches(
        monkeypatch,
        SearchResult(records=[_ONE], total=1, complete=False),
        SearchResult(records=[_TWO], total=1, complete=True),
        SearchResult(records=[_ONE], total=1, complete=True),
    )
    partial = await tool.call(context, {"query": "x"})
    complete = await tool.call(context, {"query": "x"})
    again = await tool.call(context, {"query": " x "})
    other = await tool.call(context, {"query": "x", "limit": 1})
    assert partial.text() == format_record(_ONE)
    assert complete.text() == again.text() == format_record(_TWO)
    assert len(calls) == 3
    assert other.text() == format_record(_ONE)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (PaperError("S2 is down"), "S2 is down"),
        (
            PuzzleChallengeError("blocked"),
            (
                "Semantic Scholar or OpenAlex returned a CAPTCHA: blocked. The scraping IP is"
                " rate-limited; retry later, change egress IP, or use another source."
            ),
        ),
    ],
)
async def test_a_failed_search_is_an_error_and_is_not_kept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    message: str,
) -> None:
    tool = _tool()
    calls = _searches(
        monkeypatch,
        failure,
        SearchResult(records=[_ONE], total=1, complete=True),
    )
    failed = await tool.call(ToolContext(cwd=tmp_path), {"query": "x"})
    assert failed.is_error is True
    assert failed.text() == message
    retried = await tool.call(ToolContext(cwd=tmp_path), {"query": "x"})
    assert retried.text() == format_record(_ONE)
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "named"),
    [
        ("s2", "Semantic Scholar"),
        ("openalex", "OpenAlex"),
        ("searxng", "SearXNG"),
        ("fused", "Semantic Scholar or OpenAlex"),
    ],
)
async def test_a_captcha_names_the_source_as_a_user_knows_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    named: str,
) -> None:
    _searches(monkeypatch, PuzzleChallengeError("blocked"))
    failed = await _tool().call(
        ToolContext(cwd=tmp_path),
        {"query": "x", "source": source},
    )
    assert failed.text().startswith(f"{named} returned a CAPTCHA: blocked.")


def test_a_source_with_no_name_of_its_own_is_named_by_its_words() -> None:
    assert _named("some_new_source") == "Some New Source"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"query": "  "}, "'query' is required."),
        (
            {"query": "x", "year_from": 2021, "year_to": 2020},
            r"'year_from' \(2021\) must not exceed 'year_to' \(2020\)\.",
        ),
        ({"query": "x", "limit": 0}, "'limit' must be at least 1"),
        ({"query": "x", "abstract_chars": 0}, "'abstract_chars' must be at least 1"),
        ({"query": "x", "source": "bing"}, "'source' must be one of"),
        ({}, "Missing argument 'query'"),
    ],
)
async def test_arguments_that_do_not_fit_are_refused_before_any_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, PlainTree],
    message: str,
) -> None:
    calls = _searches(monkeypatch)
    with pytest.raises(ValueError, match=message):
        await _tool().call(ToolContext(cwd=tmp_path), arguments)
    assert calls == []


@pytest.mark.asyncio
async def test_equal_years_bound_one_year(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _searches(monkeypatch, SearchResult(records=[], total=0, complete=True))
    await _tool().call(
        ToolContext(cwd=tmp_path),
        {"query": "x", "year_from": 2020, "year_to": 2020},
    )
    assert [(c.year_from, c.year_to) for c in calls] == [(2020, 2020)]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Call:
    """One search the tool asked for."""

    query: str
    source: str
    limit: int | None
    year_from: int | None
    year_to: int | None
    open_access_only: bool
    session: FetchSession


def _tool() -> PaperSearch:
    return PaperSearch(session=FetchSession())


def _searches(
    monkeypatch: pytest.MonkeyPatch,
    *answers: SearchResult | Exception,
) -> list[_Call]:
    """Answer each search with the next of ``answers``; return what was asked."""
    calls: list[_Call] = []
    pending = list(answers)

    def search(
        query: str,
        *,
        source: str,
        limit: int | None,
        year_from: int | None,
        year_to: int | None,
        open_access_only: bool,
        session: FetchSession,
    ) -> SearchResult:
        calls.append(
            _Call(
                query=query,
                source=source,
                limit=limit,
                year_from=year_from,
                year_to=year_to,
                open_access_only=open_access_only,
                session=session,
            ),
        )
        answer = pending.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(paper_search, "search", search)
    return calls


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
