"""Tests for PaperAuthor: author search, author metadata, and an author's papers.

``wesearch`` owns every lookup; each is replaced where the tool imports
it, so nothing reaches the network.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import TYPE_CHECKING

from wesearch.fetch import FetchSession
from wesearch.paper.authors import AuthorSearchResult
from wesearch.paper.details import Listing
from wesearch.paper.errors import PaperError
from wesearch.paper.render import (
    format_author_block,
    format_author_line,
    format_record,
)
from wesearch.paper.types import AuthorRecord, PaperRecord

import pytest

from sagent.tools import paper_author
from sagent.tools.paper_author import PaperAuthor
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from pathlib import Path

    from treekle import PlainTree


_ADA = AuthorRecord(author_id="1741101", name="Ada", h_index=40, paper_count=120)
_BOB = AuthorRecord(author_id="2262347", name="Bob")
_PAPER = PaperRecord(title="Attention", year=2017, abstract="All you need, really.")


def test_a_lookup_needs_the_session_it_shares() -> None:
    (session,) = [f for f in fields(PaperAuthor) if f.name == "session"]
    assert (session.default, session.default_factory) == (MISSING, MISSING)


@pytest.mark.parametrize(
    ("arguments", "label"),
    [
        ({"query": " Yoshua Bengio "}, "PaperAuthor search 'Yoshua Bengio'"),
        ({"query": "x", "ids": "1"}, "PaperAuthor search 'x'"),
        ({"ids": "1741101", "operation": "papers"}, "PaperAuthor papers 1741101"),
        ({"ids": ["1741101", "2262347"]}, "PaperAuthor 1741101 (+1 more)"),
        ({"query": "  "}, "PaperAuthor"),
        ({}, "PaperAuthor"),
    ],
)
def test_summary_names_what_the_call_does(
    arguments: dict[str, PlainTree],
    label: str,
) -> None:
    assert _tool().summary(arguments) == label


def test_lookups_run_beside_anything(tmp_path: Path) -> None:
    assert _tool().serialize_key(ToolContext(cwd=tmp_path), {"query": "x"}) is None


@pytest.mark.asyncio
async def test_a_name_finds_authors_and_is_kept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake(
        monkeypatch,
        "search_authors",
        AuthorSearchResult(records=[_ADA, _BOB], total=9),
    )
    tool = _tool()
    first = await tool.call(ToolContext(cwd=tmp_path), {"query": " Ada ", "limit": 2})
    again = await tool.call(ToolContext(cwd=tmp_path), {"query": "Ada", "limit": 2})
    assert (
        first.text()
        == again.text()
        == (
            f"{format_author_line(_ADA)}\n{format_author_line(_BOB)}"
            "\n... (showing 2 of 9; tighten filters for more)"
        )
    )
    assert first.call_id == ""
    assert calls == [_Call(args=("Ada",), kwargs={"limit": 2, "session": tool.session})]


@pytest.mark.asyncio
async def test_a_name_that_finds_no_one_says_so(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake(monkeypatch, "search_authors", AuthorSearchResult(records=[], total=0))
    result = await _tool().call(ToolContext(cwd=tmp_path), {"query": "Nobody"})
    assert result.text() == "(no results)"


@pytest.mark.asyncio
async def test_ids_are_looked_up_in_one_batch_and_kept_when_all_are_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake(monkeypatch, "author_metadata", [_ADA, _BOB])
    tool = _tool()
    first = await tool.call(ToolContext(cwd=tmp_path), {"ids": "1741101, 2262347"})
    again = await tool.call(
        ToolContext(cwd=tmp_path),
        {"ids": ["1741101", "2262347"]},
    )
    assert (
        first.text()
        == again.text()
        == (f"{format_author_block(_ADA)}\n\n{format_author_block(_BOB)}")
    )
    assert calls == [
        _Call(args=(["1741101", "2262347"],), kwargs={"session": tool.session}),
    ]


@pytest.mark.asyncio
async def test_an_id_not_found_says_so_and_is_asked_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake(monkeypatch, "author_metadata", [_ADA, None], [_ADA, None])
    tool = _tool()
    for _ in range(2):
        result = await tool.call(ToolContext(cwd=tmp_path), {"ids": ["1741101", "9"]})
        assert result.text() == f"{format_author_block(_ADA)}\n\n9: not found"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_an_authors_papers_come_with_every_filter_and_are_kept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake(
        monkeypatch,
        "author_papers",
        Listing(records=[_PAPER], complete=False),
        Listing(records=[], complete=True),
    )
    tool = _tool()
    arguments: dict[str, PlainTree] = {
        "ids": "1741101",
        "operation": "papers",
        "limit": 5,
        "year_from": 2015,
        "year_to": 2020,
        "abstract_chars": 3,
    }
    first = await tool.call(ToolContext(cwd=tmp_path), arguments)
    again = await tool.call(ToolContext(cwd=tmp_path), arguments)
    none = await tool.call(
        ToolContext(cwd=tmp_path),
        {"ids": "2262347", "operation": "papers"},
    )
    assert (
        first.text()
        == again.text()
        == (
            format_record(_PAPER, abstract_chars=3)
            + "\n... (more matches exist; raise 'limit' or narrow the years)"
        )
    )
    assert none.text() == "(no results)"
    assert calls[0] == _Call(
        args=("1741101",),
        kwargs={
            "limit": 5,
            "year_from": 2015,
            "year_to": 2020,
            "session": tool.session,
        },
    )
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_a_complete_listing_has_no_notice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake(monkeypatch, "author_papers", Listing(records=[_PAPER], complete=True))
    result = await _tool().call(
        ToolContext(cwd=tmp_path),
        {"ids": "1741101", "operation": "papers"},
    )
    assert result.text() == format_record(_PAPER)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("search_authors", {"query": "Ada"}),
        ("author_metadata", {"ids": "1741101"}),
        ("author_papers", {"ids": "1741101", "operation": "papers"}),
    ],
)
async def test_a_failed_lookup_is_an_error_and_is_not_kept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    arguments: dict[str, PlainTree],
) -> None:
    calls = _fake(monkeypatch, name, PaperError("S2 is down"), PaperError("still down"))
    tool = _tool()
    first = await tool.call(ToolContext(cwd=tmp_path), arguments)
    again = await tool.call(ToolContext(cwd=tmp_path), arguments)
    assert (first.is_error, first.text()) == (True, "S2 is down")
    assert again.text() == "still down"
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({}, "Set exactly one of 'query' or 'ids'."),
        ({"query": "  ", "ids": []}, "Set exactly one of 'query' or 'ids'."),
        ({"query": "Ada", "ids": "1741101"}, "Set exactly one of 'query' or 'ids'."),
        (
            {"query": "Ada", "operation": "papers"},
            "'operation=papers' needs exactly one id",
        ),
        (
            {"ids": ["1", "2"], "operation": "papers"},
            "'operation=papers' needs exactly one id",
        ),
        ({"ids": "1", "year_from": 2000}, "only apply to operation='papers'"),
        ({"query": "Ada", "year_to": 2000}, "only apply to operation='papers'"),
        (
            {"ids": "1", "operation": "papers", "year_from": 2021, "year_to": 2020},
            r"'year_from' \(2021\) must not exceed 'year_to' \(2020\)\.",
        ),
        ({"query": "Ada", "limit": 0}, "'limit' must be at least 1"),
        (
            {"ids": "1", "operation": "papers", "abstract_chars": 0},
            "'abstract_chars' must be at least 1",
        ),
        ({"ids": "1", "operation": "works"}, "'operation' must be one of"),
        ({"ids": 5}, "'ids' must be"),
    ],
)
async def test_arguments_that_contradict_are_refused_before_any_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, PlainTree],
    message: str,
) -> None:
    calls = [
        _fake(monkeypatch, name)
        for name in ("search_authors", "author_metadata", "author_papers")
    ]
    with pytest.raises(ValueError, match=message):
        await _tool().call(ToolContext(cwd=tmp_path), arguments)
    assert calls == [[], [], []]


@pytest.mark.asyncio
async def test_equal_years_bound_one_year(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake(monkeypatch, "author_papers", Listing(records=[], complete=True))
    await _tool().call(
        ToolContext(cwd=tmp_path),
        {"ids": "1", "operation": "papers", "year_from": 2020, "year_to": 2020},
    )
    assert [(c.kwargs["year_from"], c.kwargs["year_to"]) for c in calls] == [
        (2020, 2020),
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Call:
    """One lookup the tool asked for."""

    args: tuple[object, ...]
    kwargs: dict[str, object]


def _tool() -> PaperAuthor:
    return PaperAuthor(session=FetchSession())


def _fake(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *answers: object,
) -> list[_Call]:
    """Replace ``paper_author.<name>``: answer in turn, raise an exception given."""
    calls: list[_Call] = []
    pending = list(answers)

    def lookup(*args: object, **kwargs: object) -> object:
        calls.append(_Call(args=args, kwargs=kwargs))
        answer = pending.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(paper_author, name, lookup)
    return calls


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
