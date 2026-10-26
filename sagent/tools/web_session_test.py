"""Tests that every web and paper tool forwards its session to wesearch."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import asyncio

from wesearch.fetch import FetchSession
from wesearch.paper.authors import AuthorSearchResult
from wesearch.paper.details import Listing
from wesearch.paper.search import SearchResult
from wesearch.paper.types import PaperRecord

import pytest

from sagent.tools.paper_author import PaperAuthor
from sagent.tools.paper_details import PaperDetails
from sagent.tools.paper_fetch import PaperFetch
from sagent.tools.paper_search import PaperSearch
from sagent.tools.web_search import WebSearch


if TYPE_CHECKING:
    from pathlib import Path


_RECORD = PaperRecord(title="T")


def _calls(mock: Mock) -> list[object]:
    return [call.kwargs.get("session") for call in mock.call_args_list]


def test_web_search_forwards_its_session() -> None:
    session = FetchSession()
    with patch(
        "sagent.tools.web_search.search",
        return_value=[],
    ) as search:
        asyncio.run(WebSearch(session=session).run({"query": "q"}))
        asyncio.run(
            WebSearch(session=session).run({"query": "q", "transport": "curl"}),
        )
    assert _calls(search) == [session, session]


def test_paper_search_forwards_its_session() -> None:
    session = FetchSession()
    with patch(
        "sagent.tools.paper_search.search",
        return_value=SearchResult(records=[], total=0, complete=True),
    ) as search:
        asyncio.run(PaperSearch(session=session).run({"query": "unique q 1"}))
    assert _calls(search) == [session]


@pytest.mark.parametrize(
    ("args", "target", "result"),
    [
        ({"ids": "2106.15928"}, "metadata", _RECORD),
        (
            {"ids": ["2106.15928", "2106.15929"]},
            "metadata_batch",
            [_RECORD, _RECORD],
        ),
        (
            {"ids": "2106.15930", "operation": "references"},
            "references",
            Listing(records=[], complete=True),
        ),
        (
            {"ids": "2106.15931", "operation": "citations"},
            "citations",
            Listing(records=[], complete=True),
        ),
    ],
)
def test_paper_details_forwards_its_session(
    args: dict[str, object],
    target: str,
    result: object,
) -> None:
    session = FetchSession()
    with patch(
        f"sagent.tools.paper_details.{target}",
        return_value=result,
    ) as call:
        asyncio.run(PaperDetails(session=session).run(args))
    assert _calls(call) == [session]


@pytest.mark.parametrize(
    ("args", "target", "result"),
    [
        (
            {"query": "unique author 1"},
            "search_authors",
            AuthorSearchResult(records=[], total=0),
        ),
        ({"ids": "111"}, "author_metadata", [None]),
        ({"ids": ["112", "113"]}, "author_metadata", [None, None]),
        (
            {"ids": "114", "operation": "papers"},
            "author_papers",
            Listing(records=[], complete=True),
        ),
    ],
)
def test_paper_author_forwards_its_session(
    args: dict[str, object],
    target: str,
    result: object,
) -> None:
    session = FetchSession()
    with patch(
        f"sagent.tools.paper_author.{target}",
        return_value=result,
    ) as call:
        asyncio.run(PaperAuthor(session=session).run(args))
    assert _calls(call) == [session]


def test_paper_fetch_forwards_its_session(tmp_path: Path) -> None:
    session = FetchSession()
    tool = PaperFetch(cache_dir=tmp_path, session=session)
    with (
        patch(
            "sagent.tools.paper_fetch.batch_oa_urls",
            return_value=[None, None],
        ) as batch,
        patch(
            "sagent.tools.paper_fetch.download",
            return_value=(b"%PDF-" + b"x" * 200, "arxiv"),
        ) as download,
    ):
        asyncio.run(tool.run({"ids": ["2106.15928", "2106.15929"]}))
    assert _calls(batch) == [session]
    assert _calls(download) == [session, session]


@pytest.mark.parametrize(
    "cls",
    [WebSearch, PaperSearch, PaperDetails, PaperAuthor, PaperFetch],
)
def test_a_tool_built_alone_owns_a_session(
    cls: type[WebSearch | PaperSearch | PaperDetails | PaperAuthor | PaperFetch],
) -> None:
    first, second = cls(), cls()
    assert isinstance(first.session, FetchSession)
    assert first.session is not second.session


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
