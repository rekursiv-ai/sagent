"""Tests for PaperDetails: metadata, the citation graph, its cache, and errors.

``wesearch`` owns every lookup; each is replaced where the tool imports
it, so nothing reaches the network.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import TYPE_CHECKING, cast, get_args, get_type_hints

import re

from wesearch.fetch import FetchSession
from wesearch.paper.details import Listing
from wesearch.paper.errors import PaperError
from wesearch.paper.render import format_block, format_record
from wesearch.paper.types import PaperRecord

import pytest

from sagent.tools import (
    paper_details,
)
from sagent.tools.paper_details import PaperDetails
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from pathlib import Path

    from treekle import PlainTree


_ONE = PaperRecord(title="Attention", year=2017, abstract="All you need, really.")
_TWO = PaperRecord(title="Diffusion", year=2020)


def test_a_lookup_needs_the_session_it_shares() -> None:
    (session,) = [f for f in fields(PaperDetails) if f.name == "session"]
    assert (session.default, session.default_factory) == (MISSING, MISSING)


def test_summary_names_the_operation_and_the_ids() -> None:
    tool = _tool()
    assert tool.summary({"ids": "2106.15928"}) == "PaperDetails 2106.15928"
    assert tool.summary({"ids": ["a", "b"], "operation": "citations"}) == (
        "PaperDetails citations a (+1 more)"
    )
    assert tool.summary({"ids": "a", "operation": "references"}) == (
        "PaperDetails references a"
    )
    assert tool.summary({"ids": "a", "operation": "other"}) == "PaperDetails a"
    assert tool.summary({}) == "PaperDetails ?"


def test_lookups_run_beside_anything(tmp_path: Path) -> None:
    assert _tool().serialize_key(ToolContext(cwd=tmp_path), {"ids": "a"}) is None


@pytest.mark.asyncio
async def test_one_id_is_looked_up_alone_with_the_shared_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake(monkeypatch, "metadata", _ONE)
    tool = _tool()
    result = await tool.call(ToolContext(cwd=tmp_path), {"ids": "arXiv:2106.15928"})
    assert result.text() == format_block(_ONE)
    assert result.call_id == ""
    assert calls == [
        _Call(args=("arxiv", "2106.15928"), kwargs={"session": tool.session}),
    ]


@pytest.mark.asyncio
async def test_many_ids_are_looked_up_in_one_batch_in_input_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake(monkeypatch, "metadata_batch", [_ONE, None, _TWO])
    tool = _tool()
    result = await tool.call(
        ToolContext(cwd=tmp_path),
        {"ids": ["2106.15928", "10.1000/gone", "doi:10.1000/b"], "abstract_chars": 4},
    )
    assert result.text() == (
        f"{format_block(_ONE, abstract_chars=4)}\n\n10.1000/gone: not found\n\n"
        f"{format_block(_TWO, abstract_chars=4)}"
    )
    assert calls == [
        _Call(
            args=(["ARXIV:2106.15928", "DOI:10.1000/gone", "DOI:10.1000/b"],),
            kwargs={"session": tool.session},
        ),
    ]


@pytest.mark.asyncio
async def test_metadata_is_kept_by_canonical_id_only_when_every_paper_is_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _tool()
    context = ToolContext(cwd=tmp_path)
    batch = _fake(monkeypatch, "metadata_batch", [_ONE, None], [_ONE, _TWO])
    single = _fake(monkeypatch, "metadata", _ONE)
    await tool.call(context, {"ids": ["2106.15928", "10.1000/x"]})
    await tool.call(context, {"ids": ["2106.15928", "10.1000/x"]})
    kept = await tool.call(context, {"ids": ["arXiv:2106.15928", "doi:10.1000/x"]})
    await tool.call(context, {"ids": "2106.15928"})
    await tool.call(context, {"ids": "arxiv:2106.15928"})
    assert kept.text() == f"{format_block(_ONE)}\n\n{format_block(_TWO)}"
    assert (len(batch), len(single)) == (2, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "name", "kwargs"),
    [
        (
            {"ids": "10.1000/x", "operation": "references", "limit": 3},
            "references",
            {"limit": 3, "source": "s2"},
        ),
        (
            {"ids": "10.1000/x", "operation": "references", "source": "openalex"},
            "references",
            {"limit": None, "source": "openalex"},
        ),
        (
            {
                "ids": "10.1000/x",
                "operation": "citations",
                "influential_only": True,
                "year_from": 2020,
                "limit": 7,
            },
            "citations",
            {
                "limit": 7,
                "source": "s2",
                "influential_only": True,
                "year_from": 2020,
            },
        ),
        (
            {"ids": "10.1000/x", "operation": "citations", "source": "openalex"},
            "citations",
            {
                "limit": None,
                "source": "openalex",
                "influential_only": False,
                "year_from": None,
            },
        ),
    ],
)
async def test_the_graph_is_walked_from_one_paper_with_every_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, PlainTree],
    name: str,
    kwargs: dict[str, object],
) -> None:
    calls = _fake(monkeypatch, name, Listing(records=[_ONE, _TWO], complete=True))
    tool = _tool()
    result = await tool.call(ToolContext(cwd=tmp_path), arguments)
    assert result.text() == f"{format_record(_ONE)}\n{format_record(_TWO)}"
    assert result.call_id == ""
    assert calls == [
        _Call(args=("doi", "10.1000/x"), kwargs={**kwargs, "session": tool.session}),
    ]


@pytest.mark.asyncio
async def test_a_listing_says_when_more_exist_and_when_there_are_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake(
        monkeypatch,
        "references",
        Listing(records=[_ONE], complete=False),
        Listing(records=[], complete=True),
    )
    context = ToolContext(cwd=tmp_path)
    more = await _tool().call(
        context,
        {"ids": "10.1000/x", "operation": "references", "abstract_chars": 3},
    )
    none = await _tool().call(context, {"ids": "10.1000/y", "operation": "references"})
    assert more.text() == (
        format_record(_ONE, abstract_chars=3)
        + "\n... (more matches exist; raise 'limit' to see them)"
    )
    assert none.text() == "(no results)"


@pytest.mark.asyncio
async def test_a_walk_is_kept_by_paper_and_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _tool()
    context = ToolContext(cwd=tmp_path)
    calls = _fake(
        monkeypatch,
        "citations",
        Listing(records=[_ONE], complete=True),
        Listing(records=[_TWO], complete=True),
        Listing(records=[_TWO], complete=True),
    )
    first = await tool.call(context, {"ids": "10.1000/x", "operation": "citations"})
    again = await tool.call(context, {"ids": "doi:10.1000/x", "operation": "citations"})
    other = await tool.call(
        context,
        {"ids": "10.1000/x", "operation": "citations", "year_from": 2000},
    )
    paper = await tool.call(context, {"ids": "10.1000/y", "operation": "citations"})
    assert first.text() == again.text() == format_record(_ONE)
    assert again.call_id == ""
    assert other.text() == paper.text() == format_record(_TWO)
    assert len(calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("metadata", {"ids": "10.1000/x"}),
        ("metadata_batch", {"ids": ["10.1000/x", "10.1000/y"]}),
        ("references", {"ids": "10.1000/x", "operation": "references"}),
    ],
)
async def test_a_failed_lookup_is_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    arguments: dict[str, PlainTree],
) -> None:
    _fake(monkeypatch, name, PaperError("S2 is down"))
    result = await _tool().call(ToolContext(cwd=tmp_path), arguments)
    assert (result.is_error, result.text()) == (True, "S2 is down")


@pytest.mark.asyncio
async def test_an_id_of_no_known_shape_is_an_error(tmp_path: Path) -> None:
    result = await _tool().call(ToolContext(cwd=tmp_path), {"ids": "not-an-id"})
    assert result.is_error is True
    assert "not-an-id" in result.text()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"ids": ""}, "'ids' is empty."),
        ({"ids": []}, "'ids' is empty."),
        (
            {"ids": ["a", "b"], "operation": "citations"},
            "'operation' (references/citations) needs exactly one id.",
        ),
        (
            {"ids": "10.1000/x", "influential_only": True},
            "'influential_only' and 'year_from' only apply to operation='citations'.",
        ),
        (
            {"ids": "10.1000/x", "operation": "references", "year_from": 2000},
            "'influential_only' and 'year_from' only apply to operation='citations'.",
        ),
        ({"ids": "10.1000/x", "limit": 0}, "'limit' must be at least 1, got 0."),
        (
            {"ids": "10.1000/x", "abstract_chars": 0},
            "'abstract_chars' must be at least 1, got 0.",
        ),
        (
            {"ids": "10.1000/x", "operation": "cites"},
            "'operation' must be one of ['references', 'citations'], got 'cites'.",
        ),
        ({"ids": 5}, "'ids' must be array, got 5. 'ids' must be string, got 5."),
        ({}, "Missing argument 'ids'."),
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
        for name in (
            "metadata",
            "metadata_batch",
            "references",
            "citations",
        )
    ]
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        await _tool().call(ToolContext(cwd=tmp_path), arguments)
    assert not any(calls)


@pytest.mark.asyncio
async def test_a_source_not_offered_is_refused_naming_those_that_are(
    tmp_path: Path,
) -> None:
    hint = cast(
        object,
        get_type_hints(PaperDetails.Arguments, include_extras=True)["source"],
    )
    offered = list(get_args(get_args(hint)[0]))
    assert offered[:2] == ["s2", "openalex"]
    with pytest.raises(
        ValueError,
        match=f"^{re.escape(f"'source' must be one of {offered}, got 'bing'.")}$",
    ):
        await _tool().call(
            ToolContext(cwd=tmp_path),
            {"ids": "10.1000/x", "source": "bing"},
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Call:
    """One lookup the tool asked for."""

    args: tuple[object, ...]
    kwargs: dict[str, object]


def _tool() -> PaperDetails:
    return PaperDetails(session=FetchSession())


def _fake(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *answers: object,
) -> list[_Call]:
    """Replace ``paper_details.<name>``: answer in turn, raise an exception given."""
    calls: list[_Call] = []
    pending = list(answers)

    def lookup(*args: object, **kwargs: object) -> object:
        calls.append(_Call(args=args, kwargs=kwargs))
        answer = pending.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(paper_details, name, lookup)
    return calls


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
