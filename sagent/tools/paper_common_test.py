"""Tests for what the Paper tools share: reading ``ids``, labels, and errors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wesearch.paper.details import Listing
from wesearch.paper.render import format_record
from wesearch.paper.types import PaperRecord

import pytest

from sagent.session.types import PaperSearchResult
from sagent.tools.paper_common import error, listed, paper_ids, summary_ids


if TYPE_CHECKING:
    from treekle import PlainTree


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        ("2106.15928", ["2106.15928"]),
        ("  2106.15928 ", ["2106.15928"]),
        (("2106.15928", "10.1/x"), ["2106.15928", "10.1/x"]),
        ('["2106.15928", "10.1/x"]', ["2106.15928", "10.1/x"]),
        ('["2106.15928","10.1/x"]', ["2106.15928", "10.1/x"]),
        ("2106.15928, 10.1000/xyz", ["2106.15928", "10.1000/xyz"]),
        ("2106.15928\n10.1000/xyz\n", ["2106.15928", "10.1000/xyz"]),
        (("2106.15928,10.1000/xyz",), ["2106.15928", "10.1000/xyz"]),
        (('["2106.15928"]',), ["2106.15928"]),
        # A DOI may hold a comma; one part that is no id keeps the whole whole.
        ("10.1000/a,b", ["10.1000/a,b"]),
        ("2106.15928, not an id", ["2106.15928, not an id"]),
        ('["2106.15928", 1]', ['["2106.15928", 1]']),
        ("[not json]", ["[not json]"]),
        (("", "  ", "2106.15928"), ["2106.15928"]),
        ('["", " x "]', ["x"]),
        ((), []),
    ],
)
def test_paper_ids_splits_only_a_bundle_of_ids(
    ids: tuple[str, ...] | str,
    expected: list[str],
) -> None:
    assert paper_ids(ids) == expected


def test_paper_ids_judges_a_bundle_by_the_test_it_is_given() -> None:
    assert paper_ids("1741101, 2262347", looks_like_id=str.isdigit) == [
        "1741101",
        "2262347",
    ]
    assert paper_ids("1741101, 2106.15928", looks_like_id=str.isdigit) == [
        "1741101, 2106.15928",
    ]


@pytest.mark.parametrize(
    ("arguments", "label"),
    [
        ({"ids": "2106.15928"}, "2106.15928"),
        ({"ids": ["2106.15928"]}, "2106.15928"),
        ({"ids": ["2106.15928", "10.1/x", "10.1/y"]}, "2106.15928 (+2 more)"),
        ({"ids": "2106.15928, 10.1000/xyz"}, "2106.15928 (+1 more)"),
        ({"ids": "x" * 40}, "x" * 40),
        ({"ids": "a" * 2 + "b" * 39}, "…" + "b" * 38),
        ({}, "?"),
        ({"ids": ""}, "?"),
        ({"ids": []}, "?"),
        ({"ids": 5}, "?"),
        ({"ids": [5, "2106.15928"]}, "2106.15928"),
        ({"ids": {"a": "b"}}, "?"),
    ],
)
def test_summary_ids_labels_whatever_was_passed(
    arguments: dict[str, PlainTree],
    label: str,
) -> None:
    """A summary labels the call before it is checked, so it must not raise."""
    assert summary_ids(arguments) == label


_PAPERS = [
    PaperRecord(title="Attention", year=2017, abstract="a" * 50),
    PaperRecord(title="Diffusion", doi="10.1/x"),
]


def test_a_listing_is_one_paper_per_line_with_each_abstract_cut() -> None:
    text = listed(Listing(records=_PAPERS, complete=True), 10, more="exist")

    assert text == "\n".join(format_record(r, abstract_chars=10) for r in _PAPERS)
    assert "a" * 11 not in text


@pytest.mark.parametrize("more", ["exist", "may exist"])
def test_an_incomplete_listing_says_whether_more_exist(more: str) -> None:
    text = listed(Listing(records=_PAPERS[:1], complete=False), None, more=more)

    assert text == (
        format_record(_PAPERS[0])
        + f"\n... (more matches {more}; raise 'limit' to see them)"
    )


def test_an_empty_listing_says_so() -> None:
    assert listed(Listing(records=[], complete=False), 5, more="exist") == (
        "(no results)"
    )


def test_error_is_an_error_result() -> None:
    assert error(PaperSearchResult, "boom") == PaperSearchResult(
        call_id="",
        shown="boom",
        is_error=True,
    )


@pytest.mark.parametrize("complete", [False, True])
def test_listed_returns_no_results_for_an_empty_listing(complete: bool) -> None:
    assert listed(Listing(records=[], complete=complete), None, more="exist") == (
        "(no results)"
    )


@pytest.mark.parametrize("abstract_chars", [None, 2])
@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("more", ["exist", "may exist"])
def test_listed_joins_records_and_marks_only_incomplete_results(
    abstract_chars: int | None,
    complete: bool,
    more: str,
) -> None:
    listing = Listing(
        records=[
            PaperRecord(title="First", abstract="abcd"),
            PaperRecord(title="Second"),
        ],
        complete=complete,
    )
    abstract = "abcd" if abstract_chars is None else "ab..."
    expected = (
        f"[no-id] First - unknown, ?, ?\n    abstract:\n    {abstract}\n"
        "[no-id] Second - unknown, ?, ?"
    )
    if not complete:
        expected += f"\n... (more matches {more}; raise 'limit' to see them)"
    assert listed(listing, abstract_chars, more=more) == expected


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
