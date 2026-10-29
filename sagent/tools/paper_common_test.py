"""Tests for what the Paper tools share: reading ``ids``, labels, and errors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sagent.session.types import UncategorizedToolResult
from sagent.tools.paper_common import error, paper_ids, summary_ids


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


def test_error_is_an_error_result() -> None:
    assert error("boom") == UncategorizedToolResult(
        call_id="",
        content="boom",
        is_error=True,
    )


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
