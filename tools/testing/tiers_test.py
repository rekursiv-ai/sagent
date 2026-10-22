"""Tests for the pytest tier table."""

from __future__ import annotations

from itertools import combinations

import pytest

from tools.testing.tiers import TIERS, Tier


@pytest.mark.parametrize(
    ("name", "marker"),
    [
        (
            "pytest-global",
            (
                "not ci_smoke and not cluster and not cuda and not integration"
                " and not performance and not slow"
            ),
        ),
        (
            "pytest-slow-global",
            "slow and not cluster and not cuda and not integration and not performance",
        ),
        ("pytest-ci-smoke-global", "ci_smoke"),
        ("pytest-integration-global", "integration"),
    ],
)
def test_each_tier_spells_its_selection_as_one_marker_expression(
    name: str,
    marker: str,
) -> None:
    assert TIERS[name].marker == marker


def test_a_tier_selects_by_required_and_excluded_markers() -> None:
    tier = Tier(name="t", markers=("slow",), excluded=("cuda",))

    assert tier.selects({"slow"})
    assert tier.selects({"slow", "network"})
    assert not tier.selects({"slow", "cuda"})
    assert not tier.selects(set())


def test_no_test_runs_in_two_whole_suite_tiers() -> None:
    """Only the smoke tier overlaps the rest; any other overlap runs a test twice."""
    names = sorted({name for tier in TIERS.values() for name in tier.excluded})
    for size in range(len(names) + 1):
        for markers in map(set, combinations(names, size)):
            owners = [
                tier.name
                for tier in TIERS.values()
                if tier.selects(markers) and tier.name != "pytest-ci-smoke-global"
            ]
            assert len(owners) <= 1, (markers, owners)


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
