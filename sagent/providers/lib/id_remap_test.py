"""Tests for ``providers.lib.id_remap``: cross-provider tool-call ID remapping."""

from __future__ import annotations

from sagent.providers.lib.id_remap import IdRemapper


def test_first_mapping_uses_prefix_zero() -> None:
    r = IdRemapper("toolu_")
    assert r.map("orig-1") == "toolu_0"


def test_subsequent_mappings_increment() -> None:
    r = IdRemapper("call_")
    assert r.map("a") == "call_0"
    assert r.map("b") == "call_1"
    assert r.map("c") == "call_2"


def test_repeated_id_returns_same_mapping() -> None:
    r = IdRemapper("fc_")
    first = r.map("x")
    second = r.map("x")
    assert first == second
    # Counter does not advance for a repeated id.
    assert r.map("y") == "fc_1"


def test_tool_call_result_pair_round_trip() -> None:
    """Tool-call/result pairing within a single request stays consistent."""
    r = IdRemapper("toolu_")
    call_native = r.map("ext-abc")
    result_native = r.map("ext-abc")
    assert call_native == result_native


def test_distinct_prefix_independent_counters() -> None:
    a = IdRemapper("a_")
    b = IdRemapper("b_")
    assert a.map("x") == "a_0"
    assert b.map("x") == "b_0"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
