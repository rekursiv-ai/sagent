"""Tests for the shared CLI-subprocess respawn cadence."""

from __future__ import annotations

from sagent.providers.lib.cli_respawn import (
    CONTEXT_FRACTION_RESPAWN_THRESHOLD,
    TURN_RESPAWN_THRESHOLD,
    respawn_for_cadence,
)


def test_respawn_when_turn_cap_reached() -> None:
    assert respawn_for_cadence(
        turn_count=TURN_RESPAWN_THRESHOLD,
        last_input_tokens=0,
        max_request_tokens=1_000_000,
    )


def test_no_respawn_below_turn_cap_and_context_fraction() -> None:
    assert not respawn_for_cadence(
        turn_count=TURN_RESPAWN_THRESHOLD - 1,
        last_input_tokens=1,
        max_request_tokens=1_000_000,
    )


def test_respawn_when_context_fraction_crossed() -> None:
    max_tokens = 1_000_000
    over = int(max_tokens * CONTEXT_FRACTION_RESPAWN_THRESHOLD) + 1
    assert respawn_for_cadence(
        turn_count=0,
        last_input_tokens=over,
        max_request_tokens=max_tokens,
    )


def test_no_respawn_exactly_at_context_fraction() -> None:
    # Strict ``>``: exactly at the fraction does not trip.
    max_tokens = 1_000_000
    at = int(max_tokens * CONTEXT_FRACTION_RESPAWN_THRESHOLD)
    assert not respawn_for_cadence(
        turn_count=0,
        last_input_tokens=at,
        max_request_tokens=max_tokens,
    )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
