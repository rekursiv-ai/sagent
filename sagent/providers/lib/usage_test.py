"""Tests for ``providers.lib.usage``: header -> UsageSnapshot normalization."""

from __future__ import annotations

import time

from sagent.providers.lib.usage import (
    anthropic_usage,
    openai_usage,
)


def test_anthropic_usage_parses_windows() -> None:
    snap = anthropic_usage(
        {
            "anthropic-ratelimit-unified-5h-utilization": "0.56",
            "anthropic-ratelimit-unified-5h-reset": "1780531800",
            "anthropic-ratelimit-unified-5h-status": "allowed",
            "anthropic-ratelimit-unified-7d-utilization": "0.89",
            "anthropic-ratelimit-unified-7d-reset": "1780750800",
            "anthropic-ratelimit-unified-7d-status": "allowed_warning",
        }
    )
    assert snap is not None
    labels = {w.label: w for w in snap.windows}
    assert labels["5h"].utilization == 0.56
    assert labels["7d"].utilization == 0.89
    # allowed_warning is NOT blocked (Issue#316).
    assert labels["7d"].blocked is False
    assert labels["7d"].resets_at == 1780750800.0


def test_anthropic_usage_marks_rejected_window_blocked() -> None:
    snap = anthropic_usage(
        {
            "anthropic-ratelimit-unified-7d-utilization": "1.0",
            "anthropic-ratelimit-unified-7d-status": "rejected",
        }
    )
    assert snap is not None
    assert snap.windows[0].blocked is True
    assert snap.windows[0].utilization == 1.0


def test_anthropic_usage_omnibus_rejected_blocks_all_windows() -> None:
    # The omnibus unified-status (what the retry layer keys off) flips first;
    # the usage surface must agree -- every window is blocked.
    snap = anthropic_usage(
        {
            "anthropic-ratelimit-unified-status": "rejected",
            "anthropic-ratelimit-unified-5h-utilization": "0.6",
        }
    )
    assert snap is not None
    assert all(w.blocked for w in snap.windows)


def test_anthropic_usage_status_only_window() -> None:
    snap = anthropic_usage({"anthropic-ratelimit-unified-5h-status": "rejected"})
    assert snap is not None
    assert snap.windows[0].blocked is True
    assert snap.windows[0].utilization is None


def test_anthropic_usage_partial_windows() -> None:
    snap = anthropic_usage({"anthropic-ratelimit-unified-5h-utilization": "0.1"})
    assert snap is not None
    assert len(snap.windows) == 1
    assert snap.windows[0].label == "5h"


def test_anthropic_usage_clamps_overshoot_utilization() -> None:
    snap = anthropic_usage({"anthropic-ratelimit-unified-7d-utilization": "1.05"})
    assert snap is not None
    assert snap.windows[0].utilization == 1.0


def test_anthropic_usage_rejects_nonfinite_utilization() -> None:
    snap = anthropic_usage({"anthropic-ratelimit-unified-7d-utilization": "inf"})
    assert snap is None


def test_anthropic_usage_case_insensitive_keys() -> None:
    snap = anthropic_usage({"Anthropic-RateLimit-Unified-5h-Utilization": "0.4"})
    assert snap is not None
    assert snap.windows[0].utilization == 0.4


def test_anthropic_usage_none_without_headers() -> None:
    assert anthropic_usage({"x-other": "v"}) is None


def test_openai_usage_derives_utilization() -> None:
    before = time.time()
    snap = openai_usage(
        {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "250",
            "x-ratelimit-reset-requests": "6m0s",
            "x-ratelimit-limit-tokens": "100000",
            "x-ratelimit-remaining-tokens": "100000",
        }
    )
    after = time.time()
    assert snap is not None
    by = {w.label: w for w in snap.windows}
    assert by["requests"].utilization == 0.75  # 1 - 250/1000
    # resets_at is a wall-clock epoch (delay 360s applied to now).
    reset = by["requests"].resets_at
    assert reset is not None
    assert before + 360.0 <= reset <= after + 360.0
    assert by["tokens"].utilization == 0.0


def test_openai_usage_blocked_when_remaining_zero() -> None:
    snap = openai_usage(
        {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "0",
        }
    )
    assert snap is not None
    assert snap.windows[0].blocked is True
    assert snap.windows[0].utilization == 1.0


def test_openai_usage_reset_milliseconds() -> None:
    before = time.time()
    snap = openai_usage({"x-ratelimit-reset-tokens": "500ms"})
    after = time.time()
    assert snap is not None
    reset = snap.windows[0].resets_at
    assert reset is not None
    assert before + 0.5 <= reset <= after + 0.5


def test_openai_usage_reset_bare_zero_is_now() -> None:
    snap = openai_usage(
        {"x-ratelimit-limit-tokens": "100", "x-ratelimit-reset-tokens": "0"}
    )
    assert snap is not None
    reset = snap.windows[0].resets_at
    assert reset is not None
    assert abs(reset - time.time()) < 1.0


def test_openai_usage_reset_compound_units() -> None:
    before = time.time()
    snap = openai_usage({"x-ratelimit-reset-tokens": "1m30s"})
    after = time.time()
    assert snap is not None
    reset = snap.windows[0].resets_at
    assert reset is not None
    assert before + 90.0 <= reset <= after + 90.0


def test_openai_usage_none_without_headers() -> None:
    assert openai_usage({"content-type": "json"}) is None


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
