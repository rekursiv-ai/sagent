"""Tests for provider native usage dataclass and helpers."""

from __future__ import annotations

from sagent.providers.usage import ProviderUsage, format_countdown, format_percent


def test_provider_usage_creation() -> None:
    usage = ProviderUsage(
        provider="openai-sub",
        account="work",
        summary="5h: 15.0% used · 7d: 5.0% used",
        details=(
            "5-hour window: 15.0% used (resets in 2h 41m)",
            "7-day window: 5.0% used (resets in 5d 12h)",
        ),
        raw={"used_percent": 15},
    )
    assert usage.provider == "openai-sub"
    assert usage.account == "work"
    assert usage.summary == "5h: 15.0% used · 7d: 5.0% used"
    assert len(usage.details) == 2
    assert usage.raw["used_percent"] == 15


def test_format_countdown() -> None:
    assert format_countdown(0) == "now"
    assert format_countdown(-10) == "now"
    assert format_countdown(45) == "45s"
    assert format_countdown(120) == "2m"
    assert format_countdown(150) == "2m 30s"
    assert format_countdown(3600) == "1h"
    assert format_countdown(3720) == "1h 2m"
    assert format_countdown(86400) == "1d"
    assert format_countdown(90000) == "1d 1h"


def test_format_percent() -> None:
    assert format_percent(0.0) == "0.0%"
    assert format_percent(0.15) == "15.0%"
    assert format_percent(0.6286) == "62.9%"
    assert format_percent(1.0) == "100.0%"
    assert format_percent(1.5) == "100.0%"
    assert format_percent(-0.2) == "0.0%"
