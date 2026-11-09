"""Tests for Antigravity / CCPA quota fetching and formatting."""

from __future__ import annotations

from typing import Any

import httpx2
import pytest

from sagent.providers.acp.quota import (
    AcpQuotaSnapshot,
    ModelQuota,
    fetch_acp_quota,
    format_acp_usage,
    read_antigravity_token,
)


def _mock_transport(models_data: dict[str, Any], tier_data: dict[str, Any]) -> httpx2.MockTransport:
    def handler(request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        if "loadCodeAssist" in url:
            return httpx2.Response(200, json=tier_data)
        if "fetchAvailableModels" in url:
            return httpx2.Response(200, json=models_data)
        return httpx2.Response(404)

    return httpx2.MockTransport(handler)


@pytest.mark.asyncio
async def test_fetch_acp_quota_extracts_tier_and_models() -> None:
    tier_data = {
        "currentTier": {
            "id": "free-tier",
            "name": "Antigravity",
        }
    }
    models_data = {
        "models": {
            "gemini-3.5-flash-lite": {
                "displayName": "Gemini 3.5 Flash Lite",
                "quotaInfo": {
                    "remainingFraction": 0.628,
                    "resetTime": "2026-11-09T20:13:07Z",
                },
            },
            "claude-sonnet-4-6": {
                "displayName": "Claude Sonnet 4.6 (Thinking)",
                "quotaInfo": {
                    "remainingFraction": 1.0,
                    "resetTime": "2026-11-09T22:11:42Z",
                },
            },
        }
    }

    transport = _mock_transport(models_data, tier_data)
    async with httpx2.AsyncClient(transport=transport) as http:
        snapshot = await fetch_acp_quota(http, token="mock-token")
        assert snapshot is not None
        assert snapshot.tier_id == "free-tier"
        assert snapshot.tier_name == "Antigravity"
        assert len(snapshot.models) == 2
        flash = snapshot.models["gemini-3.5-flash-lite"]
        assert flash.display_name == "Gemini 3.5 Flash Lite"
        assert flash.remaining_fraction == 0.628


def test_format_acp_usage_formats_summary_and_lines() -> None:
    snapshot = AcpQuotaSnapshot(
        tier_id="free-tier",
        tier_name="Antigravity",
        models={
            "gemini-3.5-flash-lite": ModelQuota(
                model_id="gemini-3.5-flash-lite",
                display_name="Gemini 3.5 Flash Lite",
                remaining_fraction=0.628,
                seconds_until_reset=3600.0,
            ),
            "claude-sonnet-4-6": ModelQuota(
                model_id="claude-sonnet-4-6",
                display_name="Claude Sonnet 4.6 (Thinking)",
                remaining_fraction=1.0,
                seconds_until_reset=7200.0,
            ),
        },
    )
    usage = format_acp_usage(snapshot, current_model="gemini-3.5-flash-lite")
    assert usage.provider == "acp"
    assert "Gemini 3.5 Flash Lite: 62.8% left" in usage.summary
    assert any("Gemini 3.5 Flash Lite: 62.8% remaining (resets in 1h)" in line for line in usage.details)
    assert any("Claude Sonnet 4.6 (Thinking): 100.0% remaining (resets in 2h)" in line for line in usage.details)


@pytest.mark.asyncio
async def test_fetch_acp_quota_returns_none_when_no_token() -> None:
    async with httpx2.AsyncClient() as http:
        assert await fetch_acp_quota(http, token="") is None
