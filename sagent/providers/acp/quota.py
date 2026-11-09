"""Fetch native subscription and model quota for Google Antigravity / CCPA."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import logging

from treekle import json_loads

from sagent.lib.userdirs import data_dir
from sagent.providers.usage import ProviderUsage, format_countdown, format_percent


if TYPE_CHECKING:
    import httpx2


__all__ = [
    "AcpQuotaSnapshot",
    "ModelQuota",
    "fetch_acp_quota",
    "format_acp_usage",
    "read_antigravity_token",
]

logger = logging.getLogger(__name__)

CCPA_ENDPOINTS = (
    "https://cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.googleapis.com",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelQuota:
    """Quota metadata for a single Antigravity model."""

    model_id: str
    display_name: str
    remaining_fraction: float | None = None
    reset_time: str | None = None
    seconds_until_reset: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AcpQuotaSnapshot:
    """Native Antigravity quota and tier information."""

    tier_id: str = ""
    tier_name: str = ""
    email: str = ""
    models: dict[str, ModelQuota] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def read_antigravity_token(custom_path: Path | None = None) -> str | None:
    """Read the OAuth access token from agy's credential files."""
    paths: list[Path] = []
    if custom_path is not None:
        paths.append(custom_path)
    paths.append(Path.home() / ".gemini" / "antigravity-cli" / "antigravity-oauth-token")
    paths.append(Path.home() / ".gemini" / "antigravity" / "antigravity-oauth-token")
    paths.append(data_dir().parent / "antigravity-cli" / "antigravity-oauth-token")

    for path in paths:
        try:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            data = json_loads(text, dict[str, Any])
            token_field = data.get("token")
            if isinstance(token_field, dict):
                access_token = token_field.get("access_token")
                if isinstance(access_token, str) and access_token:
                    return access_token
            elif isinstance(token_field, str) and token_field:
                return token_field
        except Exception:
            continue
    return None


async def fetch_acp_quota(
    http: httpx2.AsyncClient,
    *,
    token: str | None = None,
) -> AcpQuotaSnapshot | None:
    """Query Google CCPA for current model quotas and subscription tier.

    Returns ``None`` when unauthenticated or unreachable. Never raises.
    """
    auth_token = read_antigravity_token() if token is None else token
    if not auth_token:
        return None

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
        "User-Agent": "antigravity/1.0",
    }

    tier_id, tier_name, email = "", "", ""
    models_dict: dict[str, ModelQuota] = {}
    raw_response: dict[str, Any] = {}

    for base_url in CCPA_ENDPOINTS:
        try:
            # 1. Fetch tier / plan information
            tier_resp = await http.post(
                f"{base_url}/v1internal:loadCodeAssist",
                headers=headers,
                json={},
                timeout=5.0,
            )
            if tier_resp.status_code == 200:
                tier_data = json_loads(tier_resp.text, dict[str, Any])
                raw_response["loadCodeAssist"] = tier_data
                tier_info = tier_data.get("currentTier")
                if isinstance(tier_info, dict):
                    tier_id = str(tier_info.get("id") or "")
                    tier_name = str(tier_info.get("name") or "")

            # 2. Fetch available models and quotaInfo
            models_resp = await http.post(
                f"{base_url}/v1internal:fetchAvailableModels",
                headers=headers,
                json={},
                timeout=5.0,
            )
            if models_resp.status_code == 200:
                models_data = json_loads(models_resp.text, dict[str, Any])
                raw_response["fetchAvailableModels"] = models_data
                raw_models = models_data.get("models")
                now = datetime.now(timezone.utc)
                if isinstance(raw_models, dict):
                    for model_id, model_info in raw_models.items():
                        if not isinstance(model_info, dict):
                            continue
                        disp_name = str(model_info.get("displayName") or model_id)
                        quota_info = model_info.get("quotaInfo")
                        rem_frac: float | None = None
                        reset_str: str | None = None
                        secs_until: float | None = None
                        if isinstance(quota_info, dict):
                            raw_rem = quota_info.get("remainingFraction")
                            if isinstance(raw_rem, (int, float)):
                                rem_frac = float(raw_rem)
                            reset_val = quota_info.get("resetTime")
                            if isinstance(reset_val, str) and reset_val:
                                reset_str = reset_val
                                try:
                                    dt = datetime.fromisoformat(reset_val.replace("Z", "+00:00"))
                                    secs_until = max((dt - now).total_seconds(), 0.0)
                                except Exception:
                                    pass
                        models_dict[model_id] = ModelQuota(
                            model_id=model_id,
                            display_name=disp_name,
                            remaining_fraction=rem_frac,
                            reset_time=reset_str,
                            seconds_until_reset=secs_until,
                        )
                # Successful response received from this endpoint
                break
        except Exception as error:
            logger.debug("Failed to query CCPA at %s: %s", base_url, error)
            continue

    if not models_dict and not tier_name:
        return None

    return AcpQuotaSnapshot(
        tier_id=tier_id,
        tier_name=tier_name,
        email=email,
        models=models_dict,
        raw=raw_response,
    )


def format_acp_usage(
    snapshot: AcpQuotaSnapshot,
    *,
    current_model: str = "",
) -> ProviderUsage:
    """Format an AcpQuotaSnapshot into a human-readable ProviderUsage."""
    lines: list[str] = []
    if snapshot.tier_name:
        desc = snapshot.tier_name
        if snapshot.tier_id and snapshot.tier_id != snapshot.tier_name.lower():
            desc += f" ({snapshot.tier_id})"
        lines.append(f"Tier: {desc}")

    summary_parts: list[str] = []
    if snapshot.tier_name:
        summary_parts.append(snapshot.tier_name)

    # Put the active model or primary models first
    sorted_models = sorted(
        snapshot.models.values(),
        key=lambda m: (m.model_id != current_model, m.remaining_fraction is None, m.model_id),
    )

    primary_model_summary = ""
    for m in sorted_models:
        if m.remaining_fraction is None:
            continue
        pct_left = format_percent(m.remaining_fraction)
        countdown = (
            f" (resets in {format_countdown(m.seconds_until_reset)})"
            if m.seconds_until_reset is not None
            else ""
        )
        line = f"{m.display_name}: {pct_left} remaining{countdown}"
        lines.append(line)
        if not primary_model_summary and (not current_model or current_model in m.model_id):
            primary_model_summary = f"{m.display_name}: {pct_left} left"

    if primary_model_summary:
        summary = f"{snapshot.tier_name}: {primary_model_summary}" if snapshot.tier_name else primary_model_summary
    elif snapshot.tier_name:
        summary = f"Antigravity ({snapshot.tier_name})"
    else:
        summary = "Antigravity"

    return ProviderUsage(
        provider="acp",
        summary=summary,
        details=tuple(lines),
        raw=dict(snapshot.raw),
    )
