"""Normalize provider rate-limit headers into a :class:`UsageSnapshot`.

Each provider receives rate-limit telemetry in its own header dialect.
These helpers translate the two dialects that carry per-window
utilization -- Anthropic ``unified-*`` and OpenAI ``x-ratelimit-*`` --
into the provider-agnostic :class:`UsageSnapshot` the REPL surfaces.

``UsageWindow.resets_at`` is always a Unix wall-clock epoch (seconds). The
Anthropic ``-reset`` headers are already epochs; OpenAI ``-reset`` headers are
durations-from-now and are converted to an epoch here so the field's contract
holds across providers.
"""

from __future__ import annotations

from collections.abc import Mapping

import math
import re
import time

from sagent.types.model import UsageSnapshot, UsageWindow


# Non-blocking unified statuses; ``rejected`` / ``rate_limited`` throttle.
_ANTHROPIC_ALLOWED = frozenset({"allowed", "allowed_warning"})


def anthropic_usage(headers: Mapping[str, str]) -> UsageSnapshot | None:
    """Map Anthropic ``unified-*`` headers into a :class:`UsageSnapshot`.

    The omnibus ``anthropic-ratelimit-unified-status`` is the header that
    flips to ``rejected`` first on a hard limit (the retry layer keys off it);
    when it signals a block, every window is marked blocked so the usage
    surface and the retry layer agree.

    Args:
      headers: Response headers (case-insensitive lookup; normalized here).

    Returns:
      snapshot: Per-window usage, or ``None`` when no unified headers present.

    """
    headers = _lower_keys(headers)
    omnibus = headers.get("anthropic-ratelimit-unified-status")
    omnibus_blocked = omnibus is not None and omnibus.strip().lower() not in (
        _ANTHROPIC_ALLOWED
    )
    windows: list[UsageWindow] = []
    for window in ("5h", "7d"):
        util = _finite_float(
            headers.get(f"anthropic-ratelimit-unified-{window}-utilization")
        )
        reset = _finite_float(
            headers.get(f"anthropic-ratelimit-unified-{window}-reset")
        )
        status = headers.get(f"anthropic-ratelimit-unified-{window}-status")
        if util is None and reset is None and status is None and not omnibus_blocked:
            continue
        window_blocked = status is not None and status.strip().lower() not in (
            _ANTHROPIC_ALLOWED
        )
        windows.append(
            UsageWindow(
                label=window,
                utilization=None if util is None else max(0.0, min(1.0, util)),
                resets_at=reset,
                blocked=omnibus_blocked or window_blocked,
            )
        )
    if not windows:
        return None
    return UsageSnapshot(windows=tuple(windows))


def openai_usage(headers: Mapping[str, str]) -> UsageSnapshot | None:
    """Map OpenAI ``x-ratelimit-*`` headers into a :class:`UsageSnapshot`.

    OpenAI reports ``limit`` / ``remaining`` / ``reset`` per resource;
    utilization is derived as ``1 - remaining / limit``. ``reset`` is a
    duration string (e.g. ``"6m0s"`` / ``"1.5s"`` / ``"500ms"``); it is
    converted to a wall-clock epoch so ``resets_at`` stays uniform.

    Args:
      headers: Response headers (case-insensitive lookup; normalized here).

    Returns:
      snapshot: Per-window usage, or ``None`` when no x-ratelimit headers.

    """
    headers = _lower_keys(headers)
    windows: list[UsageWindow] = []
    for resource in ("requests", "tokens"):
        limit = _finite_float(headers.get(f"x-ratelimit-limit-{resource}"))
        remaining = _finite_float(headers.get(f"x-ratelimit-remaining-{resource}"))
        delay = _openai_reset_seconds(headers.get(f"x-ratelimit-reset-{resource}"))
        if limit is None and remaining is None and delay is None:
            continue
        util: float | None = None
        if limit is not None and limit > 0 and remaining is not None:
            util = max(0.0, min(1.0, 1.0 - remaining / limit))
        windows.append(
            UsageWindow(
                label=resource,
                utilization=util,
                resets_at=None if delay is None else time.time() + delay,
                blocked=remaining is not None and remaining <= 0,
            )
        )
    if not windows:
        return None
    return UsageSnapshot(windows=tuple(windows))


def _lower_keys(headers: Mapping[str, str]) -> Mapping[str, str]:
    """Return ``headers`` with lowercase keys for case-insensitive lookup."""
    return {k.lower(): v for k, v in headers.items()}


def _finite_float(raw: str | None) -> float | None:
    """Parse a finite float; ``None`` when absent, malformed, or non-finite."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


# Duration segments like ``"6m"``, ``"0s"``, ``"500ms"``. ``ms`` precedes the
# single-letter alternatives so it is matched before a bare ``m`` + ``s``.
_DURATION_SEGMENT = re.compile(r"(\d+(?:\.\d+)?)(ms|[dhms])")
_DURATION_UNIT_SEC = {"d": 86_400.0, "h": 3_600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def _openai_reset_seconds(raw: str | None) -> float | None:
    """Parse an OpenAI reset duration (e.g. ``"6m0s"``, ``"500ms"``) to seconds.

    Returns the duration as seconds-from-now (a delta); the caller converts it
    to a wall-clock epoch. A bare ``"0"`` (reset is now) yields ``0.0``.
    """
    if raw is None:
        return None
    text = raw.strip().lower()
    if not text:
        return None
    matches = list(_DURATION_SEGMENT.finditer(text))
    consumed = sum(len(m.group(0)) for m in matches)
    if consumed != len(text):
        # Allow a bare numeric (e.g. ``"0"``) meaning seconds; reject the rest.
        try:
            return max(0.0, float(text))
        except ValueError:
            return None
    return sum(float(m.group(1)) * _DURATION_UNIT_SEC[m.group(2)] for m in matches)
