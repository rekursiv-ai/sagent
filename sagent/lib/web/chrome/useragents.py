"""Real-browser User-Agent pools, one per impersonated browser kind.

Two maintained pools of real User-Agents, filtered from the ``intoli/user-agents``
dataset: ``chrome_desktop`` (desktop Chrome) and ``chrome_android`` (Android
Chrome). Each kind pairs with the matching curl_cffi TLS-impersonation target
(:func:`impersonate_target`), so a request's UA and TLS fingerprint always agree
-- a mismatch between them is itself a bot signal.

The hot path (:func:`user_agent_pool`, :func:`draw_user_agent`,
:func:`impersonate_target`) only reads local pool files. :func:`refresh`
re-downloads and re-filters the dataset; run it as a maintenance step::

    python -m sagent.lib.web.chrome.useragents
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Literal, cast

import gzip
import json
import logging
import random


__all__ = [
    "UserAgentKind",
    "draw_user_agent",
    "impersonate_target",
    "kind_for_impersonate",
    "refresh",
    "user_agent_pool",
]

type UserAgentKind = Literal["chrome_desktop", "chrome_android"]

logger = logging.getLogger(__name__)

_RNG = random.SystemRandom()


def impersonate_target(kind: UserAgentKind) -> str:
    """Return the curl_cffi impersonation target matching ``kind``'s User-Agent."""
    return "chrome_android" if kind == "chrome_android" else "chrome"


def kind_for_impersonate(impersonate: str) -> UserAgentKind:
    """Return the UA pool kind matching a curl_cffi impersonation target.

    The inverse of :func:`impersonate_target`, kept beside it as the ONE source
    of truth for the impersonate<->kind mapping (an unknown target degrades to
    the desktop pool, matching :func:`impersonate_version_platform`).
    """
    return "chrome_android" if impersonate == "chrome_android" else "chrome_desktop"


@cache
def user_agent_pool(kind: UserAgentKind) -> tuple[str, ...]:
    """Return ``kind``'s User-Agent pool, loaded from its file (cached)."""
    return tuple(line for line in _pool_path(kind).read_text().splitlines() if line)


def draw_user_agent(kind: UserAgentKind) -> str:
    """Return a random real User-Agent from ``kind``'s pool."""
    return _RNG.choice(user_agent_pool(kind))


def refresh(kind: UserAgentKind) -> None:
    """Rewrite ``kind``'s pool file from the intoli dataset, applying its filter."""
    # Lazy: only this maintenance path needs fetch, and a top-level import would
    # form a fetch <-> useragents cycle (fetch draws from a pool).
    from sagent.lib.web.fetch import (  # noqa: PLC0415 -- breaks import cycle
        RequestParams,
        fetch,
    )

    url = (
        "https://raw.githubusercontent.com/intoli/user-agents/"
        "main/src/user-agents.json.gz"
    )
    body, _ = fetch(url, request=RequestParams(timeout_sec=30))
    parsed: object = json.loads(gzip.decompress(body))
    if not isinstance(parsed, list):
        raise RuntimeError(f"expected JSON array from {url}; upstream shape changed?")  # noqa: TRY004
    keep = _is_android_chrome if kind == "chrome_android" else _is_desktop_chrome
    selected_set: set[str] = set()
    for record in cast("list[object]", parsed):
        if not isinstance(record, dict):
            continue
        record = cast("dict[str, object]", record)
        ua = record.get("userAgent")
        device = record.get("deviceCategory")
        if isinstance(ua, str) and keep(ua, device if isinstance(device, str) else ""):
            selected_set.add(ua)
    selected = sorted(selected_set)
    if not selected:
        raise RuntimeError(f"refresh produced 0 user agents from {url} for {kind}.")
    _pool_path(kind).write_text("\n".join(selected) + "\n")
    user_agent_pool.cache_clear()
    logger.info("wrote %d user agents to %s", len(selected), _pool_path(kind))


def _is_desktop_chrome(ua: str, device: str) -> bool:
    """A plain desktop Chrome UA (not mobile, not an Edge/Opera/Samsung variant)."""
    return (
        device == "desktop"
        and "Chrome" in ua
        and "Mobile" not in ua
        and not any(v in ua for v in ("Edg/", "OPR/", "SamsungBrowser"))
    )


def _is_android_chrome(ua: str, device: str) -> bool:
    """A plain Android Chrome UA (matching the long-used search.py filter)."""
    del device  # UA string is authoritative for the Android/Chrome check.
    return (
        "Android" in ua
        and "Chrome" in ua
        and "Samsung" not in ua
        and "Android 10; K" not in ua
    )


def _pool_path(kind: UserAgentKind) -> Path:
    """The file holding ``kind``'s pool, alongside this module."""
    return Path(__file__).with_name(f"{kind}_useragents.txt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    refresh("chrome_desktop")
    refresh("chrome_android")
