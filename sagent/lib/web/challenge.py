"""Cross-site HTTP challenge detection for ``sagent.lib.web``.

This module recognizes only challenge technology shared across providers:
Cloudflare interstitials and generic CAPTCHA widgets. Provider page states belong
beside their parsers.

Cloudflare markers and titles follow FlareSolverr's structural detection model:
challenge-only markup or an interstitial title, never prose keywords.

FlareSolverr -- MIT License, Copyright (c) 2025 Diego Heras (ngosang).
https://github.com/FlareSolverr/FlareSolverr
"""

from __future__ import annotations

import re

from sagent.lib.web.errors import (
    BotDetectionError,
    CloudflareChallengeError,
    FetchError,
    PuzzleChallengeError,
)


__all__ = ["classify_challenge", "classify_http_error"]

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)


def classify_challenge(
    content: str | bytes,
    *,
    on_success_body: bool = False,
    cloudflare: tuple[str, ...] = (
        "cf_chl",
        "cf-challenge",
        "cf-challenge-running",
        "cf-please-wait",
        "challenge-spinner",
        "trk_jschal_js",
        "turnstile-wrapper",
        "cf-turnstile",
        "cf-turnstile-response",
    ),
    cloudflare_ambient: tuple[str, ...] = ("/cdn-cgi/challenge-platform/",),
    cloudflare_titles: tuple[str, ...] = (
        "just a moment...",
        "attention required! | cloudflare",
    ),
    puzzle_widget: tuple[str, ...] = (
        "g-recaptcha",
        "recaptcha",
        "h-captcha",
        "hcaptcha",
        "data-sitekey",
    ),
) -> type[BotDetectionError] | None:
    """Return the shared challenge class ``content`` proves, if any.

    Args:
      content: Response body.
      on_success_body: Whether the body arrived with a success status. Generic
        CAPTCHA widgets and Cloudflare's ambient beacon are inconclusive on a
        successful page, so only structural interstitial evidence is accepted.
      cloudflare: Cloudflare challenge-only markup markers.
      cloudflare_ambient: Cloudflare markers also present on ordinary proxied pages.
      cloudflare_titles: Cloudflare interstitial page titles.
      puzzle_widget: Generic CAPTCHA widget markers.

    Returns:
      error_type: The proven challenge type, or ``None``.

    """
    text = _text(content)
    title = _page_title(text)
    if title in cloudflare_titles:
        return CloudflareChallengeError
    if _has_markup_marker(text, cloudflare):
        return CloudflareChallengeError
    if on_success_body:
        return None
    if any(marker in text for marker in (*cloudflare, *cloudflare_ambient)):
        return CloudflareChallengeError
    if any(marker in text for marker in puzzle_widget):
        return PuzzleChallengeError
    return None


def classify_http_error(
    url: str,
    status: int,
    headers: dict[str, str],
    body: bytes,
    *,
    mitigation_statuses: tuple[int, ...] = (403, 429, 503),
) -> FetchError:
    """Build the most specific error proven by an HTTP failure.

    Args:
      url: Requested URL.
      status: HTTP status code.
      headers: Response headers.
      body: Decompressed response body.
      mitigation_statuses: Statuses where a Cloudflare front proves mitigation.

    Returns:
      error: A challenge error when detected, otherwise ``FetchError``.

    """
    error_type = classify_challenge(body)
    if (
        error_type is None
        and status in mitigation_statuses
        and _is_cloudflare_front(headers)
    ):
        error_type = CloudflareChallengeError
    if error_type is None:
        return FetchError(url, status, headers, body)
    return error_type(url=url, status=status, headers=headers, body=body)


def _text(content: str | bytes) -> str:
    decoded = (
        content.decode("utf-8", "replace") if isinstance(content, bytes) else content
    )
    return decoded.lower()


def _page_title(text: str) -> str | None:
    match = _TITLE_RE.search(text)
    return match.group(1).strip() if match is not None else None


def _has_markup_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in tag for tag in _TAG_RE.findall(text) for marker in markers)


def _is_cloudflare_front(headers: dict[str, str]) -> bool:
    lower_headers = {key.lower(): value.lower() for key, value in headers.items()}
    return (
        "cloudflare" in lower_headers.get("server", "")
        or "cf-ray" in lower_headers
        or "cf-mitigated" in lower_headers
    )
