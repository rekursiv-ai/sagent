"""Chrome request-header parity: the headers curl_cffi impersonation omits.

curl_cffi replays Chrome's TLS/HTTP-2 fingerprint and the THREE basic client
hints (``sec-ch-ua``, ``-mobile``, ``-platform``), but not the extended
low-entropy hints a real Chrome sends to every site, nor the Google-only
integrity headers it sends to Google properties. A request that carries a
Chrome User-Agent + TLS fingerprint yet omits these is provably not Chrome, a
bot signal no cookie or pacing change addresses.

Two layers, split by where a real Chrome sends them:

- :func:`chrome_client_hints` -- the extended UA client-hint set
  (``sec-ch-ua-arch`` etc.). Chrome sends these to EVERY site, so
  :mod:`sagent.lib.web.fetch` adds them on every curl request, matching the
  impersonated browser (desktop or Android). No extra requests.
- :func:`chrome_headers_for_google` -- the ``x-browser-*`` + ``x-client-data``
  headers Chrome sends ONLY to Google-owned hosts. A caller scraping a Google
  property (e.g. Scholar) layers these on; they must NOT go to arbitrary sites.
  ``x-browser-validation`` is the reverse-engineered integrity token:
  ``base64(sha1(platform_api_key + UA))``.

References:
    https://github.com/dsekz/chrome-x-browser-validation-header
        Reverse engineering of Chrome's ``x-browser-validation`` header.

"""

from __future__ import annotations

from typing import Literal

import base64
import hashlib


__all__ = [
    "ChromePlatform",
    "chrome_client_hints",
    "chrome_headers_for_google",
    "chrome_navigation_headers",
    "chrome_user_agent",
    "impersonate_version_platform",
]

type ChromePlatform = Literal["Windows", "Linux", "macOS", "Android"]

# The Chrome major version + platform each curl_cffi ``impersonate`` target
# presents (its UA + basic client hints), so the extended hints we add agree
# with what curl already put on the wire. Keep in step with the installed
# curl_cffi's DEFAULT_CHROME as it advances.
_IMPERSONATE: dict[
    str, tuple[int, ChromePlatform]
] = {  # config-globals: ignore -- static impersonate->version map.
    "chrome": (146, "macOS"),
    "chrome_android": (131, "Android"),
}


def impersonate_version_platform(impersonate: str) -> tuple[int, ChromePlatform]:
    """Resolve a curl_cffi impersonate target to its (major, platform).

    Falls back to the desktop-Chrome default for an unrecognized target, so an
    unknown value degrades to a coherent desktop identity rather than raising.
    """
    return _IMPERSONATE.get(impersonate, _IMPERSONATE["chrome"])


# Per-platform Google API keys hard-coded in chrome.dll, used as the prefix when
# hashing the User-Agent into x-browser-validation. From the reverse-engineering
# above; stable across Chrome builds to date.
_PLATFORM_API_KEY: dict[
    ChromePlatform, str
] = {  # config-globals: ignore -- Chrome's own hard-coded API keys.
    "Windows": "AIzaSyA2KlwBX3mkFo30om9LUFYQhpqLoa_BNhE",
    "Linux": "AIzaSyBqJZh-7pA44blAaAkH6490hUFOwX0KCYM",
    "macOS": "AIzaSyDr2UxVnv_U85AbhhY8XSHSIavUW0DC-sY",
    "Android": "AIzaSyA8Hr8czk2cWyLu-a_RSPVYUEHozUnu6bA",
}

# The UA OS token curl_cffi presents per platform (the string inside the UA's
# first parenthesis). Android carries a device model too, but a generic "K" is
# what a headless-of-recent Chrome reports.
_UA_OS: dict[
    ChromePlatform, str
] = {  # config-globals: ignore -- static per-platform UA OS tokens.
    "Windows": "Windows NT 10.0; Win64; x64",
    "Linux": "X11; Linux x86_64",
    "macOS": "Macintosh; Intel Mac OS X 10_15_7",
    "Android": "Linux; Android 10; K",
}


def chrome_user_agent(major: int, platform: ChromePlatform = "macOS") -> str:
    """Return the User-Agent string a Chrome of this major/platform presents."""
    if platform not in _UA_OS:
        raise ValueError(f"Unsupported platform {platform!r}.")
    tail = "Mobile Safari/537.36" if platform == "Android" else "Safari/537.36"
    return (
        f"Mozilla/5.0 ({_UA_OS[platform]}) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{major}.0.0.0 {tail}"
    )


def chrome_navigation_headers(
    *,
    major: int,
    platform: ChromePlatform = "macOS",
    method: str = "GET",
    content_type: str = "",
    origin: str = "",
) -> dict[str, str]:
    """Return the full Chrome header set for a request, in exact wire order.

    The complete core identity a real Chrome sends to any site -- the three basic
    client hints, User-Agent, Accept, Sec-Fetch-*, Priority, Accept-Encoding/
    Language -- matching the captured wire order of curl_cffi's impersonation
    (verified against real Chrome). Used by the stdlib transport, which has no
    impersonation and must reproduce the whole set by hand; the curl transport
    lets its impersonation emit these and adds only the Accept-CH extensions.

    Args:
      major: Chrome major version.
      platform: The impersonated OS (selects UA + ``sec-ch-ua-platform``).
      method: HTTP method; GET/HEAD get the navigation shape, others the
        fetch/XHR shape (``Accept: */*``, ``Origin``, cross-site Sec-Fetch).
      content_type: Request body content type, added for a non-GET/HEAD.
      origin: The request origin (``scheme://host``) for a non-GET/HEAD.

    Returns:
      headers: The ordered Chrome header map.

    """
    h: dict[str, str] = {
        "sec-ch-ua": (
            f'"Chromium";v="{major}", "Not-A.Brand";v="24", "Google Chrome";v="{major}"'
        ),
        "sec-ch-ua-mobile": "?1" if platform == "Android" else "?0",
        "sec-ch-ua-platform": f'"{platform}"',
    }
    if method in ("GET", "HEAD"):
        h["Upgrade-Insecure-Requests"] = "1"
        h["User-Agent"] = chrome_user_agent(major, platform)
        h["Accept"] = (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        )
        h["Sec-Fetch-Site"] = "none"
        h["Sec-Fetch-Mode"] = "navigate"
        h["Sec-Fetch-User"] = "?1"
        h["Sec-Fetch-Dest"] = "document"
    else:
        h["User-Agent"] = chrome_user_agent(major, platform)
        h["Accept"] = "*/*"
        if content_type:
            h["Content-Type"] = content_type
        if origin:
            h["Origin"] = origin
        h["Sec-Fetch-Site"] = "cross-site"
        h["Sec-Fetch-Mode"] = "cors"
        h["Sec-Fetch-Dest"] = "empty"
    h["Accept-Encoding"] = "gzip, deflate, br, zstd"
    h["Accept-Language"] = "en-US,en;q=0.9"
    h["Priority"] = "u=0, i"
    return h


def chrome_client_hints(
    *,
    major: int,
    platform: ChromePlatform = "macOS",
    full_version: str = "",
) -> dict[str, str]:
    """Return the extended UA client-hint headers a real Chrome sends everywhere.

    The three basic hints (``sec-ch-ua`` / ``-mobile`` / ``-platform``) are
    supplied by curl_cffi's impersonation; this adds the low-entropy extensions
    it omits, matching the impersonated browser. Mobile (Android) reports
    ``sec-ch-ua-mobile: ?1`` and empty ``-arch``/``-bitness`` (a phone has no
    desktop CPU hints), which is what a real mobile Chrome sends.

    Args:
      major: Chrome major version (must equal the impersonated UA's version, or
        the hints and the UA disagree -- itself a tell).
      platform: The impersonated OS; selects ``sec-ch-ua-platform`` and the
        mobile/arch shape.
      full_version: Full Chrome version (e.g. ``"146.0.7379.0"``) for
        ``sec-ch-ua-full-version-list``; defaults to ``"<major>.0.0.0"``.

    Returns:
      headers: The extended client-hint header map to merge onto a request.

    """
    if platform not in _UA_OS:
        raise ValueError(f"Unsupported platform {platform!r}.")
    version = full_version or f"{major}.0.0.0"
    mobile = platform == "Android"
    # The full-version-list brand token must match curl_cffi's own sec-ch-ua
    # brand exactly ("Not-A.Brand";v="24" for chrome146) -- a real Chrome's
    # sec-ch-ua and -full-version-list always carry the same brand string.
    full_list = (
        f'"Chromium";v="{version}", "Not-A.Brand";v="24.0.0.0", '
        f'"Google Chrome";v="{version}"'
    )
    # Key order mirrors a real Chrome's opted-in wire order exactly:
    # arch, platform-version, model, bitness, wow64, full-version-list.
    return {
        "sec-ch-ua-arch": '""' if mobile else '"x86"',
        "sec-ch-ua-platform-version": '""',
        "sec-ch-ua-model": '"K"' if mobile else '""',
        "sec-ch-ua-bitness": '""' if mobile else '"64"',
        "sec-ch-ua-wow64": "?0",
        "sec-ch-ua-full-version-list": full_list,
    }


def chrome_headers_for_google(
    *,
    major: int,
    platform: ChromePlatform = "macOS",
    x_client_data: str = "CInbygE=",
) -> dict[str, str]:
    """Return the Google-ONLY Chrome headers (``x-browser-*`` + ``x-client-data``).

    These are the integrity/telemetry headers a real Chrome sends ONLY to
    Google-owned hosts (google.com, scholar.google.com, gstatic, googleapis) --
    send them to a Google property, never to an arbitrary site. The extended
    client hints are NOT included here: :mod:`sagent.lib.web.fetch` adds those to
    every origin that opts in via ``Accept-CH`` (Google does), matching a real
    browser, so a caller need only add the Google-only set.

    Args:
      major: Chrome major version; must match the impersonated UA.
      platform: The impersonated OS; selects the validation API key + UA.
      x_client_data: The opaque ``x-client-data`` install token (a real
        low-entropy value by default, shared across many installs).

    Returns:
      headers: The Google-only header map.

    """
    return {
        "x-browser-channel": "stable",
        "x-browser-year": "2026",
        "x-browser-validation": _validation(
            chrome_user_agent(major, platform), platform
        ),
        "x-client-data": x_client_data,
    }


def _validation(user_agent: str, platform: ChromePlatform) -> str:
    """Compute ``x-browser-validation``: ``base64(sha1(api_key + UA))``."""
    data = (_PLATFORM_API_KEY[platform] + user_agent).encode()
    return base64.b64encode(hashlib.sha1(data).digest()).decode()  # noqa: S324
