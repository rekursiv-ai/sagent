"""Tests for sagent.lib.web.chrome.headers."""

from __future__ import annotations

from typing import cast

import base64
import hashlib

import pytest

from sagent.lib.web.chrome.headers import (
    ChromePlatform,
    chrome_client_hints,
    chrome_headers_for_google,
    impersonate_version_platform,
)


class TestChromeClientHints:
    def test_desktop_extended_hints(self) -> None:
        h = chrome_client_hints(major=146, platform="macOS")
        assert h["sec-ch-ua-arch"] == '"x86"'
        assert h["sec-ch-ua-bitness"] == '"64"'
        assert h["sec-ch-ua-model"] == '""'
        assert h["sec-ch-ua-wow64"] == "?0"
        assert 'v="146' in h["sec-ch-ua-full-version-list"]

    def test_android_hints_are_mobile_shaped(self) -> None:
        # A phone reports empty arch/bitness and a device model, not x86/64.
        h = chrome_client_hints(major=131, platform="Android")
        assert h["sec-ch-ua-arch"] == '""'
        assert h["sec-ch-ua-bitness"] == '""'
        assert h["sec-ch-ua-model"] == '"K"'

    def test_does_not_include_the_basic_hints(self) -> None:
        # sec-ch-ua / -mobile / -platform are curl_cffi's job (its impersonate
        # emits them); this helper adds ONLY the extended set curl omits.
        h = chrome_client_hints(major=146, platform="macOS")
        assert "sec-ch-ua" not in h
        assert "sec-ch-ua-mobile" not in h
        assert "sec-ch-ua-platform" not in h

    def test_full_version_flows_into_full_version_list(self) -> None:
        h = chrome_client_hints(
            major=146, platform="macOS", full_version="146.0.7379.0"
        )
        assert "146.0.7379.0" in h["sec-ch-ua-full-version-list"]

    def test_unsupported_platform_rejected(self) -> None:
        with pytest.raises(ValueError, match="platform"):
            chrome_client_hints(major=146, platform=cast(ChromePlatform, "BeOS"))


class TestGoogleChromeHeaders:
    def test_has_google_only_headers_not_client_hints(self) -> None:
        # chrome_headers_for_google is the Google-ONLY set; the extended client
        # hints are fetch's Accept-CH job, so they must NOT appear here.
        h = chrome_headers_for_google(major=146, platform="macOS")
        for name in (
            "x-browser-channel",
            "x-browser-year",
            "x-browser-validation",
            "x-client-data",
        ):
            assert name in h
        assert "sec-ch-ua-arch" not in h
        assert "sec-ch-ua-full-version-list" not in h

    def test_validation_is_base64_sha1_of_key_plus_ua(self) -> None:
        # x-browser-validation = base64(sha1(linux_api_key + linux UA)).
        h = chrome_headers_for_google(major=146, platform="Linux")
        ua = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
        )
        key = "AIzaSyBqJZh-7pA44blAaAkH6490hUFOwX0KCYM"
        want = base64.b64encode(hashlib.sha1((key + ua).encode()).digest()).decode()  # noqa: S324
        assert h["x-browser-validation"] == want

    def test_validation_differs_by_platform(self) -> None:
        mac = chrome_headers_for_google(major=146, platform="macOS")
        lin = chrome_headers_for_google(major=146, platform="Linux")
        assert mac["x-browser-validation"] != lin["x-browser-validation"]

    def test_android_validation_uses_mobile_ua(self) -> None:
        # Android's UA ends "Mobile Safari"; its validation token must reflect
        # that (differs from the desktop token even at the same major).
        droid = chrome_headers_for_google(major=131, platform="Android")
        desk = chrome_headers_for_google(major=131, platform="Linux")
        assert droid["x-browser-validation"] != desk["x-browser-validation"]


class TestImpersonateVersionPlatform:
    def test_chrome_desktop(self) -> None:
        assert impersonate_version_platform("chrome") == (146, "macOS")

    def test_chrome_android(self) -> None:
        assert impersonate_version_platform("chrome_android") == (131, "Android")

    def test_unknown_falls_back_to_desktop(self) -> None:
        assert impersonate_version_platform("safari") == (146, "macOS")


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
