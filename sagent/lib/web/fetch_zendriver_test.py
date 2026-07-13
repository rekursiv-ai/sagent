"""Tests for ``sagent.lib.web.fetch_zendriver`` (zendriver headless fetch backend).

Hermetic: a fake async browser stands in for zendriver, so the transport logic
(cookie-domain filtering, challenge detection, redirect callback, pool reuse)
is exercised with no Chrome and no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import asyncio

import pytest

from sagent.lib.web import fetch_zendriver as fz_mod
from sagent.lib.web.errors import CloudflareChallengeError, PuzzleChallengeError
from sagent.lib.web.fetch_zendriver import BrowserResult, _BrowserPool, _navigate


# A fake profile dir; the browser is mocked in every test, so it is never
# touched on disk.
_PROFILE = Path("test-profile")


@dataclass(slots=True, kw_only=True)
class _FakeCookie:
    name: str
    value: str
    domain: str


class _FakeCookieJar:
    def __init__(self, cookies: list[_FakeCookie]) -> None:
        self._cookies = cookies

    async def get_all(self) -> list[_FakeCookie]:
        return self._cookies


class _FakeTab:
    def __init__(self, *, content: str, href: str) -> None:
        self._content = content
        self._href = href
        self.closed = False

    async def wait_for_ready_state(
        self,
        until: str = "interactive",
        timeout: int = 10,  # noqa: ASYNC109 -- mirrors zendriver's Tab API.
    ) -> bool:
        del until, timeout
        return True

    async def evaluate(self, expr: str) -> str:
        del expr
        return self._href

    async def get_content(self) -> str:
        return self._content

    async def close(self) -> None:
        self.closed = True


class _FakeBrowser:
    """An async stand-in for ``zendriver.Browser`` with scripted content."""

    def __init__(
        self,
        *,
        content: str = "<html>ok</html>",
        href: str = "",
        cookies: list[_FakeCookie] | None = None,
    ) -> None:
        self._content = content
        self._href = href
        self.cookies = _FakeCookieJar(cookies or [])
        self.stopped = False
        self.gets: list[str] = []
        self.stop_calls = 0
        self.last_tab: _FakeTab | None = None

    async def get(self, url: str, new_tab: bool = False) -> _FakeTab:
        del new_tab
        self.gets.append(url)
        self.last_tab = _FakeTab(content=self._content, href=self._href or url)
        return self.last_tab

    async def stop(self) -> None:
        self.stop_calls += 1
        self.stopped = True


class _StubPool:
    """A pool whose ``browser`` always yields one preset fake browser."""

    def __init__(self, browser: _FakeBrowser) -> None:
        self._browser = browser

    async def browser(
        self, egress: str, profile_dir: Path, *, headless: bool
    ) -> _FakeBrowser:
        del egress, profile_dir, headless
        return self._browser


def _patch_pool(monkeypatch: pytest.MonkeyPatch, browser: _FakeBrowser) -> None:
    monkeypatch.setattr(fz_mod, "_pool", lambda: _StubPool(browser))


# -- _navigate: body + cookie harvest ----------------------------------------


def test_navigate_returns_body_and_domain_cookies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    browser = _FakeBrowser(
        content="<html>results</html>",
        cookies=[
            _FakeCookie(name="SID", value="abc", domain=".scholar.google.com"),
            _FakeCookie(name="OTHER", value="zzz", domain="example.com"),
        ],
    )
    _patch_pool(monkeypatch, browser)
    result = asyncio.run(
        _navigate(
            "https://scholar.google.com/scholar?q=x",
            profile_dir=_PROFILE,
            egress="1.2.3.4",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        )
    )
    assert isinstance(result, BrowserResult)
    assert result.body == b"<html>results</html>"
    # Only the domain-matching cookie is harvested; the foreign one is dropped.
    assert result.cookies == {"SID": "abc"}
    # The per-fetch tab is closed after harvest -- the memory-teardown contract
    # (Chrome process stays warm; the scraped page's tab does not).
    assert browser.last_tab is not None
    assert browser.last_tab.closed is True


def test_navigate_closes_tab_even_on_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A challenge raises, but the tab must STILL close (finally), so an errored
    # fetch never leaks a resident page.
    browser = _FakeBrowser(content="<html><title>Just a moment...</title></html>")
    _patch_pool(monkeypatch, browser)
    with pytest.raises(CloudflareChallengeError):
        asyncio.run(
            _navigate(
                "https://walled.example/",
                profile_dir=_PROFILE,
                egress="e",
                timeout_sec=5.0,
                headless=True,
                on_redirect=None,
            )
        )
    assert browser.last_tab is not None
    assert browser.last_tab.closed is True


def test_navigate_matches_exact_host_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    browser = _FakeBrowser(
        cookies=[_FakeCookie(name="H", value="1", domain="example.com")]
    )
    _patch_pool(monkeypatch, browser)
    result = asyncio.run(
        _navigate(
            "https://example.com/",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        )
    )
    assert result.cookies == {"H": "1"}


# -- _navigate: challenge detection ------------------------------------------


def test_navigate_raises_on_cloudflare_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    browser = _FakeBrowser(content="<html><title>Just a moment...</title></html>")
    _patch_pool(monkeypatch, browser)
    with pytest.raises(CloudflareChallengeError):
        asyncio.run(
            _navigate(
                "https://walled.example/",
                profile_dir=_PROFILE,
                egress="e",
                timeout_sec=5.0,
                headless=True,
                on_redirect=None,
            )
        )


def test_navigate_raises_on_scholar_captcha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    browser = _FakeBrowser(content='<form id="gs_captcha_f"></form>')
    _patch_pool(monkeypatch, browser)
    with pytest.raises(PuzzleChallengeError):
        asyncio.run(
            _navigate(
                "https://scholar.google.com/scholar?q=x",
                profile_dir=_PROFILE,
                egress="e",
                timeout_sec=5.0,
                headless=True,
                on_redirect=None,
            )
        )


# -- _navigate: redirect callback --------------------------------------------


def test_navigate_fires_on_redirect_when_final_url_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    browser = _FakeBrowser(href="https://example.com/landing")
    _patch_pool(monkeypatch, browser)
    seen: list[str] = []
    asyncio.run(
        _navigate(
            "https://example.com/start",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=seen.append,
        )
    )
    assert seen == ["https://example.com/landing"]


def test_navigate_no_redirect_when_url_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    url = "https://example.com/x"
    browser = _FakeBrowser(href=url)
    _patch_pool(monkeypatch, browser)
    seen: list[str] = []
    asyncio.run(
        _navigate(
            url,
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=seen.append,
        )
    )
    assert seen == []


# -- _BrowserPool: reuse + replacement ---------------------------------------


def test_pool_reuses_browser_per_key(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[_FakeBrowser] = []

    async def fake_launch(
        self: _BrowserPool, profile_dir: Path, *, headless: bool
    ) -> _FakeBrowser:
        del self, profile_dir, headless
        b = _FakeBrowser()
        launched.append(b)
        return b

    monkeypatch.setattr(_BrowserPool, "_launch", fake_launch)
    pool = _BrowserPool()
    try:

        async def go() -> bool:
            a = await pool.browser("e", _PROFILE, headless=True)
            b = await pool.browser("e", _PROFILE, headless=True)
            return a is b

        assert pool.run(go())
        assert len(launched) == 1
    finally:
        pool.shutdown()


def test_pool_relaunches_stopped_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[_FakeBrowser] = []

    async def fake_launch(
        self: _BrowserPool, profile_dir: Path, *, headless: bool
    ) -> _FakeBrowser:
        del self, profile_dir, headless
        b = _FakeBrowser()
        launched.append(b)
        return b

    monkeypatch.setattr(_BrowserPool, "_launch", fake_launch)
    pool = _BrowserPool()
    try:

        async def go() -> None:
            first = await pool.browser("e", _PROFILE, headless=True)
            cast("Any", first).stopped = True  # simulate Chrome exit
            second = await pool.browser("e", _PROFILE, headless=True)
            assert second is not first

        pool.run(go())
        assert len(launched) == 2
    finally:
        pool.shutdown()


def test_pool_keys_separate_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[_FakeBrowser] = []

    async def fake_launch(
        self: _BrowserPool, profile_dir: Path, *, headless: bool
    ) -> _FakeBrowser:
        del self, profile_dir, headless
        b = _FakeBrowser()
        launched.append(b)
        return b

    monkeypatch.setattr(_BrowserPool, "_launch", fake_launch)
    pool = _BrowserPool()
    try:

        async def go() -> None:
            await pool.browser("egress-a", _PROFILE, headless=True)
            await pool.browser("egress-b", _PROFILE, headless=True)

        pool.run(go())
        assert len(launched) == 2  # distinct egress -> distinct browser
    finally:
        pool.shutdown()


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
