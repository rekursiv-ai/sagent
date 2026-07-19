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
import zendriver

from sagent.lib.web import fetch_zendriver as fz_mod
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
        self.seeded: list[Any] = []

    async def get_all(self) -> list[_FakeCookie]:
        return self._cookies

    async def set_all(self, cookies: list[Any]) -> None:
        self.seeded = cookies


class _FakeTab:
    def __init__(self, *, content: str, href: str) -> None:
        self._content = content
        self._href = href
        self.closed = False
        self.navigations: list[str] = []
        self.commands: list[Any] = []

    async def send(self, command: Any) -> None:
        self.commands.append(command)

    async def get(self, url: str) -> _FakeTab:
        self.navigations.append(url)
        if not self._href:
            self._href = url
        return self

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
        self.last_tab = _FakeTab(content=self._content, href=self._href)
        return self.last_tab

    async def stop(self) -> None:
        self.stop_calls += 1
        self.stopped = True


class _StubPool:
    """A pool whose ``browser`` always yields one preset fake browser."""

    def __init__(self, browser: _FakeBrowser) -> None:
        self._browser = browser

    async def browser(
        self,
        egress: str,
        profile_dir: Path,
        *,
        headless: bool,
    ) -> _FakeBrowser:
        del egress, profile_dir, headless
        return self._browser


def _patch_pool(monkeypatch: pytest.MonkeyPatch, browser: _FakeBrowser) -> _StubPool:
    pool = _StubPool(browser)
    monkeypatch.setattr(fz_mod, "_pool", lambda: pool)
    return pool


def test_launch_browser_uses_vanilla_zendriver_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    browser = _FakeBrowser()
    browser_args: list[str] = []

    async def fake_start(config: Any) -> _FakeBrowser:
        browser_args.extend(config())
        return browser

    monkeypatch.setattr(zendriver, "start", fake_start)
    result = asyncio.run(fz_mod._launch_browser(tmp_path, headless=True))

    assert result is browser
    assert not any(argument.startswith("--proxy-server=") for argument in browser_args)
    assert not any(
        argument.startswith("--proxy-bypass-list=") for argument in browser_args
    )
    assert not any(
        argument.startswith("--host-resolver-rules=") for argument in browser_args
    )


# -- headed/backend navigation parity -----------------------------------------


def test_open_instance_uses_blank_tab_before_requested_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://scholar.google.com/scholar?q=x"
    browser = _FakeBrowser()
    browser.stopped = True

    async def fake_launch(profile_dir: Path, *, headless: bool) -> _FakeBrowser:
        assert profile_dir == _PROFILE
        assert headless is False
        return browser

    monkeypatch.setattr(fz_mod, "_launch_browser", fake_launch)
    asyncio.run(fz_mod._open_instance(url, _PROFILE))

    assert browser.gets == ["about:blank"]
    assert browser.last_tab is not None
    assert browser.last_tab.navigations == [url]


def test_open_instance_releases_remote_profile_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakePool:
        def run(self, coroutine: Any) -> None:
            events.append("launch")
            coroutine.close()

    def release(_profile: Path) -> None:
        events.append("release")

    monkeypatch.setattr(
        fz_mod,
        "_request_pool_release",
        release,
        raising=False,
    )
    monkeypatch.setattr(fz_mod, "_pool", FakePool)

    fz_mod.open_instance("about:blank", profile_dir=_PROFILE)

    assert events == ["release", "launch"]


def test_pool_control_releases_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    releases: list[bool] = []
    checked: list[Path] = []
    monkeypatch.setattr(fz_mod, "_close_orphan_browser", checked.append)
    server = fz_mod._PoolControlServer(_PROFILE, lambda: releases.append(True))
    try:
        fz_mod._request_pool_release(_PROFILE)
    finally:
        server.close()
    assert releases == [True]
    assert checked == [_PROFILE]


def test_pool_release_closes_orphan_when_control_is_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    closed: list[Path] = []
    monkeypatch.setattr(fz_mod, "_close_orphan_browser", closed.append)

    fz_mod._request_pool_release(tmp_path)

    assert closed == [tmp_path]


def test_devtools_port_falls_back_to_singleton_owner(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    process = tmp_path / "proc" / "123"
    profile.mkdir()
    process.mkdir(parents=True)
    (profile / "SingletonLock").symlink_to("tron-123")
    (process / "cmdline").write_bytes(
        b"/opt/google/chrome/chrome\0"
        + f"--user-data-dir={profile}\0".encode()
        + b"--remote-debugging-port=4567\0"
        + b"about:blank\0"
    )

    assert fz_mod._devtools_port(profile, proc_root=tmp_path / "proc") == 4567


def test_devtools_port_rejects_different_profile_with_shared_prefix(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    process = tmp_path / "proc" / "123"
    profile.mkdir()
    process.mkdir(parents=True)
    (profile / "SingletonLock").symlink_to("tron-123")
    (process / "cmdline").write_text(
        "/opt/google/chrome/chrome "
        f"--user-data-dir={profile}-other "
        "--remote-debugging-port=4567"
    )

    assert fz_mod._devtools_port(profile, proc_root=tmp_path / "proc") is None


def test_devtools_port_rejects_stale_marker_without_profile_owner(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "DevToolsActivePort").write_text("4567\n/devtools/browser/id\n")

    assert fz_mod._devtools_port(profile, proc_root=tmp_path / "proc") is None


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


def test_navigate_seeds_request_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser()
    _patch_pool(monkeypatch, browser)
    asyncio.run(
        _navigate(
            "https://google.com/search?q=x",
            profile_dir=_PROFILE,
            egress="1.2.3.4",
            timeout_sec=5.0,
            headless=True,
            headers={"X-Test": "yes"},
            cookies={"CONSENT": "YES+"},
        )
    )
    assert len(browser.cookies.seeded) == 1
    assert browser.cookies.seeded[0].name == "CONSENT"
    assert browser.cookies.seeded[0].value == "YES+"
    assert browser.last_tab is not None
    assert len(browser.last_tab.commands) == 2


def test_navigate_timeout_includes_browser_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SlowPool:
        async def browser(
            self,
            egress: str,
            profile_dir: Path,
            *,
            headless: bool,
        ) -> _FakeBrowser:
            del egress, profile_dir, headless
            await asyncio.Event().wait()
            raise AssertionError("Browser acquisition escaped the timeout.")

    slow_pool = _SlowPool()
    monkeypatch.setattr(fz_mod, "_pool", lambda: slow_pool)
    with pytest.raises(TimeoutError):
        asyncio.run(
            _navigate(
                "https://example.com/",
                profile_dir=_PROFILE,
                egress="e",
                timeout_sec=0.001,
                headless=True,
                on_redirect=None,
            )
        )


def test_navigate_uses_one_overall_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser()
    _patch_pool(monkeypatch, browser)

    async def reject_step_timeout(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Per-step timeout resets the request budget.")

    monkeypatch.setattr(asyncio, "wait_for", reject_step_timeout)
    asyncio.run(
        _navigate(
            "https://example.com/",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        )
    )


def test_navigate_opens_blank_tab_before_requested_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://scholar.google.com/scholar?q=x"
    browser = _FakeBrowser()
    _patch_pool(monkeypatch, browser)

    asyncio.run(
        _navigate(
            url,
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        )
    )

    assert browser.gets == ["about:blank"]
    assert browser.last_tab is not None
    assert browser.last_tab.navigations == [url]


def test_navigate_returns_rendered_page_without_semantic_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = '<html><div id="cf_chl_widget"></div></html>'
    browser = _FakeBrowser(content=body)
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

    assert result.body == body.encode()


def test_navigate_allows_embedded_captcha_on_rendered_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser(content='<html><div class="g-recaptcha"></div></html>')
    _patch_pool(monkeypatch, browser)

    result = asyncio.run(
        _navigate(
            "https://example.com/login",
            profile_dir=_PROFILE,
            egress="e",
            timeout_sec=5.0,
            headless=True,
            on_redirect=None,
        )
    )

    assert result.body == b'<html><div class="g-recaptcha"></div></html>'


def test_navigate_closes_tab_after_returning_rendered_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = _FakeBrowser(content="<html><title>Just a moment...</title></html>")
    _patch_pool(monkeypatch, browser)
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
    pool = _BrowserPool(serve_control=False)
    try:

        async def go() -> bool:
            a = await pool.browser("e", _PROFILE, headless=True)
            b = await pool.browser("e", _PROFILE, headless=True)
            return a is b

        assert pool.run(go())
        assert len(launched) == 1
    finally:
        pool.shutdown()


def test_pool_rejects_mode_change_for_live_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[_FakeBrowser] = []

    async def fake_launch(
        self: _BrowserPool, profile_dir: Path, *, headless: bool
    ) -> _FakeBrowser:
        del self, profile_dir, headless
        browser = _FakeBrowser()
        launched.append(browser)
        return browser

    monkeypatch.setattr(_BrowserPool, "_launch", fake_launch)
    pool = _BrowserPool(serve_control=False)
    try:

        async def go() -> None:
            await pool.browser("e", _PROFILE, headless=True)
            with pytest.raises(RuntimeError, match="launch mode"):
                await pool.browser("e", _PROFILE, headless=False)

        pool.run(go())
        assert len(launched) == 1
    finally:
        pool.shutdown()


def test_pool_serializes_concurrent_mode_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[_FakeBrowser] = []

    async def fake_launch(
        self: _BrowserPool, profile_dir: Path, *, headless: bool
    ) -> _FakeBrowser:
        del self, profile_dir, headless
        await asyncio.sleep(0)
        browser = _FakeBrowser()
        launched.append(browser)
        return browser

    monkeypatch.setattr(_BrowserPool, "_launch", fake_launch)
    pool = _BrowserPool(serve_control=False)
    try:

        async def go() -> tuple[object, object]:
            return await asyncio.gather(
                pool.browser("e", _PROFILE, headless=True),
                pool.browser("e", _PROFILE, headless=False),
                return_exceptions=True,
            )

        results = pool.run(go())
        assert len(launched) == 1
        assert sum(isinstance(result, RuntimeError) for result in results) == 1
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
    pool = _BrowserPool(serve_control=False)
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


def test_pool_shutdown_joins_thread_and_closes_loop() -> None:
    pool = _BrowserPool(serve_control=False)
    pool.shutdown()

    assert not pool._thread.is_alive()
    assert pool._loop.is_closed()


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
    pool = _BrowserPool(serve_control=False)
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
