#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync \
  python3 -m sagent.lib.web.fetch.zendriver "$@"
Real-browser fetch backend for ``sagent.lib.web.fetch`` (opt-in).

Drives a headless Chrome via ``zendriver`` so pages gated behind a run-the-JS
challenge (Cloudflare, Google Scholar CAPTCHA) load where the curl/stdlib
backends get a wall. Select it per call with
``RequestParams(transport="zendriver")``; the page runs under a persistent Chrome
profile, so cookies you seat (e.g. by logging in) carry across fetches.

Run this module as ``loop-web-fetch-zendriver --url URL`` to open that URL in a
HEADED Chrome on the same profile -- to debug a fetch that errored, or to seat a
login.
'''
# fmt: on

from __future__ import annotations

from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, cast, override
from urllib.parse import urlparse

import asyncio
import hashlib
import logging
import os
import socket
import socketserver
import threading
import time
import warnings

from sagent.lib.userdirs import data_dir


if TYPE_CHECKING:
    import zendriver
else:
    from wrapt import lazy_import

    # Deferred: importing zendriver pulls a large CDP-binding tree (~200ms,
    # measured) and is paid only when a browser fetch actually runs, never at
    # ``sagent.lib.web`` import.
    zendriver = lazy_import("zendriver")


__all__ = [
    "BrowserResult",
    "default_profile_dir",
    "fetch_zendriver",
    "open_instance",
    "shutdown_browsers",
]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class _PoolControlServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True

    def __init__(self, profile_dir: Path, release: Callable[[], None]) -> None:
        self.release = release
        super().__init__(_control_address(profile_dir), _PoolControlHandler)
        self._thread = threading.Thread(
            target=lambda: self.serve_forever(poll_interval=0.01),
            name="loop-web-browser-control",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self.shutdown()
        self.server_close()
        self._thread.join()


class _PoolControlHandler(socketserver.StreamRequestHandler):
    @override
    def handle(self) -> None:
        if self.rfile.readline(64) != b"release\n":
            return
        cast("_PoolControlServer", self.server).release()


def _control_address(profile_dir: Path) -> str:
    """Return the abstract Unix-socket address coordinating one profile."""
    digest = hashlib.sha256(str(profile_dir.resolve()).encode()).hexdigest()[:24]
    return f"\0loop-zendriver-{digest}"


def _request_pool_release(profile_dir: Path) -> None:
    """Ask another process's browser pool to release ``profile_dir``."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(10)
    try:
        client.connect(_control_address(profile_dir))
        client.sendall(b"release\n")
        # EOF is the acknowledgement: the handler closes only after the release
        # callback returns, including graceful browser and loop shutdown.
        if client.recv(64) != b"":
            raise RuntimeError("Zendriver browser pool returned an invalid response.")
    except (ConnectionRefusedError, FileNotFoundError):
        pass
    finally:
        client.close()
    # Another process can leave a stale control listener that acknowledges this
    # profile without owning its Chrome. Verify and close the actual owner.
    _close_orphan_browser(profile_dir)


def _close_orphan_browser(profile_dir: Path) -> None:
    """Close a live Chrome whose owning pool no longer serves control."""
    port = _devtools_port(profile_dir)
    if port is None:
        return
    try:
        connection = socket.create_connection(("127.0.0.1", port), timeout=0.2)
    except OSError:
        return
    connection.close()
    asyncio.run(_close_browser_on_port(port))
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            connection = socket.create_connection(("127.0.0.1", port), timeout=0.1)
        except OSError:
            return
        connection.close()
        time.sleep(0.05)
    raise RuntimeError(f"Chrome on DevTools port {port} did not close.")


def _devtools_port(profile_dir: Path, *, proc_root: Path = Path("/proc")) -> int | None:
    """Read the verified profile owner's active DevTools port."""
    try:
        owner = (profile_dir / "SingletonLock").readlink()
        pid = int(str(owner).rsplit("-", 1)[1])
        command = (
            (proc_root / str(pid) / "cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode()
        )
    except (FileNotFoundError, IndexError, OSError, UnicodeError, ValueError):
        return None
    if _command_flag(command, "--user-data-dir=") != str(profile_dir.resolve()):
        return None
    try:
        port_text = _command_flag(command, "--remote-debugging-port=") or ""
        port = int(port_text.split(maxsplit=1)[0])
        if port:
            return port
        return int((profile_dir / "DevToolsActivePort").read_text().splitlines()[0])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def _command_flag(command: str, marker: str) -> str | None:
    """Extract a Chrome flag value from NUL- or space-flattened proc args."""
    _, found, suffix = command.partition(marker)
    if not found:
        return None
    return suffix.split(" --", 1)[0].strip()


async def _close_browser_on_port(port: int) -> None:
    browser = await zendriver.start(host="127.0.0.1", port=port)
    await browser.stop()


def _sandbox() -> bool:
    """Whether to run Chrome sandboxed (yes, unless we are root).

    Chrome's setuid sandbox refuses to start as root, so a root context (CI,
    containers) must pass ``--no-sandbox``. A normal desktop user keeps the
    sandbox -- disabling it there needlessly weakens security AND makes Chrome
    show a persistent "unsupported command-line flag: --no-sandbox" banner.
    """
    return os.geteuid() != 0


def default_profile_dir() -> Path:
    """The fresh dedicated Chrome ``user_data_dir`` the browser backend uses.

    A per-user directory distinct from the live ``~/.config/google-chrome``
    (which Chrome singleton-locks while running), seeded once by the
    ``loop-web-fetch-zendriver`` entrypoint and reused headless thereafter.
    """
    return data_dir("loop") / "lib" / "web" / "fetch-zendriver"


class BrowserResult(NamedTuple):
    """What a browser fetch yields: the rendered page and the cookies it holds.

    Attributes:
      body: The rendered ``document`` HTML, UTF-8 encoded.
      cookies: Cookies the browser holds for the fetched URL's domain
        (``name -> value``), for the caller to persist and thread onward.

    """

    body: bytes
    cookies: dict[str, str]


def fetch_zendriver(
    url: str,
    *,
    profile_dir: Path,
    egress: str,
    timeout_sec: float = 30.0,
    headless: bool = True,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    on_redirect: Callable[[str], None] | None = None,
) -> BrowserResult:
    """Fetch ``url`` in a pooled headless Chrome; return its body and cookies.

    Navigates a warm browser (one per ``(egress, profile_dir)``) to ``url``,
    waits for the load to complete, then returns the rendered HTML and the
    cookies the browser acquired for the URL's domain. Page-state validation
    belongs to the provider consuming the rendered response.

    Args:
      url: Fully-qualified URL to navigate to.
      profile_dir: Chrome ``user_data_dir`` supplying the logged-in identity.
      egress: Public egress IP the pooled browser is keyed to (a rotation keys
        a fresh browser).
      timeout_sec: Overall budget for navigation + load, in seconds.
      headless: Run Chrome headless (the default); ``False`` opens a window.
      headers: Extra headers applied to this tab before navigation.
      cookies: Cookies seeded for ``url`` before navigation.
      on_redirect: Called with the final URL when navigation lands somewhere
        other than ``url`` (a redirect); observational.

    Returns:
      result: The rendered body and the browser's cookies for the URL's domain.

    """
    return _pool().run(
        _navigate(
            url,
            profile_dir=profile_dir,
            egress=egress,
            timeout_sec=timeout_sec,
            headless=headless,
            headers=headers,
            cookies=cookies,
            on_redirect=on_redirect,
        )
    )


def shutdown_browsers() -> None:
    """Close every pooled browser and stop the pool's loop thread.

    Idempotent, and a NO-OP when no pool has been created -- it must never
    construct one just to tear it down (that would spin up a Chrome-driving loop
    thread only to stop it, and on an egress rotation before any browser fetch it
    would poison the not-yet-used singleton). Only an existing pool is shut down;
    the singleton is then cleared so the next browser fetch builds a fresh pool.
    """
    global _pool_singleton  # noqa: PLW0603 -- reset the shared pool after teardown.
    with _pool_lock:
        pool = _pool_singleton
        _pool_singleton = None
    if pool is not None:
        pool.shutdown()


def open_instance(url: str, *, profile_dir: Path | None = None) -> None:
    """Open a HEADED Chrome on the profile dir at ``url``; block until closed.

    Launches a visible Chrome under ``profile_dir`` (the fresh dedicated dir by
    default), navigates to ``url``, and blocks until the user closes the window.
    Use it to eyeball a URL the headless :func:`fetch_zendriver` backend failed on
    -- you see exactly what Chrome renders (a challenge, a login wall, a broken
    page) under the SAME profile the backend uses, and any cookies you seat while
    there (e.g. by logging in) persist for later headless fetches. Runs OUTSIDE
    the pool (a one-shot headed browser owned by this call).

    Args:
      url: The page to open -- typically the URL whose headless fetch you are
        debugging.
      profile_dir: Chrome ``user_data_dir`` to open; defaults to
        :func:`default_profile_dir`.

    """
    target = default_profile_dir() if profile_dir is None else profile_dir
    _request_pool_release(target)
    _pool().run(_open_instance(url, target))


async def _open_instance(url: str, profile_dir: Path) -> None:
    """Open a headed browser, navigate to ``url``, and block until it is closed."""
    browser = await _launch_browser(profile_dir, headless=False)
    try:
        await _navigate_tab(browser, url)
        # Block until the user closes the window (Chrome exits, so the browser reports
        # stopped). Polled, not event-driven: the window-closed signal is Chrome's
        # process exit, which zendriver exposes only as the polled ``stopped`` flag.
        while not browser.stopped:  # noqa: ASYNC110 -- no event source; poll the flag.
            await asyncio.sleep(0.5)
    except BaseException:
        await browser.stop()
        raise


async def _launch_browser(
    profile_dir: Path,
    *,
    headless: bool,
) -> zendriver.Browser:
    """Launch vanilla Chrome on the persistent profile."""
    profile_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 -- one-shot setup.
    return await zendriver.start(
        zendriver.Config(
            headless=headless,
            user_data_dir=str(profile_dir),
            sandbox=_sandbox(),
            browser_connection_timeout=1.0,
        )
    )


async def _navigate_tab(
    browser: zendriver.Browser,
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> zendriver.Tab:
    """Open a blank tab, apply headers, then navigate normally."""
    tab = await browser.get("about:blank", new_tab=True)
    try:
        if headers:
            await tab.send(zendriver.cdp.network.enable())
            await tab.send(
                zendriver.cdp.network.set_extra_http_headers(
                    zendriver.cdp.network.Headers(headers)
                )
            )
        await tab.get(url)
        await tab.wait_for_ready_state("complete")
    except BaseException:
        await tab.close()
        raise
    return tab


async def _navigate(
    url: str,
    *,
    profile_dir: Path,
    egress: str,
    timeout_sec: float,
    headless: bool,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    on_redirect: Callable[[str], None] | None = None,
) -> BrowserResult:
    """Drive a pooled browser to ``url`` in a fresh tab; harvest body + cookies.

    Each fetch runs in its OWN tab that is CLOSED when the fetch returns. The
    Chrome process stays warm in the pool (fast reuse), but the tab -- which
    holds the scraped page's DOM, JS heap, and images -- is the unit of memory
    teardown, so a sequence of fetches does not accumulate resident pages. A
    per-fetch tab also isolates concurrent fetches sharing the one browser.

    Readiness is Chrome's real load signal (``document.readyState ==
    "complete"``), not a fixed sleep, bounded by ``timeout_sec``. The transport
    returns what Chrome rendered without assigning provider semantics to it.
    """
    async with asyncio.timeout(timeout_sec):
        browser = await _pool().browser(egress, profile_dir, headless=headless)
        if cookies:
            await browser.cookies.set_all(
                [
                    zendriver.cdp.network.CookieParam(
                        name=name,
                        value=value,
                        url=url,
                    )
                    for name, value in cookies.items()
                ]
            )
        tab = await _navigate_tab(browser, url, headers=headers)
        try:
            body = await tab.get_content()
            final_url = cast("str", await tab.evaluate("document.location.href")) or url
            if on_redirect is not None and final_url != url:
                on_redirect(final_url)
            # Cookies are browser-wide (shared jar), so harvest before closing the
            # tab; the closed tab's cookies persist in the profile regardless.
            cookies = await _domain_cookies(browser, url)
        finally:
            await tab.close()
    return BrowserResult(body=body.encode(), cookies=cookies)


async def _domain_cookies(browser: zendriver.Browser, url: str) -> dict[str, str]:
    """Return the browser's cookies whose domain matches ``url``'s host."""
    host = urlparse(url).hostname or ""
    jar: dict[str, str] = {}
    for cookie in await browser.cookies.get_all():
        domain = (cookie.domain or "").lstrip(".")
        if domain and (host == domain or host.endswith(f".{domain}")):
            jar[cookie.name] = cookie.value or ""
    return jar


# The single pooled browser manager, built once on first browser fetch. A
# deliberate module singleton: it owns a live loop thread and open Chrome
# processes -- shared runtime resources, not a tunable.
# config-globals: ignore -- live pool of open browsers + its loop thread.
_pool_singleton: _BrowserPool | None = None
_pool_lock = threading.Lock()  # config-globals: ignore -- guards the singleton.


def _pool() -> _BrowserPool:
    """Return the process-wide browser pool, creating it once."""
    global _pool_singleton  # noqa: PLW0603 -- memoize the shared pool.
    with _pool_lock:
        if _pool_singleton is None:
            _pool_singleton = _BrowserPool()
        return _pool_singleton


class _BrowserPool:
    """Pooled headless browsers over one persistent event loop on a daemon thread.

    zendriver browsers bind to the loop running their coroutines, so a stable
    loop is a hard requirement for reuse across sync calls. This pool owns that
    loop on a background thread and dispatches every browser coroutine to it via
    :meth:`run`, keeping one warm :class:`zendriver.Browser` per
    ``(egress, profile_dir)`` key and rejecting incompatible launch modes.

    The hot spare justifies this lifecycle machinery: eight matched
    ``https://example.com/`` requests measured a 0.130-second median with one
    reused browser versus 4.196 seconds when launching Chrome per request, a
    32.28x speedup.
    """

    def __init__(self, *, serve_control: bool = True) -> None:
        self._loop = asyncio.new_event_loop()
        self._serve_control = serve_control
        self._controls: dict[str, _PoolControlServer] = {}
        self._browsers: dict[tuple[str, str], tuple[bool, zendriver.Browser]] = {}
        self._launch_lock = asyncio.Lock()
        self._lock = threading.Lock()
        # Started LAST, so every field _run_loop touches exists before it runs.
        self._thread = threading.Thread(
            target=self._run_loop, name="loop-web-browser", daemon=True
        )
        self._thread.start()

    def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run a coroutine on the pool's loop from a sync caller; return its result."""
        future: Future[_T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def browser(
        self,
        egress: str,
        profile_dir: Path,
        *,
        headless: bool,
    ) -> zendriver.Browser:
        """Return the warm browser for one egress and profile."""
        key = (egress, str(profile_dir))
        async with self._launch_lock:
            control_key = _control_address(profile_dir)
            with self._lock:
                owns_profile = control_key in self._controls
            if self._serve_control and not owns_profile:
                await asyncio.to_thread(_request_pool_release, profile_dir)
            self._ensure_control(profile_dir)
            with self._lock:
                existing = self._browsers.get(key)
            if existing is not None and not existing[1].stopped:
                if existing[0] != headless:
                    raise RuntimeError(
                        "Cannot change Zendriver launch mode for a live profile."
                    )
                return existing[1]
            launched = await self._launch(profile_dir, headless=headless)
            with self._lock:
                self._browsers[key] = (headless, launched)
            return launched

    def shutdown(self) -> None:
        """Close every pooled browser and stop the loop thread (idempotent)."""
        if self._loop.is_closed():
            return
        with self._lock:
            browsers = [browser for _, browser in self._browsers.values()]
            controls = list(self._controls.values())
            self._browsers.clear()
            self._controls.clear()
        for browser in browsers:
            try:
                self.run(browser.stop())
            except Exception:  # noqa: BLE001 -- teardown must not raise.
                logger.debug("browser stop failed during shutdown", exc_info=True)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()
        for control in controls:
            control.close()

    def _ensure_control(self, profile_dir: Path) -> None:
        """Serve graceful cross-process release requests for ``profile_dir``."""
        if not self._serve_control:
            return
        key = _control_address(profile_dir)
        with self._lock:
            if key in self._controls:
                return
            self._controls[key] = _PoolControlServer(profile_dir, shutdown_browsers)

    async def _launch(
        self,
        profile_dir: Path,
        *,
        headless: bool,
    ) -> zendriver.Browser:
        """Launch one Chrome under ``profile_dir`` on the pool's loop."""
        return await _launch_browser(profile_dir, headless=headless)

    def _run_loop(self) -> None:
        """Run the pool's loop until :meth:`shutdown`, muting zendriver's warnings.

        zendriver's CDP dispatch calls the deprecated ``asyncio.iscoroutinefunction``
        (connection.py) and leaves reader pipes for the GC to close, emitting a
        ``DeprecationWarning`` / ``ResourceWarning`` from inside the browser
        coroutine. Under a ``-W error`` caller (the repo's pytest turns every
        warning into an exception) that warning would raise INSIDE the awaited CDP
        handler and wedge the fetch. The pool loop thread runs only zendriver
        coroutines, so scoping the filter to this thread's ``run_forever`` mutes
        the upstream noise without hiding warnings from any caller's own code.
        """
        asyncio.set_event_loop(self._loop)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module=r"zendriver\..*",
            )
            warnings.filterwarnings("ignore", category=ResourceWarning)
            self._loop.run_forever()


def _main() -> int:
    """Open a URL in a headed Chrome on the backend's profile; return exit code."""
    import argparse  # noqa: PLC0415 -- CLI-only import, off the library path.

    parser = argparse.ArgumentParser(
        prog="loop-web-fetch-zendriver",
        description=(
            "Open a URL in a headed Chrome on the zendriver backend's dedicated "
            "profile -- the same profile the headless RequestParams(transport="
            '"zendriver") fetch uses. Use it to debug a fetch that errored: you '
            "see exactly what Chrome renders (a challenge, a login wall, a broken "
            "page), and any cookies you seat while there (e.g. by logging in) "
            "persist for later headless fetches. Close the window when done."
        ),
        epilog=(
            "Examples:\n"
            "  sh zendriver.py https://accounts.google.com/\n"
            "  sh zendriver.py https://scholar.google.com/\n"
            "  sh zendriver.py https://the-site-that-failed.example/\n"
            "  sh zendriver.py # opens blank; navigate by hand"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="about:blank",
        help="The URL to open (typically the one whose headless fetch failed). "
        "Omit to open a blank page and navigate by hand.",
    )
    args = parser.parse_args()
    print(  # noqa: T201 -- CLI user feedback.
        f"Opening {args.url} in Chrome on {default_profile_dir()} -- "
        "close the window when done."
    )
    open_instance(args.url)
    print("Window closed.")  # noqa: T201 -- CLI feedback.
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
# vim: ft=python
