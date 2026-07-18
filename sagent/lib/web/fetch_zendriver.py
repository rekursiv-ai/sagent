#!/bin/sh
# ruff: noqa: EXE003, D300 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Real-browser fetch backend for ``sagent.lib.web`` (opt-in).

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
from urllib.parse import urlparse, urlsplit

import asyncio
import contextlib
import hashlib
import ipaddress
import logging
import os
import select
import socket
import socketserver
import threading
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


class _ProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self) -> None:
        self._pins: dict[str, str] = {}
        self._pins_lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), _ProxyHandler)

    def pin(self, hostname: str, ip: str) -> None:
        """Pin ``hostname`` to a caller-validated public IP."""
        _require_public_ip(hostname, ip)
        with self._pins_lock:
            self._pins[hostname.lower()] = ip

    def resolve(self, hostname: str) -> str:
        """Resolve to a public IP at connect time."""
        with self._pins_lock:
            pinned = self._pins.get(hostname.lower())
        if pinned is not None:
            return pinned
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        for info in infos:
            ip = str(info[4][0])
            try:
                _require_public_ip(hostname, ip)
            except ValueError:
                continue
            return ip
        raise ValueError(
            f"Refusing browser connection to non-public host {hostname!r}."
        )


class _ProxyHandler(socketserver.StreamRequestHandler):
    @override
    def handle(self) -> None:
        """Proxy one Chrome request while pinning DNS at connect time."""
        request_line = self.rfile.readline(65_537)
        if not request_line or len(request_line) > 65_536:
            return
        try:
            method, target, version = (
                request_line.decode("latin-1").rstrip().split(" ", 2)
            )
            headers = self._read_headers()
            if method.upper() == "CONNECT":
                self._tunnel(target)
            else:
                self._forward_http(method, target, version, headers)
        except (OSError, UnicodeError, ValueError):
            with contextlib.suppress(OSError):
                self.wfile.write(
                    b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n"
                )

    def _read_headers(self) -> list[bytes]:
        headers: list[bytes] = []
        total = 0
        while True:
            line = self.rfile.readline(65_537)
            total += len(line)
            if len(line) > 65_536 or total > 262_144:
                raise ValueError("proxy request headers too large")
            if line in (b"\r\n", b"\n", b""):
                return headers
            headers.append(line)

    def _tunnel(self, authority: str) -> None:
        host, port = _authority(authority, 443)
        server = cast("_ProxyServer", self.server)
        upstream = socket.create_connection((server.resolve(host), port), timeout=30)
        try:
            self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            self.wfile.flush()
            _relay(self.connection, upstream)
        finally:
            upstream.close()

    def _forward_http(
        self,
        method: str,
        target: str,
        version: str,
        headers: list[bytes],
    ) -> None:
        parsed = urlsplit(target)
        if parsed.scheme not in ("http", "https") or parsed.hostname is None:
            raise ValueError("proxy expected an absolute HTTP URL")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        upstream = socket.create_connection(
            (cast("_ProxyServer", self.server).resolve(parsed.hostname), port),
            timeout=30,
        )
        try:
            path = parsed.path or "/"
            if parsed.query:
                path = f"{path}?{parsed.query}"
            upstream.sendall(f"{method} {path} {version}\r\n".encode("latin-1"))
            for header in headers:
                if header.lower().startswith((b"proxy-connection:", b"connection:")):
                    continue
                upstream.sendall(header)
            upstream.sendall(b"Connection: close\r\n\r\n")
            while chunk := upstream.recv(65_536):
                self.connection.sendall(chunk)
        finally:
            upstream.close()


class _FilteringProxy:
    def __init__(self) -> None:
        self._server = _ProxyServer()
        self._thread = threading.Thread(
            target=lambda: self._server.serve_forever(poll_interval=0.01),
            name="loop-web-browser-proxy",
            daemon=True,
        )
        self._thread.start()

    @property
    def url(self) -> str:
        address = cast("tuple[str, int]", self._server.server_address)
        return f"http://{address[0]}:{address[1]}"

    def pin(self, hostname: str, ip: str) -> None:
        self._server.pin(hostname, ip)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


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
        return
    finally:
        client.close()


def _require_public_ip(hostname: str, raw_ip: str) -> None:
    ip = ipaddress.ip_address(raw_ip)
    if not ip.is_global:
        raise ValueError(f"Refusing browser connection to {hostname!r} at {ip}.")


def _authority(value: str, default_port: int) -> tuple[str, int]:
    parsed = urlparse(f"//{value}")
    if parsed.hostname is None:
        raise ValueError("proxy authority has no hostname")
    return parsed.hostname, parsed.port or default_port


def _relay(left: socket.socket, right: socket.socket) -> None:
    sockets = (left, right)
    while True:
        readable, _, _ = select.select(sockets, (), (), 30)
        if not readable:
            return
        for source in readable:
            data = source.recv(65_536)
            if not data:
                return
            (right if source is left else left).sendall(data)


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
    resolve_host: Callable[[str], str] | None = None,
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
      resolve_host: Optional resolver returning the pinned connect IP for the
        initial hostname. The browser proxy independently rejects non-public IPs.
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
            resolve_host=resolve_host,
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
    pool = _pool()
    pool.run(_open_instance(url, target, proxy_url=pool.proxy_url))


async def _open_instance(url: str, profile_dir: Path, *, proxy_url: str) -> None:
    """Open a headed browser, navigate to ``url``, and block until it is closed."""
    browser = await _launch_browser(
        profile_dir,
        headless=False,
        proxy_url=proxy_url,
    )
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
    proxy_url: str = "",
) -> zendriver.Browser:
    """Launch Chrome through the filtering proxy."""
    profile_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 -- one-shot setup.
    return await zendriver.start(
        headless=headless,
        user_data_dir=str(profile_dir),
        browser_args=(
            [f"--proxy-server={proxy_url}", "--proxy-bypass-list=<-loopback>"]
            if proxy_url
            else []
        ),
        sandbox=_sandbox(),
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
    resolve_host: Callable[[str], str] | None = None,
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
        pool = _pool()
        host = urlparse(url).hostname or ""
        if host and resolve_host is not None:
            pool.pin(host, resolve_host(host))
        browser = await pool.browser(egress, profile_dir, headless=headless)
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
    """

    def __init__(self, *, serve_control: bool = True) -> None:
        self._loop = asyncio.new_event_loop()
        self._proxy = _FilteringProxy()
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

    @property
    def proxy_url(self) -> str:
        return self._proxy.url

    def pin(self, hostname: str, ip: str) -> None:
        self._proxy.pin(hostname, ip)

    def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run a coroutine on the pool's loop from a sync caller; return its result."""
        future: Future[_T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def browser(
        self, egress: str, profile_dir: Path, *, headless: bool
    ) -> zendriver.Browser:
        """Return the warm browser for a key, launching one on first use.

        A stopped browser (its Chrome exited or crashed) is replaced, so a dead
        entry never wedges the pool.
        """
        key = (egress, str(profile_dir))
        async with self._launch_lock:
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
        self._proxy.close()
        for control in controls:
            control.close()

    def _ensure_control(self, profile_dir: Path) -> None:
        """Serve graceful cross-process release requests for ``profile_dir``."""
        if not self._serve_control:
            return
        key = str(profile_dir.resolve())
        with self._lock:
            if key in self._controls:
                return
            self._controls[key] = _PoolControlServer(profile_dir, shutdown_browsers)

    async def _launch(self, profile_dir: Path, *, headless: bool) -> zendriver.Browser:
        """Launch one Chrome under ``profile_dir`` on the pool's loop."""
        return await _launch_browser(
            profile_dir,
            headless=headless,
            proxy_url=self.proxy_url,
        )

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
            "profile -- the same profile the headless RequestParams(backend="
            '"zendriver") fetch uses. Use it to debug a fetch that errored: you '
            "see exactly what Chrome renders (a challenge, a login wall, a broken "
            "page), and any cookies you seat while there (e.g. by logging in) "
            "persist for later headless fetches. Close the window when done."
        ),
        epilog=(
            "Examples:\n"
            "  sh fetch_zendriver.py https://accounts.google.com/\n"
            "  sh fetch_zendriver.py https://scholar.google.com/\n"
            "  sh fetch_zendriver.py https://the-site-that-failed.example/\n"
            "  sh fetch_zendriver.py # opens blank; navigate by hand"
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
