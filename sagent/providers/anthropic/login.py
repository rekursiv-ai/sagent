"""A Claude subscription login: the one ``claude`` keeps, or one in a named file.

``claude`` reads the same store, so signing in here signs it in too.

Typical usage example:

  login = ClaudeLogin(resources=shared)        # ~/.claude, or the macOS Keychain.
  work = ClaudeLogin(Path("~/.claude/.credentials-work.json"), resources=shared)

  await login.sign_in()    # In the browser; over ssh, paste the code it shows.
  await login.token()      # An access token, renewed before it expires.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlparse

import asyncio
import base64
import getpass
import hashlib
import os
import platform
import secrets
import time
import webbrowser

from treekle import from_plain, json_dumps, json_loads

from sagent.lib.posix.child import Child
from sagent.model.exceptions import AuthError
from sagent.providers.lock import file_lock


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from sagent.providers.types import ProviderSharedResources


__all__ = [
    "ClaudeLogin",
]


class ClaudeLogin:
    """A login to a Claude subscription, kept fresh.

    Args:
      path: A credentials file; ``None`` uses the store ``claude`` keeps its
        own login in, which is the Keychain on macOS.
      resources: Where the HTTP client comes from.

    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        resources: ProviderSharedResources,
    ) -> None:
        if path is None and platform.system() == "Darwin":
            self._store: _File | _Keychain = _Keychain()
        else:
            self._store = _File(path or self.default_path())
        self._resources = resources
        self._access = ""
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    @classmethod
    def default_path(cls) -> Path:
        """Return the file ``claude`` keeps its login in, off macOS."""
        return Path("~/.claude/.credentials.json").expanduser()

    async def token(self) -> str:
        """Return an access token good for at least five more minutes.

        Returns:
          token: The stored token, or a renewed one when it was about to expire.

        """
        async with self._lock:
            if self._expiring():
                self._adopt(await self._store.read())
            if self._expiring():
                await self._renew()
        return self._access

    async def renew(self) -> None:
        """Replace the access token, as after the server rejects it."""
        async with self._lock:
            await self._renew()

    async def sign_in(
        self,
        *,
        headless: bool = False,
        show: Callable[[str], object] = print,
    ) -> None:
        """Sign in through the browser, or with a code pasted from it.

        Args:
          headless: Skip the browser and take a pasted code at once.
          show: Where to write the sign-in link for the user.

        Raises:
          AuthError: The pasted code belongs to another sign-in, or the server
            refused the code.

        """
        verifier = secrets.token_urlsafe(64)
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        state = secrets.token_urlsafe(32)
        code, redirect = "", ""
        if not headless:
            code, redirect = await _browser_code(challenge, state, show=show)
        if not code:
            code, redirect = await _pasted_code(challenge, state, show=show)
        grant = await self._grant(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect,
                "code_verifier": verifier,
                "state": state,
            },
        )
        plan = await self._plan(grant)
        async with self._lock, file_lock(self._store.lock):
            await self._save(grant, plan=plan)

    def _expiring(self) -> bool:
        # Five minutes early, so a token never expires mid-request.
        return time.time() > self._expires_at - 300.0

    def _adopt(self, stored: Mapping[str, object]) -> None:
        self._access = from_plain(stored.get("accessToken"), str, default="")
        self._expires_at = (
            from_plain(stored.get("expiresAt"), float, default=0.0) / 1000.0
        )

    # A refresh token works once: two processes refreshing together leave one
    # of them holding a revoked token.
    async def _renew(self) -> None:
        """Rotate the token, under a lock every process on this login shares."""
        async with file_lock(self._store.lock):
            before = self._access
            stored = await self._store.read()
            self._adopt(stored)
            if self._access != before and not self._expiring():
                return  # Another process rotated it while this one waited.
            grant = await self._grant(
                {
                    "grant_type": "refresh_token",
                    "refresh_token": from_plain(
                        stored.get("refreshToken"),
                        str,
                        default="",
                    ),
                    "scope": " ".join(_scopes()),
                },
            )
            await self._save(grant, plan={})

    async def _grant(self, request: Mapping[str, str]) -> dict[str, object]:
        response = await self._resources.http.post(
            "https://platform.claude.com/v1/oauth/token",
            json={**request, "client_id": _client_id()},
            timeout=15.0,
        )
        if response.status_code in (400, 401):
            raise AuthError(
                "the Claude login has expired or been revoked; sign in again",
            )
        response.raise_for_status()
        return json_loads(response.text, dict[str, object])

    async def _plan(self, grant: Mapping[str, object]) -> dict[str, object]:
        """Return the plan ``claude`` keeps beside the token; ``{}`` when unknown."""
        response = await self._resources.http.get(
            "https://api.anthropic.com/api/oauth/profile",
            headers={
                "authorization": f"Bearer {grant.get('access_token')}",
                "anthropic-beta": "oauth-2025-04-20",
            },
            timeout=10.0,
        )
        if response.status_code != 200:
            return {}
        profile = json_loads(response.text, dict[str, object])
        organization = from_plain(
            profile.get("organization"),
            dict[str, object],
            default={},
        )
        return {
            "subscriptionType": organization.get("organization_type"),
            "rateLimitTier": organization.get("rate_limit_tier"),
        }

    async def _save(
        self,
        grant: Mapping[str, object],
        *,
        plan: Mapping[str, object],
    ) -> None:
        """Store a grant, keeping every other field ``claude`` keeps beside it."""
        expires_in = from_plain(grant.get("expires_in"), float)
        stored = await self._store.read()
        stored |= {
            "accessToken": from_plain(grant.get("access_token"), str),
            "refreshToken": from_plain(grant.get("refresh_token"), str),
            "expiresAt": int((time.time() + expires_in) * 1000),
        }
        stored |= {key: value for key, value in plan.items() if value is not None}
        stored.setdefault("scopes", list(_scopes()))
        await self._store.write(stored)
        self._adopt(stored)


class _File:
    """Credentials in a JSON file, as ``claude`` writes them off macOS."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.lock = self.path.with_suffix(".lock")

    async def read(self) -> dict[str, object]:
        return _login_of(await asyncio.to_thread(_read_text, self.path))

    async def write(self, stored: Mapping[str, object]) -> None:
        document = _document(await asyncio.to_thread(_read_text, self.path))
        await asyncio.to_thread(
            _write_private,
            self.path,
            json_dumps(document | {"claudeAiOauth": stored}),
        )


class _Keychain:
    """Credentials in the macOS Keychain entry ``claude`` keeps them in."""

    def __init__(self) -> None:
        self.lock = Path("~/.claude/.credentials.lock").expanduser()

    async def read(self) -> dict[str, object]:
        return _login_of(await _security(*_find_entry(), "-w"))

    async def write(self, stored: Mapping[str, object]) -> None:
        document = _document(await _security(*_find_entry(), "-w"))
        secret = json_dumps(document | {"claudeAiOauth": stored}).encode().hex()
        # Read from stdin, not given as an argument: any local user can read a
        # running process's arguments, and this one carries the refresh token.
        user = getpass.getuser()
        await _security(
            "-i",
            commands=(
                f'add-generic-password -U -a "{user}" -s "Claude Code-credentials"'
                f" -X {secret}\n"
            ),
        )


# ``""`` when no browser reaches us within five minutes, as over ssh.
async def _browser_code(
    challenge: str,
    state: str,
    *,
    show: Callable[[str], object],
) -> tuple[str, str]:
    """Return the code the browser redirects back with, and the redirect it used."""
    code = asyncio.Future[str]()
    server = await asyncio.start_server(
        lambda reader, writer: _redirected(reader, writer, state=state, code=code),
        "127.0.0.1",
        0,
    )
    redirect = f"http://localhost:{server.sockets[0].getsockname()[1]}/callback"
    url = _authorize_url(redirect, challenge, state)
    async with server:
        show(f"Sign in to Claude in your browser:\n\n  {url}\n")
        webbrowser.open(url)
        try:
            return await asyncio.wait_for(code, timeout=300.0), redirect
        except TimeoutError:
            return "", redirect


async def _pasted_code(
    challenge: str,
    state: str,
    *,
    show: Callable[[str], object],
) -> tuple[str, str]:
    """Return the code the user pastes from the sign-in page, and its redirect."""
    redirect = "https://platform.claude.com/oauth/code/callback"
    url = _authorize_url(redirect, challenge, state)
    show(f"Open this, sign in, and paste the code it shows:\n\n  {url}\n")
    pasted = (await asyncio.to_thread(input, "Code: ")).strip()
    code, _, pasted_state = pasted.partition("#")
    if pasted_state != state:
        raise AuthError("the pasted code belongs to another sign-in")
    return code, redirect


async def _redirected(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    state: str,
    code: asyncio.Future[str],
) -> None:
    """Answer the browser's redirect, and take the code it carries."""
    request_line = (await reader.readline()).decode(errors="replace")
    target = request_line.split(" ")[1] if request_line.count(" ") >= 2 else ""
    url = urlparse(target)
    params = parse_qs(url.query)
    received = (params.get("code") or [""])[0]
    ok = url.path == "/callback" and bool(received)
    ok = ok and (params.get("state") or [""])[0] == state
    page = (
        "Signed in; you can close this tab."
        if ok
        else "Sign-in failed; see the terminal."
    )
    writer.write(
        f"HTTP/1.1 {200 if ok else 400} {'OK' if ok else 'Bad Request'}\r\n"
        f"content-type: text/plain\r\ncontent-length: {len(page)}\r\n"
        f"connection: close\r\n\r\n{page}".encode(),
    )
    await writer.drain()
    writer.close()
    if ok and not code.done():
        code.set_result(received)


def _authorize_url(redirect: str, challenge: str, state: str) -> str:
    query = urlencode(
        {
            "code": "true",
            "client_id": _client_id(),
            "response_type": "code",
            "redirect_uri": redirect,
            "scope": " ".join(_scopes()),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        },
    )
    return f"https://claude.com/cai/oauth/authorize?{query}"


def _client_id() -> str:
    """Return the OAuth client ``claude`` signs in as; the same for everyone."""
    return "9d1c250a-e61b-44d9-88ed-5944d1962f5e"


def _scopes() -> tuple[str, ...]:
    return (
        "user:profile",
        "user:inference",
        "user:sessions:claude_code",
        "user:mcp_servers",
        "user:file_upload",
    )


# The store is ``claude``'s, and holds more than this login: its MCP servers'
# logins sit beside it, so a write replaces only ``claudeAiOauth``.
def _document(text: str) -> dict[str, object]:
    return json_loads(text or "{}", dict[str, object])


def _login_of(text: str) -> dict[str, object]:
    stored = _document(text)
    return from_plain(stored.get("claudeAiOauth"), dict[str, object], default={})


def _read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _write_private(path: Path, text: str) -> None:
    """Replace ``path`` with ``text``, never readable by anyone else, even briefly."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staged = path.with_suffix(".tmp")
    fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    # ``O_CREAT``'s mode applies only to a new file; one left over keeps its own.
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as file:
        file.write(text)
    staged.replace(path)


def _find_entry() -> tuple[str, ...]:
    """Return the arguments that find the Keychain entry ``claude`` keeps its login in."""
    return (
        "find-generic-password",
        "-a",
        getpass.getuser(),
        "-s",
        "Claude Code-credentials",
    )


async def _security(*args: str, commands: str = "") -> str:
    """Run ``security`` with ``args``, writing ``commands`` to its stdin."""
    async with await Child.start(
        "/usr/bin/security",
        *args,
        grace_sec=0.0,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    ) as child:
        out, _ = await child.process.communicate(commands.encode())
    return out.decode().strip() if child.process.returncode == 0 else ""
