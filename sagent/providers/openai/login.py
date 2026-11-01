"""A ChatGPT login: the one ``codex login`` keeps, or one in a named file.

``codex`` reads the same file, so signing in here signs it in too. The OAuth
client, scopes, and endpoints are those of OpenAI's open-source Codex CLI
(Apache-2.0, https://github.com/openai/codex, ``codex-rs/login``). Use of a
ChatGPT plan this way is governed by OpenAI's Terms of Use; the API-key
provider is the path no such question touches.

Typical usage example:

  login = CodexLogin(resources=shared)          # $CODEX_HOME or ~/.codex
  work = CodexLogin(Path("~/.codex/auth-work.json"), resources=shared)

  await login.sign_in()          # In the browser; over ssh, paste the redirect.
  await login.sign_in_device()   # Or enter a code on another device.
  token, account = await login.token()    # Renewed before it expires.
  await login.lend(rejected="")           # A ``TokenLease``, for a borrower.

A server that keeps its users' logins itself runs the same steps one request
at a time: :func:`request_device_code`, then :func:`poll_device_code` until it
returns the tokens; :func:`refresh_grant` and :func:`revoke_refresh_token` after.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final
from urllib.parse import urlencode

import asyncio
import base64
import hashlib
import os
import secrets
import time

from treekle import ReadError, from_plain, json_dumps, json_loads

from sagent.lib.files.owner_only import write_private
from sagent.model.exceptions import AuthError
from sagent.providers.login import (
    file_lock,
    open_browser,
    read_pasted_code,
    redirected,
)
from sagent.providers.openai.lease import LeasedToken


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import httpx2

    from sagent.providers.types import ProviderResources


__all__ = [
    "ISSUER",
    "ORIGINATOR",
    "CodexAccount",
    "CodexLogin",
    "CodexTokens",
    "DeviceCode",
    "codex_account",
    "poll_device_code",
    "refresh_grant",
    "request_device_code",
    "revoke_refresh_token",
]


ISSUER: Final = "https://auth.openai.com"
"""Where ChatGPT logins are issued, renewed and revoked."""

ORIGINATOR: Final = "codex_cli_rs"
"""The Codex CLI's own name for itself; ChatGPT files any other under "Other"
(codex-rs default_client.rs, is_first_party_originator)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CodexAccount:
    """What a login says of its own account."""

    plan: str
    """The ChatGPT plan (``plus``, ``pro``); empty when no token says."""

    email: str


@dataclass(frozen=True, slots=True, kw_only=True)
class CodexTokens:
    """A login's tokens, as a sign-in or a renewal issues them."""

    access_token: str
    refresh_token: str
    account_id: str
    """The ChatGPT account the tokens are for."""

    expires_at: float
    """When the access token expires, in seconds since the epoch."""

    id_token: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceCode:
    """A device sign-in in progress: what the user enters, and how to poll it."""

    verification_url: str
    """The page the user enters the code on."""

    user_code: str
    device_auth_id: str
    interval_sec: float
    """How long to wait between polls."""


class CodexLogin:
    """A login to a ChatGPT plan, kept fresh.

    Args:
      path: A credentials file; ``None`` uses the one ``codex`` keeps, under
        ``$CODEX_HOME`` or ``~/.codex``.
      resources: Where the HTTP client comes from.

    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        resources: ProviderResources,
    ) -> None:
        self.path = (path or self.default_path()).expanduser()
        self._resources = resources
        self._adopt({})
        self._lock = asyncio.Lock()
        self._file_lock = self.path.with_suffix(".json.lock")

    @classmethod
    def default_path(cls) -> Path:
        """Return the file ``codex`` keeps its login in: under ``$CODEX_HOME``, else ``~/.codex``."""
        home = os.environ.get("CODEX_HOME") or ""
        return (
            Path(home) / "auth.json" if home else Path("~/.codex/auth.json")
        ).expanduser()

    async def token(self) -> tuple[str, str]:
        """Return an access token good for at least five more minutes.

        Returns:
          token: The stored token, or a renewed one when it was about to expire.
          account_id: The ChatGPT account the token is for.

        Raises:
          AuthError: There is no login, or it has expired or been revoked.

        """
        async with self._lock:
            if self._expiring():
                self._adopt(await self._tokens())
            if self._expiring():
                await self._renew()
        return self._access, self._account_id

    async def renew(self) -> None:
        """Replace the access token, as after the server rejects it."""
        async with self._lock:
            await self._renew()

    async def lend(self, *, rejected: str) -> LeasedToken:
        """Lend this login's token to a borrower, as a ``TokenLease``.

        Args:
          rejected: The token the borrower's server refused; empty when it
            only aged out.

        Returns:
          token: The token held, renewed first when it is the one refused.

        """
        token, account = await self.token()
        if rejected == token:
            await self.renew()
            token, account = await self.token()
        return LeasedToken(
            access_token=token,
            account_id=account,
            expires_at=self._expires_at,
        )

    async def sign_in(
        self,
        *,
        headless: bool = False,
        show: Callable[[str], object] = print,
    ) -> None:
        """Sign in through the browser, or with the redirect pasted from it.

        Args:
          headless: Skip the browser and take a pasted redirect at once.
          show: Where to write the sign-in link for the user.

        Raises:
          AuthError: The pasted redirect belongs to another sign-in, or the
            server refused the code.

        """
        verifier = secrets.token_urlsafe(64)
        # A SHA-256 digest is 32 bytes, which base64 pads with exactly one ``=``.
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .removesuffix(b"=")
            .decode()
        )
        state = secrets.token_urlsafe()
        # The registered redirect is this exact string: ``127.0.0.1`` or any
        # other port is refused by the server.
        redirect = "http://localhost:1455/auth/callback"
        url = _authorize_url(redirect, challenge=challenge, state=state)
        code = await _code(url, state=state, show=show, headless=headless)
        grant = await _exchange(
            self._resources.http,
            code,
            redirect=redirect,
            verifier=verifier,
            issuer=ISSUER,
        )
        async with self._lock, file_lock(self._file_lock):
            await self._save(_issued(grant))

    async def sign_in_device(self, *, show: Callable[[str], object] = print) -> None:
        """Sign in by entering a one-time code on another device's browser.

        Args:
          show: Where to write the page and the code for the user.

        Raises:
          TimeoutError: The code was not approved within fifteen minutes.

        """
        http = self._resources.http
        code = await request_device_code(http)
        show(f"Open {code.verification_url} and enter the code {code.user_code}.\n")
        async with asyncio.timeout(900.0):
            while True:
                await asyncio.sleep(max(code.interval_sec, 1.0))
                tokens = await poll_device_code(http, code)
                if tokens is not None:
                    break
        async with self._lock, file_lock(self._file_lock):
            await self._save(tokens)

    async def sign_out(self) -> None:
        """Revoke the stored refresh token, as ``codex logout`` does."""
        tokens = await self._tokens()
        await revoke_refresh_token(
            self._resources.http,
            _text(tokens, "refresh_token"),
        )

    async def account(self) -> CodexAccount:
        """Return the plan and the email the stored tokens carry.

        Returns:
          account: What the ID token says, else the access token.

        """
        tokens = await self._tokens()
        return _account(_text(tokens, "id_token"), _text(tokens, "access_token"))

    def _expiring(self) -> bool:
        # Five minutes early, so a token never expires mid-request.
        return time.time() > self._expires_at - 300.0

    def _adopt(self, tokens: Mapping[str, object]) -> None:
        self._access = _text(tokens, "access_token")
        self._account_id = _text(tokens, "account_id")
        self._expires_at = _expiry(self._access)

    # A refresh token works once: two processes refreshing together leave one
    # of them holding a revoked token.
    async def _renew(self) -> None:
        """Rotate the token, under a lock every process on this login shares."""
        async with file_lock(self._file_lock):
            before = self._access
            tokens = await self._tokens()
            self._adopt(tokens)
            if self._access != before and not self._expiring():
                return  # Another process rotated it while this one waited.
            # What the answer leaves out stays stored: ``_save`` keeps it.
            await self._save(
                await refresh_grant(
                    self._resources.http,
                    _text(tokens, "refresh_token"),
                ),
            )

    async def _tokens(self) -> dict[str, object]:
        """Return the stored tokens."""
        stored = await asyncio.to_thread(_stored, self.path)
        tokens = from_plain(stored.get("tokens"), dict[str, object], default={})
        if _text(tokens, "refresh_token"):
            return tokens
        if (
            stored.get("auth_mode") in ("apikey", "api_key")
            or "OPENAI_API_KEY" in stored
        ):
            raise AuthError(
                f"{self.path} holds an API key, not a ChatGPT login; sign in again.",
            )
        raise AuthError(f"{self.path} holds no ChatGPT login; sign in first.")

    async def _save(self, issued: CodexTokens) -> None:
        """Store tokens, keeping every other field ``codex`` keeps beside them."""
        stored = await asyncio.to_thread(_stored, self.path)
        tokens = from_plain(stored.get("tokens"), dict[str, object], default={})
        tokens["access_token"] = issued.access_token
        for name, value in (
            ("refresh_token", issued.refresh_token),
            ("id_token", issued.id_token),
            ("account_id", issued.account_id),
        ):
            if value:
                tokens[name] = value
        stored |= {
            "auth_mode": "chatgpt",
            "tokens": tokens,
            "last_refresh": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        await asyncio.to_thread(write_private, self.path, json_dumps(stored))
        self._adopt(tokens)
        # A token that carries no expiry lives as long as its grant said.
        self._expires_at = issued.expires_at


async def request_device_code(
    http: httpx2.AsyncClient,
    *,
    issuer: str = ISSUER,
) -> DeviceCode:
    """Ask the issuer for a one-time code the user enters on its device page.

    Args:
      http: The client the request goes through.
      issuer: Where logins are issued.

    Returns:
      code: The page and the code to show the user, and how often to poll.

    Raises:
      HTTPStatusError: The issuer refused; a 404 means device sign-in is off.

    """
    response = await http.post(
        f"{issuer}/api/accounts/deviceauth/usercode",
        json={"client_id": _client_id()},
        timeout=15.0,
    )
    response.raise_for_status()
    issued = json_loads(response.text, dict[str, object])
    # Codex reads the code under either name, and the interval as text.
    return DeviceCode(
        verification_url=f"{issuer}/codex/device",
        user_code=from_plain(issued.get("user_code") or issued.get("usercode"), str),
        device_auth_id=from_plain(issued.get("device_auth_id"), str),
        interval_sec=float(str(issued.get("interval", 5.0))),
    )


async def poll_device_code(
    http: httpx2.AsyncClient,
    code: DeviceCode,
    *,
    issuer: str = ISSUER,
) -> CodexTokens | None:
    """Ask once whether the user approved ``code``; exchange it when they have.

    Args:
      http: The client the requests go through.
      code: What :func:`request_device_code` returned.
      issuer: Where logins are issued.

    Returns:
      tokens: The login's tokens, or ``None`` while the code waits for approval.

    Raises:
      HTTPStatusError: The issuer refused the poll.
      AuthError: The issuer refused the approved code.

    """
    response = await http.post(
        f"{issuer}/api/accounts/deviceauth/token",
        json={"device_auth_id": code.device_auth_id, "user_code": code.user_code},
        timeout=15.0,
    )
    # Not approved yet; Codex keeps polling on both.
    if response.status_code in (403, 404):
        return None
    response.raise_for_status()
    approved = json_loads(response.text, dict[str, object])
    grant = await _exchange(
        http,
        from_plain(approved.get("authorization_code"), str),
        redirect=f"{issuer}/deviceauth/callback",
        verifier=from_plain(approved.get("code_verifier"), str),
        issuer=issuer,
    )
    return _issued(grant)


async def refresh_grant(
    http: httpx2.AsyncClient,
    refresh_token: str,
    *,
    account_id: str = "",
    id_token: str = "",
    issuer: str = ISSUER,
) -> CodexTokens:
    """Exchange a refresh token for fresh tokens; the old refresh token is spent.

    Args:
      http: The client the request goes through.
      refresh_token: The token posted.
      account_id: The account the login is for, kept when the answer names none.
      id_token: The login's ID token, kept when the answer holds none.
      issuer: Where logins are issued.

    Returns:
      tokens: The new tokens, keeping ``refresh_token`` and ``id_token`` when the
        answer leaves them out.

    Raises:
      AuthError: The refresh token was rotated, revoked or aged out.
      HTTPStatusError: The issuer failed otherwise.

    """
    # Codex posts a renewal as JSON, and a code exchange as a form.
    response = await http.post(
        f"{issuer}/oauth/token",
        json={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": _client_id(),
        },
        timeout=15.0,
    )
    grant = _granted(response)
    return _issued(
        grant,
        refresh_token=refresh_token,
        account_id=account_id,
        id_token=id_token,
    )


async def revoke_refresh_token(
    http: httpx2.AsyncClient,
    refresh_token: str,
    *,
    issuer: str = ISSUER,
) -> None:
    """Revoke a refresh token at the issuer, as ``codex logout`` does.

    Args:
      http: The client the request goes through.
      refresh_token: The token to revoke.
      issuer: Where logins are issued.

    Raises:
      HTTPError: The issuer refused or could not be reached.

    """
    response = await http.post(
        f"{issuer}/oauth/revoke",
        json={
            "token": refresh_token,
            "token_type_hint": "refresh_token",
            "client_id": _client_id(),
        },
        timeout=10.0,
    )
    response.raise_for_status()


def codex_account(tokens: CodexTokens) -> CodexAccount:
    """Return the plan and the email a login's tokens carry, the ID token first.

    Args:
      tokens: The login's tokens.

    Returns:
      account: The ChatGPT plan and the email; empty where no token says.

    """
    return _account(tokens.id_token, tokens.access_token)


def _account(*tokens: str) -> CodexAccount:
    """Return the plan and the email ``tokens`` carry, the first token first."""
    claims = [_claims(t) for t in tokens]
    plans = (
        _claim(c, "https://api.openai.com/auth", "chatgpt_plan_type") for c in claims
    )
    emails = (
        _claim(c, "email") or _claim(c, "https://api.openai.com/profile", "email")
        for c in claims
    )
    return CodexAccount(
        plan=next(filter(None, plans), ""),
        email=next(filter(None, emails), ""),
    )


async def _exchange(
    http: httpx2.AsyncClient,
    code: str,
    *,
    redirect: str,
    verifier: str,
    issuer: str,
) -> dict[str, object]:
    """Exchange a sign-in's code for its tokens."""
    response = await http.post(
        f"{issuer}/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect,
            "client_id": _client_id(),
            "code_verifier": verifier,
        },
        timeout=15.0,
    )
    return _granted(response)


def _granted(response: httpx2.Response) -> dict[str, object]:
    """Return what a token request was granted."""
    if response.status_code in (400, 401):
        raise AuthError(
            "The ChatGPT login has expired or been revoked; sign in again.",
        )
    response.raise_for_status()
    return json_loads(response.text, dict[str, object])


# A refresh answer may leave out what did not change.
def _issued(
    grant: Mapping[str, object],
    *,
    refresh_token: str = "",
    account_id: str = "",
    id_token: str = "",
) -> CodexTokens:
    """Return the tokens ``grant`` issued, the rest kept from before."""
    access = from_plain(grant.get("access_token"), str)
    expires_at = _expiry(access)
    if not expires_at:
        # A token that carries no expiry lives as long as the grant says.
        expires_in = from_plain(grant.get("expires_in"), float, default=0.0)
        if expires_in <= 0:
            # Read as always expiring, every request would rotate the token.
            raise AuthError(
                "The ChatGPT login was issued with no expiry; sign in again.",
            )
        expires_at = time.time() + expires_in
    return CodexTokens(
        access_token=access,
        refresh_token=_text(grant, "refresh_token") or refresh_token,
        account_id=_claim(
            _claims(access),
            "https://api.openai.com/auth",
            "chatgpt_account_id",
        )
        or account_id,
        id_token=_text(grant, "id_token") or id_token,
        expires_at=expires_at,
    )


# Headless, or when another process holds the one port the redirect may name, no
# browser is opened: the user opens the link anywhere and pastes where it lands.
# Otherwise the browser has five minutes to come back first.
async def _code(
    url: str,
    *,
    state: str,
    show: Callable[[str], object],
    headless: bool,
    port: int = 1455,
) -> str:
    """Return the code the browser redirects back to ``port`` with, or the user pastes."""
    show(f"Sign in to ChatGPT in your browser:\n\n  {url}\n")
    code = asyncio.Future[str]()
    try:
        # Loopback alone: a sign-in code must not be offered to the network.
        server = await asyncio.start_server(
            lambda reader, writer: redirected(
                reader,
                writer,
                path="/auth/callback",
                state=state,
                code=code,
            ),
            "127.0.0.1",
            port,
        )
    except OSError:
        return await read_pasted_code(state=state, show=show, browser_tried=False)
    # Held open while the user pastes, too, so nothing else listening at the
    # redirect is handed the code.
    async with server:
        if not headless:
            open_browser(url)
            # Over ssh the browser cannot reach us; its address bar holds the code.
            with suppress(TimeoutError):
                return await asyncio.wait_for(code, timeout=300.0)
        return await read_pasted_code(
            state=state,
            show=show,
            browser_tried=not headless,
        )


def _authorize_url(redirect: str, *, challenge: str, state: str) -> str:
    query = urlencode(
        {
            "client_id": _client_id(),
            "response_type": "code",
            "redirect_uri": redirect,
            "scope": "openid profile email offline_access"
            " api.connectors.read api.connectors.invoke",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "codex_cli_simplified_flow": "true",
            "id_token_add_organizations": "true",
            "originator": ORIGINATOR,
        },
    )
    return f"{ISSUER}/oauth/authorize?{query}"


def _client_id() -> str:
    """Return the OAuth client ``codex`` signs in as; the same for everyone."""
    return "app_EMoamEEZ73f0CkXaXp7hrann"


def _stored(path: Path) -> dict[str, object]:
    """Return what the login file holds; nothing when it is missing or not an object."""
    try:
        return json_loads(path.read_text(), dict[str, object])
    except (FileNotFoundError, ValueError):
        return {}


def _text(fields: Mapping[str, object], name: str) -> str:
    """Return the text field ``name``; empty when absent."""
    return from_plain(fields.get(name), str, default="")


# Every token read is lenient: a token shaped unexpectedly has no claims, so
# the paths that renew it stay reachable.
def _claims(token: str) -> dict[str, object]:
    """Return a JWT's claims, unverified."""
    try:
        _, payload, *_ = token.split(".")
        # Surplus padding is ignored; too little is an error.
        return json_loads(base64.urlsafe_b64decode(payload + "=="), dict[str, object])
    except ValueError:
        return {}


def _claim(claims: Mapping[str, object], *keys: str) -> str:
    """Return the text claim ``keys`` names, nested; empty when absent or not text."""
    *scopes, key = keys
    try:
        for scope in scopes:
            claims = from_plain(claims.get(scope), dict[str, object])
        return from_plain(claims.get(key), str)
    except ReadError:
        return ""


def _expiry(token: str) -> float:
    """Return a JWT's expiry in seconds since the epoch; ``0`` when it has none."""
    try:
        return from_plain(_claims(token).get("exp"), float)
    except ReadError:
        return 0.0
