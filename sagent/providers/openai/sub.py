"""OpenAI subscription provider (OAuth + Codex CLI billing).

Loads OAuth credentials from ``$CODEX_HOME/auth.json`` (or
``~/.codex/auth.json`` when unset), as written by ``codex login``, and uses
the OpenAI Responses API via the official Python SDK. Subscription billing
flows through the user's ChatGPT plan.

Usage::

    from sagent.providers import OpenAISubscription

    provider = OpenAISubscription.from_credentials()
    model = provider.model("gpt-5.4-mini")
    response = await model.buffer(request)

NOTICE -- Provenance and Terms of Use
-------------------------------------

The OAuth client_id, scopes, endpoints, and protocol details used
below are taken from OpenAI's open-source Codex CLI (Apache-2.0):

    https://github.com/openai/codex
        codex-rs/login/src/auth/manager.rs:921   -- CLIENT_ID
        codex-rs/login/src/auth/manager.rs:94    -- REFRESH_TOKEN_URL
        codex-rs/login/src/server.rs:51          -- DEFAULT_ISSUER
        codex-rs/login/src/server.rs:480-509     -- authorize URL (scope,
                                                    PKCE params, originator)
        codex-rs/model-provider-info/src/lib.rs:237
                                                 -- chatgpt.com/backend-api/codex

This module re-implements that documented protocol in Python against
the same ``chatgpt.com/backend-api/codex`` endpoint, using the user's
own credentials produced by ``codex login``. Subscription billing
flows through the caller's ChatGPT plan exactly as it does via the
official Codex CLI; cost figures are computed from public per-token
API pricing as a "what would this session cost at API rates" metric --
the user is not billed per-token for subscription auth.

Use of the ChatGPT subscription is governed by OpenAI's Terms of Use:

    https://openai.com/policies/row-terms-of-use/  (rest of world)
    https://openai.com/policies/terms-of-use/      (US)

Users are responsible for ensuring their own usage complies with
those terms. No representation is made that any particular usage is
authorized by OpenAI; users uncomfortable with that uncertainty
should prefer the API-key path (``OpenAI.from_key``).
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Final,
    NotRequired,
    TypedDict,
    cast,
    override,
)
from urllib.parse import quote_plus

import asyncio
import base64
import contextlib
import json
import logging
import os
import secrets
import sys
import time


if TYPE_CHECKING:
    from collections.abc import Callable

    from openai.types.responses.response_create_params import (
        ResponseCreateParamsStreaming,
    )

    import httpx2
    import openai

    from sagent.types.model import (
        ModelRequest,
        ModelResponse,
    )
    from sagent.types.runtime import (
        RuntimeEvent,
    )
else:
    from wrapt import lazy_import

    httpx2 = lazy_import("httpx2")  # 100ms cold.
    openai = lazy_import("openai")  # 493ms cold.


from sagent.catalog.openai import subscription, subscription_models
from sagent.catalog.table import ModelCatalog
from sagent.lib.atomic_file import atomic_write_bytes
from sagent.lib.codec import MutablePlainTree, ReadError, from_plain, loads
from sagent.providers.lib.oauth import (
    AuthCodeListener,
    credential_file_lock,
    credentials_path,
    parse_manual_auth_code,
    pkce_pair,
)
from sagent.providers.lib.perloop import PerLoop
from sagent.providers.openai.api import OpenAI
from sagent.providers.openai.responses import _OpenAIResponsesModel
from sagent.types.exceptions import (
    AuthRefreshError,
)


logger = logging.getLogger(__name__)

# OAuth/protocol constants below mirror OpenAI's open-source Codex CLI
# (Apache-2.0); see this module's NOTICE block for full citations.
#   CLIENT_ID:   codex-rs/login/src/auth/manager.rs:921
#   token URL:   codex-rs/login/src/auth/manager.rs:94
#   issuer:      codex-rs/login/src/server.rs:51
#   scope:       codex-rs/login/src/server.rs:493 (verbatim match)
#   base URL:    codex-rs/model-provider-info/src/lib.rs:237.
_CLIENT_ID: Final = "app_EMoamEEZ73f0CkXaXp7hrann"
_TOKEN_URL: Final = "https://auth.openai.com/oauth/token"  # noqa: S105 -- This literal is a public OAuth endpoint, not a credential.
DEFAULT_CREDENTIALS_PATH = Path(".codex") / "auth.json"
_SCOPES: Final = (
    "openid profile email offline_access api.connectors.read api.connectors.invoke"
)
DEFAULT_REFRESH_BUFFER_SEC = (
    300.0  # house-ignore[globals] -- OAuth refresh lead time, user-retunable.
)
# OAuth-registered redirect URI port. Not user-tunable: the OAuth app
# fingerprinted as Codex CLI has this port baked into its allowed
# redirects on OpenAI's side.
_CALLBACK_PORT: Final = 1455


def _default_credentials_path() -> Path:
    """Return the active Codex auth file, honoring ``CODEX_HOME``."""
    codex_home = os.environ.get("CODEX_HOME", "")
    if codex_home:
        return Path(codex_home).expanduser() / "auth.json"
    return Path.home() / DEFAULT_CREDENTIALS_PATH  # noqa: TID251 -- The Codex CLI reads this fixed vendor path (AGENTS.md rule 3); an XDG root must not relocate it.  # house-ignore[xdg-literal] -- The Codex CLI reads this fixed vendor path (AGENTS.md rule 3).


class _CredentialFileError(ValueError):
    """Stored credentials do not match the subscription OAuth schema."""


class OpenAISubscription:
    """OpenAI provider -- OAuth + ChatGPT subscription billing.

    Uses the OpenAI catalog with token limits clamped to the subscription
    wire contract.
    Cost tracking uses standard API per-token pricing even though
    subscription users pay a flat fee. This is intentional: it gives
    a consistent "what would this session cost at API rates" metric
    regardless of auth mode.
    """

    catalog = ModelCatalog(models=subscription_models(), transport=subscription())

    class Credentials(TypedDict):
        """OAuth credentials for an OpenAI ChatGPT subscription."""

        access_token: str
        refresh_token: str
        account_id: str
        expires_at: float
        id_token: NotRequired[str]

    def __init__(
        self,
        *,
        access_token: str,
        refresh_token: str,
        account_id: str,
        expires_at: float,
        account: str | None = None,
        refresh_buffer_sec: float = DEFAULT_REFRESH_BUFFER_SEC,
    ) -> None:
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._account_id = account_id  # ChatGPT account id (from JWT)
        self._account = account  # Local credential slot name.
        self._expires_at = expires_at
        self._refresh_buffer_sec = refresh_buffer_sec
        # Client and its token cached together, per loop: the client's
        # pool belongs to the loop that opened it, and a token outliving
        # its client would authenticate a connection that no longer
        # exists. The guarding lock is per loop for the same reason -- it
        # binds to the loop that first contends on it.
        self._authed: PerLoop[tuple[openai.AsyncOpenAI, str] | None] = PerLoop(
            lambda: None,
        )
        self._lock: PerLoop[asyncio.Lock] = PerLoop(asyncio.Lock)

    @classmethod
    def from_key(
        cls,
        api_key: str,
        *,
        base_url: str | None = None,
    ) -> OpenAI:
        """Create an API-key provider (delegates to ``OpenAI``).

        Subscription billing is incompatible with API keys, so this
        returns a plain ``OpenAI`` instance.

        Args:
          api_key: OpenAI API key.
          base_url: Override the default endpoint URL.

        Returns:
          provider: ``OpenAI`` provider instance.

        """
        return OpenAI.from_key(api_key, base_url=base_url)

    @classmethod
    def from_credentials(
        cls,
        creds: OpenAISubscription.Credentials | None = None,
        *,
        account: str | None = None,
        refresh_buffer_sec: float = DEFAULT_REFRESH_BUFFER_SEC,
    ) -> OpenAISubscription:
        """Create provider from OAuth credentials.

        Args:
          creds: Pre-loaded credentials, or ``None`` to auto-load from disk.
          account: Named credential slot. ``None`` uses the active Codex
            ``auth.json`` under ``$CODEX_HOME`` or ``~/.codex``.
          refresh_buffer_sec: Seconds before token expiry to trigger
              proactive refresh.

        Returns:
          provider: Subscription provider instance.

        """
        if creds is None:
            creds = cls.load(account=account)
        return cls(
            access_token=creds["access_token"],
            refresh_token=creds["refresh_token"],
            account_id=creds["account_id"],
            expires_at=creds["expires_at"],
            account=account,
            refresh_buffer_sec=refresh_buffer_sec,
        )

    @classmethod
    def login(
        cls,
        output: IO[str] | None = None,
        *,
        listener_timeout_sec: float = 300.0,
        account: str | None = None,
        manual: bool = False,
    ) -> OpenAISubscription.Credentials:
        """Perform interactive OAuth login via browser PKCE flow.

        Args:
          output: Stream for user-facing messages. ``None`` uses stdout.
          listener_timeout_sec: Seconds to wait for the browser callback.
          account: Named credential slot. ``None`` writes to the active Codex
            ``auth.json`` under ``$CODEX_HOME`` or ``~/.codex``.
          manual: Print a URL and prompt for a pasted code without waiting for
            a browser callback.

        Returns:
          creds: Fresh OAuth credentials.

        Raises:
          RuntimeError: If the callback listener fails to start or token exchange fails.
          AuthRefreshError: The issued access token carries no expiry.

        """
        out = output or sys.stdout
        verifier, challenge = pkce_pair()
        state = secrets.token_urlsafe(32)

        listener: AuthCodeListener | None = None
        # Hydra (OpenAI's OAuth server) matches redirect_uri against a
        # registered allow-list by exact string; that list holds the
        # ``localhost`` form, so ``127.0.0.1`` is rejected with
        # ``authorize_hydra_invalid_request``. Advertise ``localhost``
        # while the listener still binds IPv4 loopback.
        redirect_uri = f"http://localhost:{_CALLBACK_PORT}/auth/callback"
        if not manual:
            listener = AuthCodeListener(
                state,
                port=_CALLBACK_PORT,
                callback_path="/auth/callback",
            )
            try:
                listener.start()
                redirect_uri = listener.redirect_uri_for_host("localhost")
            except OSError as e:
                raise RuntimeError(f"Failed to start callback listener: {e}") from e

        url = (
            "https://auth.openai.com/oauth/authorize"
            f"?client_id={quote_plus(_CLIENT_ID)}"
            f"&response_type=code"
            f"&redirect_uri={quote_plus(redirect_uri)}"
            f"&scope={quote_plus(_SCOPES)}"
            f"&code_challenge={quote_plus(challenge)}"
            f"&code_challenge_method=S256"
            f"&state={quote_plus(state)}"
            f"&codex_cli_simplified_flow=true"
            f"&id_token_add_organizations=true"
            f"&originator=codex_cli_rs"
        )
        out.write(
            "Starting OpenAI ChatGPT (Codex) OAuth login.\n"
            "This reuses the OAuth flow from openai/codex (Apache-2.0).\n"
            "Subscription use is governed by OpenAI's Terms of Use:\n"
            "  https://openai.com/policies/row-terms-of-use/\n\n"
            f"Open this URL in any browser to authenticate:\n\n  {url}\n\n",
        )
        out.flush()

        if listener is None:
            auth_code = parse_manual_auth_code(
                input("Paste the authorization code or redirected URL here: "),
                state,
            )
        else:
            try:
                auth_code = listener.wait(listener_timeout_sec)
            finally:
                listener.stop()

        with httpx2.Client() as http:
            r = http.post(
                _TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": auth_code,
                    "redirect_uri": redirect_uri,
                    "client_id": _CLIENT_ID,
                    "code_verifier": verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )
            if r.status_code >= 400:
                body = r.text[:500]
                raise RuntimeError(f"Token exchange failed ({r.status_code}): {body}")
            data: dict[str, MutablePlainTree] = cast(
                dict[str, MutablePlainTree],
                r.json(),
            )

        access_token = str(data["access_token"])
        account_id = _jwt_claim(
            access_token,
            "https://api.openai.com/auth",
            "chatgpt_account_id",
        )
        creds = cls.Credentials(
            access_token=access_token,
            refresh_token=str(data["refresh_token"]),
            account_id=account_id,
            expires_at=_token_expiry(access_token, data),
        )
        id_token = data.get("id_token")
        if isinstance(id_token, str):
            creds["id_token"] = id_token

        cls.save(creds, account=account)
        plan = _jwt_claim(
            access_token,
            "https://api.openai.com/auth",
            "chatgpt_plan_type",
        )
        out.write(f"Authenticated (plan={plan}, account={account_id[:8]}...).\n")
        out.flush()
        return creds

    def model(
        self,
        model_id: str | None = None,
    ) -> _OpenAISubModel:
        """Create a Responses API model backend.

        Args:
          model_id: Catalog id with optional tags, or a role name.

        Returns:
          model: Responses API model backend.

        Raises:
          UnknownModelError: ``model_id`` is not in the catalog.
          UnsupportedTagError: The id asks for a tag this model or
              transport does not offer.

        """
        mid = model_id if model_id is not None else "default"
        # ``+1m`` is absent from the narrowed subscription catalog, so a
        # ``+1m`` id raises rather than silently clamping: the suffix buys
        # nothing under the subscription wire contract.
        capability, settings = self.catalog.resolve(mid)
        return _OpenAISubModel(
            provider=self,
            capability=capability,
            settings=settings,
        )

    # -- Token management ----------------------------------------------

    @property
    def expired(self) -> bool:
        """True if the access token is within ``refresh_buffer_sec`` of expiry."""
        return time.time() > self._expires_at - self._refresh_buffer_sec

    async def get_sdk(self) -> openai.AsyncOpenAI:
        """Return OAuth-authed SDK client, refreshing as needed.

        Returns:
          client: Authenticated ``AsyncOpenAI`` SDK client.

        """
        token = await self._ensure_valid()
        cached = self._authed.peek()
        if cached is not None and cached[1] == token:
            return cached[0]
        # No refresh here: ``_ensure_valid`` is the one path that holds the
        # cross-process credential lock, and a refresh outside it lets two
        # processes spend one rotating refresh token.
        async with self._lock.get():
            token = self._access_token
            cached = self._authed.peek()
            if cached is not None and cached[1] == token:
                return cached[0]
            sdk = openai.AsyncOpenAI(
                api_key=token,
                base_url="https://chatgpt.com/backend-api/codex",
                default_headers={
                    "chatgpt-account-id": self._account_id,
                    # Required by chatgpt.com/backend-api/codex to identify
                    # the calling client; mirrors openai/codex's request
                    # signing (codex-rs/login/src/auth/default_client.rs).
                    "originator": "codex",
                },
            )
            self._authed.set((sdk, token))
            if cached is not None:
                await cached[0].close()
            return sdk

    async def close_sdk(self) -> None:
        """Close and clear the shared OAuth SDK client.

        Idempotent: a no-op when no SDK has been created. This loop's
        client only -- a pool belongs to the loop that opened it, so
        closing another loop's from here would break a pool still in use.
        """
        async with self._lock.get():
            cached = self._authed.peek()
            self._authed.clear()
        if cached is not None:
            await cached[0].close()

    # A concurrent process holding the same account may have refreshed and written newer
    # creds to disk. Load them under the already-held file lock and adopt only when the
    # on-disk access token (a) DIFFERS from ours and (b) is itself non-expired. The
    # boolean answers exactly "did we just adopt a token we can use without a network
    # refresh?" -- never "is our current token valid?".
    #
    # Two failure modes this guards, both of which return False so the caller refreshes:
    # - disk token MATCHES ours: a sibling did not refresh; our token is the one that
    # needs replacing (especially under ``handle_auth_error``, where the server already
    # rejected it regardless of the local clock). - disk token DIFFERS but is itself
    # already expired (its own refresh aged out, or clock skew): adopting it would 401
    # again next call.
    #
    # ``load`` raises ``ValueError`` on malformed JSON or an incomplete record; it
    # means "no usable disk creds", so it returns False.
    #
    # Disk I/O runs in a worker thread so the event loop is not blocked on a slow/NFS
    # read while the credential lock is held. Adopting a new token invalidates the
    # cached SDK; the stale client is closed (not merely dropped) so its pooled HTTP
    # connections are released immediately rather than orphaned until GC.
    async def _adopt_fresher_disk_creds(self) -> bool:
        """Adopt a DIFFERENT, still-valid sibling-written disk token."""
        try:
            creds = await asyncio.to_thread(
                OpenAISubscription.load,
                account=self._account,
            )
            disk_at = creds["access_token"]
        except (FileNotFoundError, ValueError):
            return False
        if disk_at == self._access_token:
            return False
        self._access_token = disk_at
        self._refresh_token = creds["refresh_token"]
        self._account_id = creds["account_id"]
        self._expires_at = creds["expires_at"]
        await self._discard_sdk()
        return not self.expired

    # Idempotent. The caller must already hold ``self._lock``.
    async def _discard_sdk(self) -> None:
        """Drop and close the cached SDK so its pooled connections release."""
        cached = self._authed.peek()
        self._authed.clear()
        if cached is not None:
            await cached[0].close()

    # Holds a cross-process file lock around the read-disk → maybe- POST → write-disk
    # sequence so concurrent processes can't both consume the same refresh_token and
    # have one revoked by the OAuth endpoint's rotation rule.
    async def _ensure_valid(self) -> str:
        """Return a valid access token, reloading from disk or refreshing."""
        if not self.expired:
            return self._access_token
        cred_path = credentials_path(_default_credentials_path(), self._account)
        async with self._lock.get(), credential_file_lock(cred_path):
            if not self.expired:
                return self._access_token
            if await self._adopt_fresher_disk_creds():
                return self._access_token
            await self._refresh()
            return self._access_token

    async def handle_auth_error(self) -> None:
        """Reload credentials from disk or force-refresh on 401.

        Holds the cross-process credential lock so a concurrent
        sibling can't refresh between our disk-check and POST.
        """
        cred_path = credentials_path(_default_credentials_path(), self._account)
        async with self._lock.get(), credential_file_lock(cred_path):
            # Adopt a sibling's fresher creds only if they are actually valid;
            # an adopted-but-expired token would 401 again on retry, so fall
            # through to ``_refresh`` (the same rule ``_ensure_valid`` uses).
            if await self._adopt_fresher_disk_creds():
                return
            await self._refresh()
            await self._discard_sdk()

    async def _refresh(self) -> None:
        """Exchange the refresh token for a new access token."""
        async with httpx2.AsyncClient() as http:
            r = await http.post(
                _TOKEN_URL,
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                    "client_id": _CLIENT_ID,
                },
                headers={"Content-Type": "application/json"},
                timeout=15.0,
            )
            if r.status_code in (400, 401):
                # Refresh token was rotated by another process, revoked
                # server-side, or aged out. Retrying is pointless --
                # surface a clean user-facing error so the renderer
                # shows actionable text instead of an httpx2 traceback.
                raise AuthRefreshError(
                    "OpenAI Codex subscription session expired or revoked. "
                    "Run /login to re-authenticate, or /model to switch "
                    "providers.",
                )
            r.raise_for_status()
            data: dict[str, MutablePlainTree] = cast(
                dict[str, MutablePlainTree],
                r.json(),
            )
        access_token = str(data["access_token"])
        creds = OpenAISubscription.Credentials(
            access_token=access_token,
            refresh_token=str(data["refresh_token"]),
            account_id=self._account_id,
            expires_at=_token_expiry(access_token, data),
        )
        self._access_token = creds["access_token"]
        self._refresh_token = creds["refresh_token"]
        self._expires_at = creds["expires_at"]
        try:
            creds["account_id"] = (
                await asyncio.to_thread(OpenAISubscription.load, account=self._account)
            )["account_id"]
        except FileNotFoundError:
            pass
        except ValueError:
            # Unusable creds file: malformed JSON or a record missing required
            # fields like ``{"tokens": {}}``. ``save`` below overwrites it with
            # the freshly-refreshed tokens; log first so the corruption is not
            # silently erased. Matches the swallow set in
            # ``_adopt_fresher_disk_creds``.
            logger.warning(
                "OpenAI creds file at %s was unreadable; overwriting with "
                "refreshed tokens",
                credentials_path(_default_credentials_path(), self._account),
            )
        # ``save`` merges into the file's existing ``tokens``, so fields this
        # record omits (``id_token``) survive the rewrite.
        await asyncio.to_thread(OpenAISubscription.save, creds, account=self._account)

    # -- Credential I/O ------------------------------------------------

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        *,
        account: str | None = None,
    ) -> OpenAISubscription.Credentials:
        """Load OAuth credentials from disk.

        Args:
          path: Explicit credentials file path, or ``None`` for default.
          account: Named credential slot. ``None`` reads the active Codex
            ``auth.json`` under ``$CODEX_HOME`` or ``~/.codex``.

        Returns:
          creds: Loaded OAuth credentials.

        Raises:
          FileNotFoundError: If no credentials file exists.
          ValueError: If the file is not a complete OAuth credential record.

        """
        p = path or credentials_path(_default_credentials_path(), account)
        if not p.exists():
            raise FileNotFoundError(f"No credentials at {p}")
        decoded: object = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise _CredentialFileError(
                f"{p} must contain a JSON object with OpenAI subscription "
                "OAuth credentials.",
            )
        raw = cast(dict[str, MutablePlainTree], decoded)
        raw_tokens = raw.get("tokens")
        if not isinstance(raw_tokens, dict):
            auth_mode = raw.get("auth_mode")
            if auth_mode in ("apikey", "api_key") or "OPENAI_API_KEY" in raw:
                raise _CredentialFileError(
                    f"{p} contains OpenAI API-key credentials, not ChatGPT "
                    "subscription OAuth credentials. Use --provider OpenAI "
                    "--auth env, or run `sagent --provider OpenAISubscription "
                    "login` first.",
                )
            raise _CredentialFileError(
                f"{p} does not contain OpenAI subscription OAuth tokens. Run "
                "`sagent --provider OpenAISubscription login`.",
            )
        tokens = raw_tokens
        required = ("access_token", "refresh_token", "account_id")
        missing = [
            name
            for name in required
            if not from_plain(tokens.get(name), str, default="")
        ]
        if missing:
            raise _CredentialFileError(
                f"{p} is missing required OAuth fields: {', '.join(missing)}. "
                "Run `sagent --provider OpenAISubscription login`.",
            )
        access_token = cast(str, tokens.get("access_token"))
        expires_at = _jwt_exp(access_token)
        creds = cls.Credentials(
            access_token=access_token,
            refresh_token=cast(str, tokens.get("refresh_token")),
            account_id=cast(str, tokens.get("account_id")),
            expires_at=expires_at,
        )
        id_token = tokens.get("id_token")
        if isinstance(id_token, str):
            creds["id_token"] = id_token
        return creds

    @classmethod
    def save(
        cls,
        creds: OpenAISubscription.Credentials,
        path: Path | None = None,
        *,
        account: str | None = None,
    ) -> None:
        """Persist credentials in Codex CLI-compatible format.

        Args:
          creds: OAuth credentials to persist.
          path: Explicit file path, or ``None`` for default.
          account: Named credential slot. ``None`` writes to the active Codex
            ``auth.json`` under ``$CODEX_HOME`` or ``~/.codex``.

        """
        p = path or credentials_path(_default_credentials_path(), account)
        existing: dict[str, MutablePlainTree] = {}
        if p.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                decoded: object = json.loads(p.read_text(encoding="utf-8"))
                # Shape-checked, not just parse-checked: valid JSON of the
                # wrong shape (a list, or ``tokens`` as a list) would raise
                # below and leave the caller unable to persist a re-login.
                if isinstance(decoded, dict):
                    existing = cast(dict[str, MutablePlainTree], decoded)
        raw_tokens = existing.get("tokens")
        tokens = raw_tokens if isinstance(raw_tokens, dict) else {}
        tokens["access_token"] = creds["access_token"]
        tokens["refresh_token"] = creds["refresh_token"]
        tokens["account_id"] = creds["account_id"]
        id_token = creds.get("id_token", "")
        if id_token:
            tokens["id_token"] = id_token
        existing["auth_mode"] = "chatgpt"
        existing["tokens"] = tokens
        atomic_write_bytes(p, json.dumps(existing).encode(), file_mode=0o600)


class _OpenAISubModel(_OpenAIResponsesModel):
    """OAuth retry and Codex-specific request restrictions."""

    _provider: OpenAISubscription

    @override
    def _build_kwargs(self, request: ModelRequest) -> ResponseCreateParamsStreaming:
        body = super()._build_kwargs(request)
        body.pop("temperature", None)
        body.pop("max_output_tokens", None)
        return body

    @override
    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        try:
            return await super().stream(request, publish)
        except openai.AuthenticationError:
            await self._provider.handle_auth_error()
        return await super().stream(request, publish)


def _token_expiry(access_token: str, grant: dict[str, MutablePlainTree]) -> float:
    """Return the token's JWT ``exp``, else the grant's ``expires_in`` from now."""
    # ``exp`` wins because ``load`` re-derives expiry from the token alone; a
    # local-clock deadline would disagree with it under clock skew.
    expires_at = _jwt_exp(access_token)
    if expires_at > 0:
        return expires_at
    expires_in = from_plain(grant.get("expires_in"), float, default=0.0)
    if expires_in <= 0:
        # A zero deadline reads as always expired: every request would refresh
        # and rotate the refresh token.
        raise AuthRefreshError(
            "OpenAI issued an access token with no expiry. Run /login to "
            "re-authenticate.",
        )
    return time.time() + expires_in


# Every JWT read below is lenient: a token the issuer shaped unexpectedly reads as
# "no claim", never as an exception, so ``load`` keeps its ``ValueError`` contract
# and the refresh paths stay reachable.
def _jwt_payload(token: str) -> dict[str, object]:
    """Decode JWT payload without verification."""
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    raw = parts[1]
    raw += "=" * (-len(raw) % 4)
    try:
        return from_plain(loads(base64.urlsafe_b64decode(raw)), dict[str, object])
    except (ValueError, ReadError):
        return {}


def _jwt_exp(token: str) -> float:
    """Extract ``exp`` claim from a JWT without verification."""
    try:
        return from_plain(_jwt_payload(token).get("exp"), float, default=0.0)
    except ReadError:
        return 0.0


def _jwt_claim(token: str, namespace: str, key: str) -> str:
    """Extract a nested string claim from a JWT namespace object."""
    try:
        claims = from_plain(
            _jwt_payload(token).get(namespace),
            dict[str, object],
            default={},
        )
    except ReadError:
        return ""
    claim = claims.get(key)
    return claim if isinstance(claim, str) else ""
