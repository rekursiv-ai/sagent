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

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import (
    IO,
    TYPE_CHECKING,
    ClassVar,
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
    from openai.types.responses.response_create_params import (
        ResponseCreateParamsStreaming,
    )

    import httpx2
    import openai
else:
    from wrapt import lazy_import

    httpx2 = lazy_import("httpx2")  # 100ms cold
    openai = lazy_import("openai")  # 493ms cold


from sagent.catalog import openai as openai_catalog
from sagent.lib.atomic_file import atomic_write_bytes
from sagent.lib.custom_json import (
    DictCodec,
    FloatCodec,
    MutableJSON,
)
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
from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelLimits,
)
from sagent.types.exceptions import (
    AuthRefreshError,
)
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
)
from sagent.types.providers import resolve
from sagent.types.runtime import (
    RuntimeEvent,
)


logger = logging.getLogger(__name__)

# OAuth/protocol constants below mirror OpenAI's open-source Codex CLI
# (Apache-2.0); see this module's NOTICE block for full citations.
#   CLIENT_ID:   codex-rs/login/src/auth/manager.rs:921
#   token URL:   codex-rs/login/src/auth/manager.rs:94
#   issuer:      codex-rs/login/src/server.rs:51
#   scope:       codex-rs/login/src/server.rs:493 (verbatim match)
#   base URL:    codex-rs/model-provider-info/src/lib.rs:237
_CLIENT_ID: Final = "app_EMoamEEZ73f0CkXaXp7hrann"
_TOKEN_URL: Final = "https://auth.openai.com/oauth/token"  # noqa: S105 -- not a secret; OAuth endpoint URL
_AUTHORIZE_URL: Final = "https://auth.openai.com/oauth/authorize"
_BASE_URL: Final = "https://chatgpt.com/backend-api/codex"
DEFAULT_CREDENTIALS_PATH = Path.home() / ".codex" / "auth.json"
_SCOPES: Final = (
    "openid profile email offline_access api.connectors.read api.connectors.invoke"
)
DEFAULT_REFRESH_BUFFER_SEC = (
    300.0  # config-globals: ignore -- OAuth refresh lead time, user-retunable
)
# OAuth-registered redirect URI port. Not user-tunable: the OAuth app
# fingerprinted as Codex CLI has this port baked into its allowed
# redirects on OpenAI's side.
_CALLBACK_PORT: Final = 1455
# Codex's subscription backend currently exposes a smaller practical
# context window than the public API model metadata. Local budgeting must
# plan against the wire contract so auto-compaction runs before the backend
# rejects an oversized request.
_SUBSCRIPTION_MAX_REQUEST_TOKENS = (
    272_000  # config-globals: ignore -- backend context budget, user-retunable
)
_SUBSCRIPTION_MAX_RESPONSE_TOKENS = (
    32_000  # config-globals: ignore -- backend response budget, user-retunable
)


def _default_credentials_path() -> Path:
    """Return the active Codex auth file, honoring ``CODEX_HOME``."""
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "auth.json"
    return DEFAULT_CREDENTIALS_PATH


class _CredentialFileError(ValueError):
    """Stored credentials do not match the subscription OAuth schema."""


def _subscription_context(cap: ModelCapability) -> Mapping[ContextTag, ModelLimits]:
    """Clamp the untagged window to the subscription wire contract.

    Only the default tag survives: ``+1m`` clamps to exactly the base id,
    so offering it would mislead a caller expecting 1M.
    """
    base = cap.context[""]
    return MappingProxyType(
        {
            "": replace(
                base,
                max_request_tokens=min(
                    base.max_request_tokens, _SUBSCRIPTION_MAX_REQUEST_TOKENS
                ),
                max_response_tokens=min(
                    base.max_response_tokens, _SUBSCRIPTION_MAX_RESPONSE_TOKENS
                ),
            )
        }
    )


class OpenAISubscription(OpenAI):
    """OpenAI provider -- OAuth + ChatGPT subscription billing.

    Derives CAPABILITIES from OpenAI (API pricing inherited, token
    limits clamped to the subscription wire contract, ``+1m`` ids
    dropped -- see the ``CAPABILITIES`` comprehension below).
    Cost tracking uses standard API per-token pricing even though
    subscription users pay a flat fee. This is intentional: it gives
    a consistent "what would this session cost at API rates" metric
    regardless of auth mode.
    """

    # Sub users pay a flat fee, so the most-capable model is the right default.
    # ``+1m`` only widens the input window, which the subscription backend caps
    # at ``_SUBSCRIPTION_MAX_REQUEST_TOKENS`` regardless -- so every ``+1m`` id
    # would clamp to exactly its base id. Rather than accept a suffix that buys
    # nothing (and silently mislead a caller expecting 1M), the catalog omits
    # ``+1m`` ids entirely: the base id is the only honest handle here. The
    # inherited ``OpenAI.DEFAULT_MODEL`` carries ``+1m``, so it is overridden
    # below to the base id to stay resolvable against this narrowed catalog.
    DEFAULT_MODEL: ClassVar[str] = "gpt-5.6-sol"

    CAPABILITIES: ClassVar[Mapping[str, ModelCapability]] = MappingProxyType(
        {
            name: replace(cap, context=_subscription_context(cap))
            for name, cap in openai_catalog.models().items()
        }
    )
    """The Responses wire: every advertised effort, clamped token windows."""

    TRANSPORT: ClassVar[ModelCapability] = openai_catalog.subscription()
    """Codex subscription: account auth, ``/fast`` maps to the priority tier."""

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
        self._account = account  # local credential slot name
        self._expires_at = expires_at
        self._refresh_buffer_sec = refresh_buffer_sec
        # Client and its token cached together, per loop: the client's
        # pool belongs to the loop that opened it, and a token outliving
        # its client would authenticate a connection that no longer
        # exists. The guarding lock is per loop for the same reason -- it
        # binds to the loop that first contends on it.
        self._authed: PerLoop[tuple[openai.AsyncOpenAI, str] | None] = PerLoop(
            lambda: None
        )
        self._lock: PerLoop[asyncio.Lock] = PerLoop(asyncio.Lock)

    @property
    def _sdk(self) -> openai.AsyncOpenAI | None:
        """The running loop's SDK client, if one has been opened."""
        cached = self._authed.peek()
        return cached[0] if cached else None

    @_sdk.setter
    def _sdk(self, value: openai.AsyncOpenAI | None) -> None:
        """Install or clear this loop's SDK client, keeping its token paired."""
        cached = self._authed.peek()
        if value is None:
            self._authed.clear()
        else:
            self._authed.set((value, cached[1] if cached else ""))

    @property
    def _sdk_token(self) -> str | None:
        """The token this loop's cached client authenticates with."""
        cached = self._authed.peek()
        return cached[1] if cached else None

    @_sdk_token.setter
    def _sdk_token(self, value: str | None) -> None:
        """Repoint the cached token, forcing a rotation on next use."""
        cached = self._authed.peek()
        if cached is not None and value is not None:
            self._authed.set((cached[0], value))
        else:
            self._authed.clear()

    @classmethod
    @override
    # API-key authentication returns the base provider, not the OAuth subclass.
    def from_key(  # ty: ignore[invalid-method-override] -- base returns Self; API keys cannot construct OAuth providers
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
            f"{_AUTHORIZE_URL}"
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
            data: MutableJSON = cast(MutableJSON, r.json())

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
            # Anchor on the JWT ``exp`` claim, matching ``load`` and ``_refresh``.
            # A local-clock ``time.time() + expires_in`` value would disagree with
            # what a later ``load`` reads back from the same token, forcing a
            # spurious disk-adopt/refresh cycle under clock skew. The issuer's
            # ``exp`` is the single authoritative expiry source.
            expires_at=_jwt_exp(access_token),
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

    @override
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
          UnknownModelError: ``model_id`` is not in ``CAPABILITIES``.
          UnsupportedTagError: The id asks for a tag this model or
              transport does not offer.

        """
        mid = model_id if model_id is not None else "default"
        # ``+1m`` is absent from the narrowed subscription catalog, so a
        # ``+1m`` id raises rather than silently clamping: the suffix buys
        # nothing under the subscription wire contract.
        capability, settings = resolve(
            mid, models=self.CAPABILITIES, roles=self.ROLES, transport=self.TRANSPORT
        )
        return _OpenAISubModel(
            provider=self,
            capability=capability,
            settings=settings,
        )

    @override
    def utility_model(self) -> _OpenAISubModel:
        """Return the default utility (fast/cheap) model backend.

        Returns:
          model: Utility model backend.

        """
        return self.model("utility")

    # -- Token management ----------------------------------------------

    @property
    def expired(self) -> bool:
        """True if the access token is within 5 min of expiry."""
        return time.time() > self._expires_at - self._refresh_buffer_sec

    @override
    async def get_sdk(self) -> openai.AsyncOpenAI:
        """Return OAuth-authed SDK client, refreshing as needed.

        Returns:
          client: Authenticated ``AsyncOpenAI`` SDK client.

        """
        token = await self._ensure_valid()
        cached = self._authed.peek()
        if cached is not None and cached[1] == token:
            return cached[0]
        async with self._lock.get():
            if self.expired:
                await self._refresh()
            token = self._access_token
            cached = self._authed.peek()
            if cached is not None and cached[1] == token:
                return cached[0]
            sdk = openai.AsyncOpenAI(
                api_key=token,
                base_url=_BASE_URL,
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

    @override
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

    async def _adopt_fresher_disk_creds(self) -> bool:
        """Adopt a DIFFERENT, still-valid sibling-written disk token.

        A concurrent process holding the same account may have refreshed and
        written newer creds to disk. Load them under the already-held file lock
        and adopt only when the on-disk access token (a) DIFFERS from ours and
        (b) is itself non-expired. The boolean answers exactly "did we just adopt
        a token we can use without a network refresh?" -- never "is our current
        token valid?".

        Two failure modes this guards, both of which return False so the caller
        refreshes:
          - disk token MATCHES ours: a sibling did not refresh; our token is the
            one that needs replacing (especially under ``handle_auth_error``,
            where the server already rejected it regardless of the local clock).
          - disk token DIFFERS but is itself already expired (its own refresh
            aged out, or clock skew): adopting it would 401 again next call.

        ``load`` raises ``KeyError`` on a missing-field file and ``ValueError``
        on malformed JSON; both mean "no usable disk creds", so both return False.

        Disk I/O runs in a worker thread so the event loop is not blocked on a
        slow/NFS read while the credential lock is held. Adopting a new token
        invalidates the cached SDK; the stale client is closed (not merely
        dropped) so its pooled HTTP connections are released immediately rather
        than orphaned until GC.

        Returns:
          adopted: True iff a different, non-expired disk token is now in memory.

        """
        try:
            creds = await asyncio.to_thread(
                OpenAISubscription.load, account=self._account
            )
            disk_at = creds["access_token"]
        except (FileNotFoundError, ValueError, KeyError):
            return False
        if disk_at == self._access_token:
            return False
        self._access_token = disk_at
        self._refresh_token = creds["refresh_token"]
        self._account_id = creds["account_id"]
        self._expires_at = creds["expires_at"]
        await self._discard_sdk()
        return not self.expired

    async def _discard_sdk(self) -> None:
        """Drop and close the cached SDK so its pooled connections release.

        Idempotent. The caller must already hold ``self._lock``.
        """
        old = self._sdk
        self._sdk = None
        self._sdk_token = None
        if old is not None:
            await old.close()

    async def _ensure_valid(self) -> str:
        """Return a valid access token, reloading from disk or refreshing.

        Holds a cross-process file lock around the read-disk → maybe-
        POST → write-disk sequence so concurrent processes can't both
        consume the same refresh_token and have one revoked by the
        OAuth endpoint's rotation rule.
        """
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
        logger.debug("Refreshing OpenAI OAuth token.")
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
            data: MutableJSON = cast(MutableJSON, r.json())
        self._access_token = str(data["access_token"])
        self._refresh_token = str(data["refresh_token"])
        # Anchor expiry on the JWT ``exp`` claim, matching ``load``. Deriving it
        # from ``time.time() + expires_in`` (the local clock) would disagree
        # with the value a later ``load`` reads back, so under clock skew the
        # in-memory and on-disk deadlines drift. The JWT issuer's ``exp`` is the
        # single authoritative source.
        self._expires_at = _jwt_exp(self._access_token)
        try:
            creds = OpenAISubscription.load(account=self._account)
        except FileNotFoundError:
            creds = OpenAISubscription.Credentials(
                access_token="",
                refresh_token="",
                account_id=self._account_id,
                expires_at=0.0,
            )
        except (ValueError, KeyError):
            # Unusable creds file: malformed JSON (``ValueError``) or valid JSON
            # missing required fields like ``{"tokens": {}}`` (``KeyError`` from
            # ``load``'s indexing). ``save`` below overwrites it with the
            # freshly-refreshed tokens; log first so the corruption is not
            # silently erased. Matches the swallow set in
            # ``_adopt_fresher_disk_creds``.
            logger.warning(
                "OpenAI creds file at %s was unreadable; overwriting with "
                "refreshed tokens",
                credentials_path(_default_credentials_path(), self._account),
            )
            creds = OpenAISubscription.Credentials(
                access_token="",
                refresh_token="",
                account_id=self._account_id,
                expires_at=0.0,
            )
        creds["access_token"] = self._access_token
        creds["refresh_token"] = self._refresh_token
        creds["expires_at"] = self._expires_at
        OpenAISubscription.save(creds, account=self._account)

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
                "OAuth credentials."
            )
        raw = cast(MutableJSON, decoded)
        raw_tokens = raw.get("tokens")
        if not isinstance(raw_tokens, dict):
            auth_mode = raw.get("auth_mode")
            if auth_mode in ("apikey", "api_key") or "OPENAI_API_KEY" in raw:
                raise _CredentialFileError(
                    f"{p} contains OpenAI API-key credentials, not ChatGPT "
                    "subscription OAuth credentials. Use --provider OpenAI "
                    "--auth env, or run `sagent --provider OpenAISubscription "
                    "login` first."
                )
            raise _CredentialFileError(
                f"{p} does not contain OpenAI subscription OAuth tokens. Run "
                "`sagent --provider OpenAISubscription login`."
            )
        tokens = cast(MutableJSON, raw_tokens)
        required = ("access_token", "refresh_token", "account_id")
        missing = [
            name
            for name in required
            if not isinstance(tokens.get(name), str) or not tokens.get(name)
        ]
        if missing:
            raise _CredentialFileError(
                f"{p} is missing required OAuth fields: {', '.join(missing)}. "
                "Run `sagent --provider OpenAISubscription login`."
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
        existing: MutableJSON = {}
        if p.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                decoded: object = json.loads(p.read_text(encoding="utf-8"))
                # Shape-checked, not just parse-checked: valid JSON of the
                # wrong shape (a list, or ``tokens`` as a list) would raise
                # below and leave the caller unable to persist a re-login.
                if isinstance(decoded, dict):
                    existing = cast(MutableJSON, decoded)
        raw_tokens = existing.get("tokens")
        tokens: MutableJSON = (
            cast(MutableJSON, raw_tokens) if isinstance(raw_tokens, dict) else {}
        )
        tokens["access_token"] = creds["access_token"]
        tokens["refresh_token"] = creds["refresh_token"]
        tokens["account_id"] = creds["account_id"]
        id_token = creds.get("id_token")
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


def _jwt_payload(token: str) -> dict[str, object]:
    """Decode JWT payload without verification."""
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    raw = parts[1]
    raw += "=" * (4 - len(raw) % 4)
    try:
        return DictCodec.coerce(json.loads(base64.urlsafe_b64decode(raw)))
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return {}


def _jwt_exp(token: str) -> float:
    """Extract ``exp`` claim from a JWT without verification."""
    return FloatCodec.coerce(_jwt_payload(token).get("exp"))


def _jwt_claim(token: str, namespace: str, key: str) -> str:
    """Extract a nested claim from a JWT namespace object."""
    return str(DictCodec.coerce(_jwt_payload(token).get(namespace)).get(key, ""))
