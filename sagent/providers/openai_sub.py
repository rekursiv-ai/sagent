"""OpenAI subscription provider (OAuth + Codex CLI billing).

Loads OAuth credentials from ``~/.codex/auth.json`` (written by
``codex login``) and uses the OpenAI Responses API via the official
Python SDK. Subscription billing flows through the user's ChatGPT
plan.

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

from collections.abc import AsyncIterable, Callable, Mapping
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    ClassVar,
    NotRequired,
    Protocol,
    TypedDict,
    cast,
    override,
)
from urllib.parse import quote_plus

import asyncio
import base64
import contextlib
import dataclasses
import inspect
import json
import logging
import secrets
import sys
import time


if TYPE_CHECKING:
    from openai.lib.streaming.responses import AsyncResponseStream
    from openai.types.responses.response_input_param import FunctionCallOutput

    import httpx
    import openai
    import openai.types.responses as oai_responses

    import sagent.lib.image as image_lib
else:
    from wrapt import lazy_import

    httpx = lazy_import("httpx")  # 100ms cold
    openai = lazy_import("openai")  # 493ms cold
    oai_responses = lazy_import("openai.types.responses")
    image_lib = lazy_import("sagent.lib.image")

from sagent.lib import debug_log
from sagent.lib.atomic_file import atomic_write_bytes
from sagent.lib.custom_json import MutableJSON, int_val, json_unfreeze
from sagent.providers.lib.cost import (
    ModelProfile,
    Pricing,
    compute_cost,
)
from sagent.providers.lib.errors import (
    StreamingResponseNotReadError,
    error_status_code,
    find_response_not_read,
    is_request_too_large,
    raise_if_request_too_large,
)
from sagent.providers.lib.id_remap import IdRemapper
from sagent.providers.lib.oauth import (
    AuthCodeListener,
    credential_file_lock,
    credentials_path,
    parse_manual_auth_code,
    pkce_pair,
)
from sagent.providers.lib.stop_reason import normalize_stop_reason
from sagent.providers.openai import OpenAI, _OpenAIModel
from sagent.providers.openai_compat import (
    openai_responses_reasoning_effort,
)
from sagent.types.exceptions import (
    AuthRefreshError,
    UserFacingError,
)
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    PromptTooLongError,
    StreamInterruptedError,
    TokenCount,
    base_model_id,
    strip_latency_tags,
)
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    BytesMessage,
    ModelResponsePartial,
    ModelResponseThinking,
    RuntimeEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)
from sagent.types.tools import Tool


logger = logging.getLogger(__name__)

# OAuth/protocol constants below mirror OpenAI's open-source Codex CLI
# (Apache-2.0); see this module's NOTICE block for full citations.
#   CLIENT_ID:   codex-rs/login/src/auth/manager.rs:921
#   token URL:   codex-rs/login/src/auth/manager.rs:94
#   issuer:      codex-rs/login/src/server.rs:51
#   scope:       codex-rs/login/src/server.rs:493 (verbatim match)
#   base URL:    codex-rs/model-provider-info/src/lib.rs:237
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_TOKEN_URL = "https://auth.openai.com/oauth/token"  # noqa: S105 -- not a secret; OAuth endpoint URL
_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_CREDENTIALS_PATH = Path.home() / ".codex" / "auth.json"
_SCOPES = (
    "openid profile email offline_access api.connectors.read api.connectors.invoke"
)
DEFAULT_REFRESH_BUFFER_SEC = 300.0
# OAuth-registered redirect URI port. Not user-tunable: the OAuth app
# fingerprinted as Codex CLI has this port baked into its allowed
# redirects on OpenAI's side.
_CALLBACK_PORT = 1455
# Codex's subscription backend currently exposes a smaller practical
# context window than the public API model metadata. Local budgeting must
# plan against the wire contract so auto-compaction runs before the backend
# rejects an oversized request.
_SUBSCRIPTION_MAX_REQUEST_TOKENS = 272_000
_SUBSCRIPTION_MAX_RESPONSE_TOKENS = 32_000
_STREAM_IDLE_TIMEOUT = 600.0

# Must stay identical to ``_OpenAIModel._is_effort_model``'s prefix set
# (openai.py): the two OpenAI transports share one ``KNOWN_MODELS`` catalog and
# the wire-mapping docstring requires them to agree on which ids are reasoning
# models. A bare ``"o"`` over-matches (e.g. a future ``omni-*`` id), diverging
# from the API-key path -- pin the exact reasoning families instead.
_EFFORT_PREFIXES = ("o1", "o3", "o4", "gpt-5")

_FINISH_MAP: dict[str, str] = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "stop",
}


class _CredentialFileError(ValueError):
    """Stored credentials do not match the subscription OAuth schema."""


def _subscription_profile(profile: ModelProfile) -> ModelProfile:
    """Clamp the public API profile to the subscription token contract.

    Only the token windows differ between the API and the Codex
    subscription backend, so only they are clamped. The image and wire
    byte caps are a property of the underlying OpenAI model, identical
    across the two auth paths -- inherit them from ``profile`` so a future
    per-model divergence in the parent flows through automatically instead
    of being silently overwritten by a stale local constant.
    """
    return dataclasses.replace(
        profile,
        max_request_tokens=min(
            profile.max_request_tokens,
            _SUBSCRIPTION_MAX_REQUEST_TOKENS,
        ),
        max_response_tokens=min(
            profile.max_response_tokens,
            _SUBSCRIPTION_MAX_RESPONSE_TOKENS,
        ),
    )


class OpenAISubscription(OpenAI):
    """OpenAI provider -- OAuth + ChatGPT subscription billing.

    Derives KNOWN_MODELS from OpenAI (API pricing inherited, token
    limits clamped to the subscription wire contract, ``+1m`` ids
    dropped -- see the ``KNOWN_MODELS`` comprehension below).
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

    # Keep pricing inherited from the public API profiles, but clamp limits
    # separately because subscription auth is a different backend contract.
    KNOWN_MODELS: ClassVar[dict[str, ModelProfile]] = {
        name: _subscription_profile(profile)
        for name, profile in OpenAI.KNOWN_MODELS.items()
        if not name.endswith("+1m")
    }

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
        self._sdk: openai.AsyncOpenAI | None = None
        self._sdk_token: str | None = None
        self._lock = asyncio.Lock()

    @classmethod
    @override
    def from_key(cls, api_key: str, *, base_url: str | None = None) -> OpenAI:
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
          account: Named credential slot. ``None`` uses the legacy
            ``~/.codex/auth.json`` path.
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
          account: Named credential slot. ``None`` writes to the legacy
            ``~/.codex/auth.json`` path.
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

        with httpx.Client() as http:
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

        access_token = cast(str, data["access_token"])
        account_id = _jwt_claim(
            access_token,
            "https://api.openai.com/auth",
            "chatgpt_account_id",
        )
        creds = cls.Credentials(
            access_token=access_token,
            refresh_token=cast(str, data["refresh_token"]),
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
    def model(  # ty: ignore[invalid-method-override] -- narrowed return type
        self,
        model_id: str | None = None,
        /,
        max_request_tokens: int | None = None,
    ) -> _OpenAISubModel:
        """Create a Responses API model backend.

        Args:
          model_id: Model ID. ``None`` uses ``DEFAULT_MODEL``.
          max_request_tokens: Override max input tokens. ``None`` uses profile default.

        Returns:
          model: Responses API model backend.

        Raises:
          ValueError: If ``model_id`` is not in ``KNOWN_MODELS``.

        """
        mid = model_id if model_id is not None else self.DEFAULT_MODEL
        # Fail fast -- every supported model must be in KNOWN_MODELS. Strip
        # latency tags before lookup. ``+1m`` is NOT stripped and is absent
        # from the narrowed subscription catalog, so a ``+1m`` id raises below
        # rather than silently clamping (the suffix buys nothing under the
        # subscription wire contract). Callers must use the base id.
        profile = self.KNOWN_MODELS.get(mid) or self.KNOWN_MODELS.get(
            strip_latency_tags(mid),
        )
        if profile is None:
            known = ", ".join(sorted(self.KNOWN_MODELS))
            raise ValueError(
                f"Unknown model {mid!r} for {type(self).__name__}."
                f" Known models: {known}",
            )
        return _OpenAISubModel(
            provider=self,
            model_id=mid,
            profile=profile,
            max_request_tokens=min(
                max_request_tokens
                if max_request_tokens is not None
                else profile.max_request_tokens,
                profile.max_request_tokens,
            ),
        )

    @override
    def utility_model(self) -> _OpenAISubModel:
        """Return the default utility (fast/cheap) model backend.

        Returns:
          model: Utility model backend.

        """
        return self.model(self.DEFAULT_UTILITY_MODEL)

    # -- Token management ----------------------------------------------

    @property
    def expired(self) -> bool:
        """True if the access token is within 5 min of expiry."""
        return time.time() > self._expires_at - self._refresh_buffer_sec

    async def get_sdk(self) -> openai.AsyncOpenAI:
        """Return OAuth-authed SDK client, refreshing as needed.

        Returns:
          client: Authenticated ``AsyncOpenAI`` SDK client.

        """
        token = await self._ensure_valid()
        if self._sdk is not None and token == self._sdk_token:
            return self._sdk
        async with self._lock:
            if self.expired:
                await self._refresh()
            token = self._access_token
            if self._sdk is not None and token == self._sdk_token:
                return self._sdk
            old = self._sdk
            self._sdk = openai.AsyncOpenAI(
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
            self._sdk_token = token
            if old is not None:
                await old.close()
            return self._sdk

    async def close_sdk(self) -> None:
        """Close and clear the shared OAuth SDK client.

        Idempotent: a no-op when no SDK has been created. Called from
        ``_OpenAISubModel.close`` (the model delegates teardown of the
        provider-owned SDK).
        """
        async with self._lock:
            if self._sdk is None:
                return
            sdk = self._sdk
            self._sdk = None
            self._sdk_token = None
        await sdk.close()

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
        cred_path = credentials_path(DEFAULT_CREDENTIALS_PATH, self._account)
        async with self._lock, credential_file_lock(cred_path):
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
        cred_path = credentials_path(DEFAULT_CREDENTIALS_PATH, self._account)
        async with self._lock, credential_file_lock(cred_path):
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
        async with httpx.AsyncClient() as http:
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
                # shows actionable text instead of an httpx traceback.
                raise AuthRefreshError(
                    "OpenAI Codex subscription session expired or revoked. "
                    "Run /login to re-authenticate, or /model to switch "
                    "providers.",
                )
            r.raise_for_status()
            data: MutableJSON = cast(MutableJSON, r.json())
        self._access_token = cast(str, data["access_token"])
        self._refresh_token = cast(str, data["refresh_token"])
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
                credentials_path(DEFAULT_CREDENTIALS_PATH, self._account),
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
          account: Named credential slot. ``None`` reads the legacy
            ``~/.codex/auth.json`` path.

        Returns:
          creds: Loaded OAuth credentials.

        Raises:
          FileNotFoundError: If no credentials file exists.
          ValueError: If the file is not a complete OAuth credential record.

        """
        p = path or credentials_path(DEFAULT_CREDENTIALS_PATH, account)
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
          account: Named credential slot. ``None`` writes to the legacy
            ``~/.codex/auth.json`` path.

        """
        p = path or credentials_path(DEFAULT_CREDENTIALS_PATH, account)
        existing: MutableJSON = {}
        if p.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                existing = json.loads(p.read_text(encoding="utf-8"))
        tokens = cast(MutableJSON, existing.get("tokens", {}))
        tokens["access_token"] = creds["access_token"]
        tokens["refresh_token"] = creds["refresh_token"]
        tokens["account_id"] = creds["account_id"]
        id_token = creds.get("id_token")
        if id_token:
            tokens["id_token"] = id_token
        existing["auth_mode"] = "chatgpt"
        existing["tokens"] = tokens
        atomic_write_bytes(p, json.dumps(existing).encode(), file_mode=0o600)


class _OpenAISubModel(_OpenAIModel):
    """OpenAI Responses API model -- subscription billing."""

    _provider: OpenAISubscription  # type: narrowed from OpenAICompat

    @override
    async def close(self) -> None:
        """Release the provider's Responses-API SDK and the HTTP client.

        The base ``OpenAICompatModel.close`` tears down the shared
        ``httpx.AsyncClient``; the subscription path additionally has an
        ``AsyncOpenAI`` SDK owned by the *provider* (not the model), so
        delegate to ``provider.close_sdk()`` for that.
        """
        await self._provider.close_sdk()
        await super().close()

    @override
    def _is_effort_model(self, model_id: str) -> bool:
        """True for OpenAI reasoning models that accept ``reasoning_effort``."""
        return any(model_id.startswith(p) for p in _EFFORT_PREFIXES)

    @property
    @override
    def supports_thinking(self) -> bool:
        """Whether the model accepts a thinking/reasoning request."""
        return self.supports_effort

    @property
    @override
    def valid_service_tiers(self) -> tuple[str, ...]:
        """Codex ``/fast`` slash command sets ``service_tier="priority"``."""
        return ("priority",)

    @property
    @override
    def valid_latency_modes(self) -> tuple[str, ...]:
        """``latency="fast"`` maps to ``service_tier="priority"``.

        Mirrors the Codex ``/fast`` slash command, which selects the
        ``priority`` processing tier. Unlike Anthropic fast mode (a
        separate ``speed="fast"`` inference-acceleration field), OpenAI's
        fast path is purely a queue-priority tier.
        """
        return ("fast",)

    @property
    @override
    def supports_account_auth(self) -> bool:
        """Whether the provider uses account authentication."""
        return True

    @override
    def is_context_overflow(self, error: Exception) -> bool:
        """Check whether an error indicates token context-window overflow.

        Excludes the byte wire-limit via the shared classifier first, so a
        413 byte error routes to byte-overflow recovery -- uniform with the
        compat/google siblings -- rather than being mis-read as token
        overflow because its body incidentally names the context window.

        Args:
          error: Exception raised by the API call.

        Returns:
          overflow: ``True`` if the error is a token context-length overflow.

        """
        if is_request_too_large(error_status_code(error), str(error)):
            return False
        msg = str(error).lower()
        return (
            "context_length_exceeded" in msg
            or "maximum context length" in msg
            or "exceeds the context window" in msg
        )

    @override
    def is_retryable_provider_error(self, error: Exception) -> bool:
        """OpenAI Responses API can return statusless transient errors.

        The Codex subscription endpoint occasionally raises
        ``openai.APIError`` with ``status_code=None`` and a body
        containing the "An error occurred ... You can retry your
        request" copy. The shared status-code classifier in
        ``retry.py`` misses these because the status is absent;
        match on the SDK's canonical retry-hint substring instead.
        """
        return "you can retry" in str(error).lower()

    @override
    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Send a buffered request via the streaming path.

        Args:
          request: Model request to send.

        Returns:
          response: Translated model response.

        """
        return await self.stream(request, None)

    @override
    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        """Stream a request via the OpenAI Responses API.

        Args:
          request: Model request to send.
          publish: Callback invoked with each runtime event as it arrives.

        Returns:
          response: Assembled model response after the stream closes.

        """
        sdk = await self._provider.get_sdk()
        # The ChatGPT Codex subscription endpoint rejects some public
        # Responses API knobs, including ``temperature`` and
        # ``max_output_tokens``.
        reasoning_effort = self._reasoning_effort(request)
        # The Responses API only streams reasoning-text deltas when a
        # ``summary`` is requested; without it the model reasons silently
        # and ``on_thinking`` never fires. Request the auto summary whenever
        # thinking is on so the reasoning surface is actually populated.
        # (OpenAI never returns raw reasoning, only summaries; ``auto`` is
        # the broadest tier and is chosen by the model.)
        reasoning_dict: dict[str, str] | None = None
        if reasoning_effort is not None:
            reasoning_dict = {"effort": reasoning_effort, "summary": "auto"}
            if base_model_id(self._model_id).startswith("gpt-5.6"):
                # Sagent replays complete local history with store=False, so
                # GPT-5.6 must render reasoning across all supplied turns.
                reasoning_dict["context"] = "all_turns"
        reasoning: dict[str, str] | openai.Omit = reasoning_dict or openai.omit
        create_kwargs: dict[str, object] = {
            "model": self._wire_model_id,
            "input": _build_input(
                request,
                max_image_dim=self.max_image_dim,
                max_image_bytes=self.max_image_bytes,
            ),
            "instructions": request.system or "",
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"]
            if self.supports_effort
            else openai.omit,
            "tools": _build_tools(request.tools) if request.tools else openai.omit,
            "reasoning": reasoning,
        }
        tier = self.effective_service_tier(request)
        if tier is not None:
            create_kwargs["service_tier"] = tier
        debug_log.trace(
            "api_call",
            kind="openai_responses",
            model=self._wire_model_id,
            latency=request.latency,
            service_tier=tier,
        )
        try:
            try:
                event_stream: AsyncResponseStream = cast(
                    "AsyncResponseStream",
                    await sdk.responses.create(**create_kwargs),  # pyright: ignore[reportCallIssue, reportArgumentType]  # ty: ignore[no-matching-overload] -- dynamic kwargs; SDK overload resolution fails on stream=True literal
                )
            except openai.AuthenticationError:
                # In-memory bearer was revoked server-side between our
                # local expiry check and the request arriving. Reload
                # disk creds (or force-refresh) and retry once.
                await self._provider.handle_auth_error()
                sdk = await self._provider.get_sdk()
                event_stream = cast(
                    "AsyncResponseStream",
                    await sdk.responses.create(**create_kwargs),  # pyright: ignore[reportCallIssue, reportArgumentType]  # ty: ignore[no-matching-overload] -- dynamic kwargs; SDK overload resolution fails on stream=True literal
                )
            return await _consume_stream(
                event_stream,
                pricing=self._profile.pricing,
                publish=publish,
            )
        except Exception as exc:
            not_read = find_response_not_read(exc)
            if not_read is not None:
                raise StreamingResponseNotReadError(
                    provider_name="OpenAI subscription",
                    cause=not_read,
                ) from exc
            raise_if_request_too_large(error_status_code(exc), str(exc), cause=exc)
            if self.is_context_overflow(exc):
                # The compactor's shrink-and-retry path keys off
                # PromptTooLongError; leaking raw SDK errors makes outer
                # recovery retry unchanged history instead of dropping groups.
                raise PromptTooLongError(str(exc)) from exc
            raise

    def _reasoning_effort(self, request: ModelRequest) -> str | None:
        """Map sagent thinking/effort knobs onto OpenAI reasoning effort."""
        if not self.supports_effort:
            return None
        if request.effort is not None:
            return (
                openai_responses_reasoning_effort(self._model_id, request.effort)
                or "high"
            )
        if request.thinking == "adaptive":
            return "medium"
        if request.thinking == "enabled":
            return "high"
        return None


def _build_tools(
    tools: list[Tool],
) -> list[oai_responses.FunctionToolParam]:
    """Translate each ``Tool`` into a Responses API function-tool param."""
    return [_build_tool(t) for t in tools]


def _build_tool(tool: Tool) -> oai_responses.FunctionToolParam:
    """Wire-shape a single ``Tool`` for the Responses API."""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": cast(
            dict[str, object] | None, json_unfreeze(tool.directive_schema)
        ),
        "strict": None,
    }


def _build_input(
    request: ModelRequest,
    *,
    max_image_dim: int = 0,
    max_image_bytes: int = 0,
) -> oai_responses.ResponseInputParam:
    """Convert history entries to Responses API input items.

    Args:
      request: Fully-built model request.
      max_image_dim: Maximum image dimension (pixels); larger inputs are
          resized before encoding. ``0`` (the default) means no cap; the
          live caller passes the model profile's ``max_image_dim``.
      max_image_bytes: Maximum image size in bytes after resize. ``0`` (the
          default) means no cap.

    Returns:
      items: Responses API input items in send order.

    """
    ids = IdRemapper("fc_")
    items: oai_responses.ResponseInputParam = []
    for entry in request.messages:
        if isinstance(entry, (AgentSendMessage, UserMessage)):
            items.append(
                _build_user_item(entry, max_image_dim, max_image_bytes),
            )
        elif isinstance(entry, AssistantMessage):
            _build_assistant_items(entry, items, ids)
        else:
            items.append(_build_tool_result_item(entry, ids))
    return items


def _build_user_item(
    entry: AgentSendMessage | UserMessage,
    max_image_dim: int,
    max_image_bytes: int,
) -> oai_responses.EasyInputMessageParam:
    """Wire-shape a user-side entry, inlining any image attachments."""
    image_atts = [att for att in entry.attachments if _is_image_attachment(att)]
    non_image_atts = [att for att in entry.attachments if not _is_image_attachment(att)]
    for att in non_image_atts:
        logger.warning(
            "OpenAI subscription: skipping non-image attachment (mime=%s)",
            att.descriptor,
        )
    if not image_atts:
        return {"role": "user", "content": entry.text}
    blocks: list[oai_responses.ResponseInputContentParam] = []
    if entry.text:
        blocks.append({"type": "input_text", "text": entry.text})
    for att in image_atts:
        raw, mime = image_lib.resize(
            att.data, max_dim=max_image_dim, max_bytes=max_image_bytes
        )
        b64 = base64.b64encode(raw).decode()
        blocks.append(
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:{mime};base64,{b64}",
            }
        )
    return {"role": "user", "content": blocks}


def _is_image_attachment(att: BytesMessage) -> bool:
    """True for ``BytesMessage`` attachments whose descriptor is an image mime."""
    return att.descriptor.startswith("image/")


def _build_assistant_items(
    entry: AssistantMessage,
    items: oai_responses.ResponseInputParam,
    ids: IdRemapper,
) -> None:
    """Expand an AssistantMessage into assistant + function_call items."""
    for block in entry.thinking_blocks:
        if block.get("type") == "reasoning" and isinstance(
            block.get("encrypted_content"), str
        ):
            # OpenAI documents replaying the reasoning output item verbatim in
            # stateless mode. Copy the opaque mapping so persisted session data
            # never mutates while the SDK serializes the request.
            items.append(cast("oai_responses.ResponseReasoningItemParam", dict(block)))
    if entry.text:
        items.append({"role": "assistant", "content": entry.text})
    for tc in entry.tool_calls:
        native_id = ids.map(tc.id)
        items.append(
            {
                "type": "function_call",
                "id": native_id,
                "call_id": native_id,
                "name": tc.name,
                "arguments": json.dumps(dict(tc.args)),
                "status": "completed",
            }
        )


def _build_tool_result_item(
    entry: ToolResult,
    ids: IdRemapper,
) -> FunctionCallOutput:
    """Wire-shape a ``ToolResult`` as a Responses API function_call_output."""
    native_id = ids.map(entry.call_id)
    output = entry.content
    if entry.is_error and output:
        output = f"[Error] {output}"
    return {
        "type": "function_call_output",
        "call_id": native_id,
        "output": output,
        "status": "completed",
    }


class _UsageDetails(Protocol):
    """Minimum SDK usage-details surface needed for forward-compatible fields."""

    def to_dict(self) -> dict[str, object]: ...


def _terminal_metadata(response: object) -> tuple[str, str, int, int, int, int]:
    """Extract terminal response metadata, including SDK-extra usage fields."""
    message_id = str(getattr(response, "id", "") or "")
    status = str(getattr(response, "status", "") or "")
    usage = getattr(response, "usage", None)
    if usage is None:
        return message_id, status, 0, 0, 0, 0
    input_tokens = int_val(getattr(usage, "input_tokens", None), 0)
    output_tokens = int_val(getattr(usage, "output_tokens", None), 0)
    cache_read = 0
    cache_write = 0
    details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        # OpenAI SDK 2.44 predates the typed ``cache_write_tokens`` field, but
        # Stainless models preserve unknown response fields and expose them
        # through ``to_dict``. Reading the mapping keeps new usage data without
        # an Any cast or waiting for an SDK schema release.
        raw_details = cast(_UsageDetails, details).to_dict()
        cache_read = int_val(raw_details.get("cached_tokens"), 0)
        cache_write = int_val(raw_details.get("cache_write_tokens"), 0)
    return (
        message_id,
        status,
        input_tokens,
        output_tokens,
        cache_read,
        cache_write,
    )


_STREAM_ERROR_STATUS: dict[str, int] = {
    "rate_limit": 429,
    "rate_limit_error": 429,
    "rate_limit_exceeded": 429,
    "api_error": 500,
    "overloaded_error": 529,
    "server_error": 500,
}


class _OpenAIStreamError(UserFacingError):
    """In-band Responses error retaining retry/rate-limit classification data."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        normalized_code = code or "unknown_error"
        normalized_type = (
            "rate_limit_error"
            if normalized_code in {"rate_limit", "rate_limit_exceeded"}
            else normalized_code
        )
        self.status_code = _STREAM_ERROR_STATUS.get(normalized_code)
        self.body: Mapping[str, object] = {
            "type": "error",
            "error": {
                "type": normalized_type,
                "code": normalized_code,
                "message": message,
                "param": param,
            },
        }


async def _consume_stream(
    stream: AsyncIterable[object],
    *,
    pricing: Pricing,
    publish: Callable[[RuntimeEvent], None] | None,
) -> ModelResponse:
    """Parse a Responses API event stream into an assembled ModelResponse."""
    text_parts: list[str] = []
    refusal_parts: list[str] = []
    thinking_parts: list[str] = []
    encrypted_reasoning: list[Mapping[str, object]] = []
    tool_calls: list[ToolCall] = []
    tool_args: dict[str, list[str]] = {}
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_write = 0
    message_id = ""
    finish_reason: str | None = None
    completed = False

    loop = asyncio.get_running_loop()
    deadline = loop.time() + _STREAM_IDLE_TIMEOUT
    # Single cleanup rule: every path that does not return a terminal response
    # closes the stream. A mid-stream error event (raises below), an idle
    # timeout / cancellation, and the no-completion fallthrough all leave the
    # SSE connection open otherwise -- a per-error connection leak. ``completed``
    # gates the one exit that legitimately keeps reading until done.
    try:
        async with asyncio.timeout_at(deadline) as watchdog:
            async for event in stream:
                watchdog.reschedule(loop.time() + _STREAM_IDLE_TIMEOUT)
                if isinstance(event, oai_responses.ResponseRefusalDeltaEvent):
                    refusal_parts.append(event.delta)
                    text_parts.append(event.delta)
                    if publish is not None:
                        publish(ModelResponsePartial(event.delta))
                elif isinstance(event, oai_responses.ResponseRefusalDoneEvent):
                    # Normally deltas already contain the full refusal. Keep a
                    # done-only stream useful and append only an unseen suffix.
                    seen = "".join(refusal_parts)
                    suffix = (
                        event.refusal[len(seen) :]
                        if event.refusal.startswith(seen)
                        else ""
                    )
                    if not seen:
                        suffix = event.refusal
                    if suffix:
                        refusal_parts.append(suffix)
                        text_parts.append(suffix)
                        if publish is not None:
                            publish(ModelResponsePartial(suffix))
                elif isinstance(event, oai_responses.ResponseTextDeltaEvent):
                    text_parts.append(event.delta)
                    if publish is not None:
                        publish(ModelResponsePartial(event.delta))
                elif isinstance(
                    event,
                    (
                        oai_responses.ResponseReasoningTextDeltaEvent,
                        oai_responses.ResponseReasoningSummaryTextDeltaEvent,
                    ),
                ):
                    thinking_parts.append(event.delta)
                    if publish is not None:
                        publish(ModelResponseThinking(event.delta))
                elif isinstance(
                    event,
                    oai_responses.ResponseFunctionCallArgumentsDeltaEvent,
                ):
                    tool_args.setdefault(event.item_id, []).append(event.delta)
                elif isinstance(event, oai_responses.ResponseOutputItemDoneEvent):
                    item = event.item
                    if item.type == "function_call":
                        tc_id = str(item.call_id or "")
                        tc_name = str(item.name or "")
                        item_id = item.id or ""
                        delta_args = "".join(tool_args.get(item_id, []))
                        done_args = str(item.arguments or "")
                        args = _parse_tool_arguments(
                            delta_args,
                            done_args,
                            tool_name=tc_name,
                            call_id=tc_id,
                        )
                        tool_calls.append(
                            ToolCall(
                                id=tc_id,
                                name=tc_name,
                                args=cast(Mapping[str, object], args),
                            )
                        )
                    elif item.type == "reasoning":
                        reasoning_item = item.to_dict(exclude_none=True)
                        if isinstance(reasoning_item.get("encrypted_content"), str):
                            encrypted_reasoning.append(reasoning_item)
                elif isinstance(event, oai_responses.ResponseErrorEvent):
                    raise _openai_stream_event_error(event)
                elif isinstance(event, oai_responses.ResponseFailedEvent):
                    raise _openai_stream_response_error(event.response)
                elif isinstance(
                    event,
                    (
                        oai_responses.ResponseCompletedEvent,
                        oai_responses.ResponseIncompleteEvent,
                    ),
                ):
                    resp = event.response
                    (
                        message_id,
                        finish_reason,
                        input_tokens,
                        output_tokens,
                        cache_read,
                        cache_write,
                    ) = _terminal_metadata(resp)
                    if finish_reason == "incomplete":
                        incomplete = getattr(resp, "incomplete_details", None)
                        if getattr(incomplete, "reason", None) == "content_filter":
                            finish_reason = "content_filter"
                    debug_log.trace(
                        "api_response",
                        kind="openai_responses",
                        service_tier=getattr(resp, "service_tier", None),
                    )
                    # Set last: if a usage/attr read above raises on an SDK shape
                    # change, ``completed`` stays False so the ``finally`` still
                    # closes the stream instead of leaking the connection.
                    completed = True
    finally:
        if not completed:
            await _close_stream(stream)

    if refusal_parts and finish_reason in (None, "completed", "stop"):
        finish_reason = "content_filter"
    response = _build_stream_response(
        text_parts=text_parts,
        thinking_parts=thinking_parts,
        encrypted_reasoning=encrypted_reasoning,
        tool_calls=tool_calls,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_write=cache_write,
        cache_read=cache_read,
        finish_reason=finish_reason,
        message_id=message_id,
        pricing=pricing,
    )
    if not completed:
        raise StreamInterruptedError(response)
    return response


def _openai_stream_event_error(event: object) -> UserFacingError:
    code = getattr(event, "code", None)
    param = getattr(event, "param", None)
    message = getattr(event, "message", "OpenAI subscription stream error")
    details = ["OpenAI subscription stream error"]
    if isinstance(code, str) and code:
        details.append(f"code={code}")
    if isinstance(param, str) and param:
        details.append(f"param={param}")
    details.append(str(message))
    return _OpenAIStreamError(
        ": ".join(details),
        code=code if isinstance(code, str) else None,
        param=param if isinstance(param, str) else None,
    )


def _openai_stream_response_error(response: object) -> UserFacingError:
    response_id = getattr(response, "id", "")
    status = getattr(response, "status", "")
    error = getattr(response, "error", None)
    message = getattr(error, "message", None) if error is not None else None
    code = getattr(error, "code", None) if error is not None else None
    incomplete = getattr(response, "incomplete_details", None)
    reason = getattr(incomplete, "reason", None) if incomplete is not None else None
    details = ["OpenAI subscription stream ended without completion"]
    if isinstance(status, str) and status:
        details.append(f"status={status}")
    if isinstance(response_id, str) and response_id:
        details.append(f"response_id={response_id}")
    if isinstance(code, str) and code:
        details.append(f"code={code}")
    if isinstance(reason, str) and reason:
        details.append(f"reason={reason}")
    if isinstance(message, str) and message:
        details.append(message)
    return _OpenAIStreamError(
        ": ".join(details),
        code=code if isinstance(code, str) else None,
    )


def _build_stream_response(
    *,
    text_parts: list[str],
    thinking_parts: list[str],
    encrypted_reasoning: list[Mapping[str, object]],
    tool_calls: list[ToolCall],
    input_tokens: int,
    output_tokens: int,
    cache_write: int,
    cache_read: int,
    finish_reason: str | None,
    message_id: str,
    pricing: Pricing,
) -> ModelResponse:
    raw_reason = _FINISH_MAP.get(finish_reason or "", finish_reason)

    # The Responses API reports a cache-inclusive prompt total; keep ordinary,
    # cache-write, and cache-read pools disjoint in Sagent's normalized usage.
    non_cached_input = max(0, input_tokens - cache_read - cache_write)
    in_cost, out_cost, total_cost = compute_cost(
        pricing,
        non_cached_input,
        output_tokens,
        cache_creation=cache_write,
        cache_read=cache_read,
    )
    thinking_blocks: list[Mapping[str, object]] = list(encrypted_reasoning)
    if thinking_parts:
        thinking_blocks.append({"type": "reasoning", "text": "".join(thinking_parts)})
    return ModelResponse(
        message=AssistantMessage(
            text="".join(text_parts),
            thinking_blocks=tuple(thinking_blocks),
            tool_calls=tuple(tool_calls),
        ),
        tokens=TokenCount(
            input_tokens=non_cached_input,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_write,
            cache_read_tokens=cache_read,
        ),
        stop_reason=normalize_stop_reason(
            raw_reason,
            kind="openai",
            has_tool_use=bool(tool_calls),
        ),
        message_id=message_id,
        request_id=message_id,
        input_cost=in_cost,
        output_cost=out_cost,
        total_cost=total_cost,
    )


async def _close_stream(stream: object) -> None:
    """Close a Responses stream after timeout, cancellation, or stream error.

    Runs in a ``finally`` / cleanup context, so a failure HERE must never
    replace the error that triggered the close (e.g. a ``UserFacingError`` from a
    ``ResponseFailedEvent``). A broken pipe on ``aclose`` would otherwise surface
    to the user instead of the real cause -- swallow and log it.
    """
    close = getattr(stream, "aclose", None)
    if close is None:
        close = getattr(stream, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception:  # noqa: BLE001 -- cleanup must not mask the original error
        logger.debug("stream close raised during cleanup", exc_info=True)


def _parse_tool_arguments(
    delta_args: str,
    done_args: str,
    *,
    tool_name: str,
    call_id: str,
) -> MutableJSON:
    """Parse streamed function-call arguments with completed-item fallback."""
    saw_args = False
    parsed_by_source: dict[str, MutableJSON] = {}
    for source, args_str in (("delta", delta_args), ("done", done_args)):
        if not args_str:
            continue
        saw_args = True
        try:
            parsed = json.loads(args_str)
        except json.JSONDecodeError:
            logger.warning(
                "OpenAI Responses tool arguments were invalid JSON: "
                "source=%s tool=%s call_id=%s chars=%d",
                source,
                tool_name,
                call_id,
                len(args_str),
            )
            continue
        if isinstance(parsed, dict):
            parsed_obj = cast(MutableJSON, parsed)
            parsed_by_source[source] = parsed_obj
            if not parsed_obj:
                logger.warning(
                    "OpenAI Responses tool arguments were an empty JSON object: "
                    "source=%s tool=%s call_id=%s chars=%d",
                    source,
                    tool_name,
                    call_id,
                    len(args_str),
                )
            continue
        logger.warning(
            "OpenAI Responses tool arguments were not a JSON object: "
            "source=%s tool=%s call_id=%s",
            source,
            tool_name,
            call_id,
        )
    if not saw_args:
        logger.warning(
            "OpenAI Responses tool arguments were empty: tool=%s call_id=%s",
            tool_name,
            call_id,
        )
    delta = parsed_by_source.get("delta")
    done = parsed_by_source.get("done")
    if delta is not None and done is not None and delta != done:
        logger.warning(
            "OpenAI Responses tool arguments differed between delta and done: "
            "tool=%s call_id=%s delta_chars=%d done_chars=%d "
            "delta_keys=%d done_keys=%d",
            tool_name,
            call_id,
            len(delta_args),
            len(done_args),
            len(delta),
            len(done),
        )
    if done:
        logger.debug(
            "OpenAI Responses selected completed tool arguments: "
            "tool=%s call_id=%s delta_chars=%d done_chars=%d "
            "delta_keys=%d done_keys=%d",
            tool_name,
            call_id,
            len(delta_args),
            len(done_args),
            len(delta) if delta is not None else -1,
            len(done),
        )
        return done
    if delta:
        logger.debug(
            "OpenAI Responses selected streamed delta tool arguments: "
            "tool=%s call_id=%s delta_chars=%d done_chars=%d delta_keys=%d",
            tool_name,
            call_id,
            len(delta_args),
            len(done_args),
            len(delta),
        )
        return delta
    if done is not None:
        return done
    if delta is not None:
        return delta
    return {}


def _jwt_payload(token: str) -> MutableJSON:
    """Decode JWT payload without verification."""
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    raw = parts[1]
    raw += "=" * (4 - len(raw) % 4)
    try:
        return cast(MutableJSON, json.loads(base64.urlsafe_b64decode(raw)))
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
        return {}


def _jwt_exp(token: str) -> float:
    """Extract ``exp`` claim from a JWT without verification."""
    return float(cast(float, _jwt_payload(token).get("exp", 0)))


def _jwt_claim(token: str, namespace: str, key: str) -> str:
    """Extract a nested claim from a JWT namespace object."""
    ns = _jwt_payload(token).get(namespace, {})
    if isinstance(ns, dict):
        return str(cast(MutableJSON, ns).get(key, ""))
    return ""
