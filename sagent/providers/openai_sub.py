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

from collections.abc import Callable
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
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
import secrets
import sys
import time


if TYPE_CHECKING:
    from openai.lib.streaming.responses import AsyncResponseStream
    from openai.types.responses.response_input_param import FunctionCallOutput

    import httpx
    import openai
    import openai.types.responses as oai_responses
    import openai.types.shared as oai_shared
else:
    from sagent.lib.lazy_import import lazy_import

    httpx = lazy_import("httpx")  # 100ms cold
    openai = lazy_import("openai")  # 493ms cold
    oai_responses = lazy_import("openai.types.responses")
    oai_shared = lazy_import("openai.types.shared")

from sagent.custom_types import (
    Message,
    ModelRequest,
    ModelResponse,
    MultipartMessage,
    TextMessage,
    TokenCount,
    Tool,
)
from sagent.lib.atomic_file import atomic_write_bytes
from sagent.lib.json import (
    MutableJSON,
    json_freeze,
    json_unfreeze,
)
from sagent.lib.message import (
    get_directive,
    get_queue_id,
    get_tool_name,
    tool_call_message,
)
from sagent.providers.lib.cost import (
    Pricing,
    compute_cost,
)
from sagent.providers.lib.id_remap import IdRemapper
from sagent.providers.lib.oauth import (
    AuthCodeListener,
    credentials_path,
    parse_manual_auth_code,
    pkce_pair,
)
from sagent.providers.lib.stop_reason import normalize_stop_reason
from sagent.providers.openai import OpenAI, _OpenAIModel


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
_DEFAULT_PATH = Path.home() / ".codex" / "auth.json"
_SCOPES = (
    "openid profile email offline_access api.connectors.read api.connectors.invoke"
)
_REFRESH_BUFFER_SEC = 300.0
_CALLBACK_PORT = 1455

_EFFORT_PREFIXES = ("o", "gpt-5")

_FINISH_MAP: dict[str, str] = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "stop",
}


class OpenAISubscription(OpenAI):
    """OpenAI provider -- OAuth + ChatGPT subscription billing.

    Inherits KNOWN_MODELS (limits + API pricing) from OpenAI.
    Cost tracking uses standard API per-token pricing even though
    subscription users pay a flat fee. This is intentional: it gives
    a consistent "what would this session cost at API rates" metric
    regardless of auth mode.
    """

    # Most capable model -- sub users don't pay per-token.
    # OpenAI (API key) defaults to gpt-5.5 for current frontier capability.
    DEFAULT_MODEL = "gpt-5.5"
    DEFAULT_UTILITY_MODEL = "gpt-5.4-mini"

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
    ) -> None:
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._account_id = account_id  # ChatGPT account id (from JWT)
        self._account = account  # local credential slot name
        self._expires_at = expires_at
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
    ) -> OpenAISubscription:
        """Create provider from OAuth credentials.

        Args:
          creds: Pre-loaded credentials, or ``None`` to auto-load from disk.
          account: Named credential slot. ``None`` uses the legacy
            ``~/.codex/auth.json`` path.

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
        redirect_uri = f"http://127.0.0.1:{_CALLBACK_PORT}/auth/callback"
        if not manual:
            listener = AuthCodeListener(
                state,
                port=_CALLBACK_PORT,
                callback_path="/auth/callback",
            )
            try:
                listener.start()
                redirect_uri = listener.redirect_uri
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
            expires_at=time.time() + cast(float, data["expires_in"]),
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
        # Fail fast -- every supported model must be in KNOWN_MODELS.
        profile = self.KNOWN_MODELS.get(mid)
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
            max_request_tokens=(
                max_request_tokens
                if max_request_tokens is not None
                else profile.max_request_tokens
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
        return time.time() > self._expires_at - _REFRESH_BUFFER_SEC

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

    async def _ensure_valid(self) -> str:
        """Return a valid access token, refreshing if needed."""
        if not self.expired:
            return self._access_token
        async with self._lock:
            if self.expired:
                await self._refresh()
            return self._access_token

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
            r.raise_for_status()
            data: MutableJSON = cast(MutableJSON, r.json())
        self._access_token = cast(str, data["access_token"])
        self._refresh_token = cast(str, data["refresh_token"])
        self._expires_at = time.time() + cast(float, data["expires_in"])
        try:
            creds = OpenAISubscription.load(account=self._account)
        except (FileNotFoundError, ValueError):
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

        """
        p = path or credentials_path(_DEFAULT_PATH, account)
        if not p.exists():
            raise FileNotFoundError(f"No credentials at {p}")
        raw: MutableJSON = json.loads(p.read_text(encoding="utf-8"))
        tokens = cast(MutableJSON, raw["tokens"])
        access_token = cast(str, tokens["access_token"])
        expires_at = _jwt_exp(access_token)
        creds = cls.Credentials(
            access_token=access_token,
            refresh_token=cast(str, tokens["refresh_token"]),
            account_id=cast(str, tokens["account_id"]),
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
        p = path or credentials_path(_DEFAULT_PATH, account)
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


# -- Model backend -----------------------------------------------------


class _OpenAISubModel(_OpenAIModel):
    """OpenAI Responses API model -- subscription billing."""

    _provider: OpenAISubscription  # type: narrowed from OpenAICompat

    @override
    def _is_effort_model(self, model_id: str) -> bool:
        return any(model_id.startswith(p) for p in _EFFORT_PREFIXES)

    @property
    @override
    def supports_thinking(self) -> bool:
        """Whether the model surfaces reasoning/thinking text."""
        return False

    @property
    @override
    def supports_account_auth(self) -> bool:
        """Whether the provider uses account authentication."""
        return True

    @override
    def is_context_overflow(self, error: Exception) -> bool:
        """Check whether an error indicates context-window overflow.

        Args:
          error: Exception raised by the API call.

        Returns:
          overflow: ``True`` if the error is a context-length overflow.

        """
        msg = str(error).lower()
        return (
            "context_length_exceeded" in msg
            or "maximum context length" in msg
            or "exceeds the context window" in msg
        )

    @override
    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Send a buffered request via the streaming path.

        Args:
          request: Model request to send.

        Returns:
          response: Translated model response.

        """
        return await self.stream(request, on_text=None)

    @override
    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Stream a request via the OpenAI Responses API.

        Args:
          request: Model request to send.
          on_text: Callback invoked with each text chunk as it arrives.
          on_thinking: Reserved; the OpenAI Responses API does not
              expose reasoning summaries as separate stream deltas.

        Returns:
          response: Assembled model response after the stream closes.

        """
        del on_thinking  # not exposed as a stream delta on this endpoint
        sdk = await self._provider.get_sdk()
        # The ChatGPT Codex subscription endpoint rejects some public
        # Responses API knobs, including ``temperature`` and
        # ``max_output_tokens``.
        reasoning: oai_shared.Reasoning | openai.Omit | None = (
            oai_shared.Reasoning(
                effort=cast("oai_shared.ReasoningEffort", request.effort),
            )
            if request.effort is not None and self.supports_effort
            else openai.omit
        )
        event_stream: AsyncResponseStream = await sdk.responses.create(  # pyright: ignore[reportCallIssue,reportUnknownVariableType]  # ty: ignore[no-matching-overload] -- SDK overload resolution fails on stream=True literal
            model=self._model_id,
            input=_build_input(request),
            instructions=request.system or "",
            store=False,
            stream=True,
            tools=(_build_tools(request.tools) if request.tools else openai.omit),
            reasoning=reasoning,  # pyright: ignore[reportArgumentType] -- Reasoning|Omit is valid; overload resolution failure cascades
        )
        return await _consume_stream(
            event_stream,  # pyright: ignore[reportUnknownArgumentType] -- typed above; overload suppression loses it
            pricing=self._profile.pricing,
            on_text=on_text,
        )


# -- Request building --------------------------------------------------


def _build_tools(tools: list[Tool]) -> list[oai_responses.FunctionToolParam]:
    return [_build_tool(t) for t in tools]


def _build_tool(tool: Tool) -> oai_responses.FunctionToolParam:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": cast(
            dict[str, object] | None, json_unfreeze(tool.directive_schema)
        ),
        "strict": None,
    }


def _build_input(request: ModelRequest) -> oai_responses.ResponseInputParam:
    """Convert a ModelRequest to Responses API input items."""
    ids = IdRemapper("fc_")
    items: oai_responses.ResponseInputParam = []
    for msg in request.messages:
        if msg.descriptor == "text/x-user-message":
            items.append({"role": "user", "content": cast(str, msg.content)})
        elif msg.descriptor == "multipart/x-user-message":
            items.append(_build_user_item(msg))
        elif msg.descriptor == "multipart/x-model-message":
            _build_assistant_items(msg, items, ids)
        elif msg.descriptor == "multipart/x-tool-result":
            items.append(_build_tool_result_item(msg, ids))
    return items


def _build_user_item(msg: Message) -> oai_responses.EasyInputMessageParam:
    content_parts = cast(tuple[Message, ...], msg.content)
    text_parts = [
        cast(str, p.content) for p in content_parts if p.descriptor == "text/plain"
    ]
    return {"role": "user", "content": "\n".join(text_parts)}


def _build_assistant_items(
    msg: Message,
    items: oai_responses.ResponseInputParam,
    ids: IdRemapper,
) -> None:
    """Expand a model message into assistant text and function_call items."""
    parts = cast(tuple[Message, ...], msg.content)
    text_parts: list[str] = []
    for part in parts:
        if part.descriptor == "text/plain":
            text_parts.append(cast(str, part.content))
        elif part.descriptor == "multipart/x-tool-call":
            if text_parts:
                items.append({"role": "assistant", "content": "\n".join(text_parts)})
                text_parts.clear()
            directive = get_directive(part)
            native_id = ids.map(get_queue_id(part))
            items.append(
                {
                    "type": "function_call",
                    "id": native_id,
                    "call_id": native_id,
                    "name": get_tool_name(part),
                    "arguments": json.dumps(json_unfreeze(directive)),
                }
            )
    if text_parts:
        items.append({"role": "assistant", "content": "\n".join(text_parts)})


def _build_tool_result_item(
    msg: Message,
    ids: IdRemapper,
) -> FunctionCallOutput:
    native_id = ids.map(get_queue_id(msg))
    parts = cast(tuple[Message, ...], msg.content)
    text = "\n".join(
        str(p.content) for p in parts if p.descriptor in ("text/plain", "text/x-error")
    )
    return {
        "type": "function_call_output",
        "call_id": native_id,
        "output": text,
    }


# -- Response parsing --------------------------------------------------


async def _consume_stream(
    stream: AsyncResponseStream,
    *,
    pricing: Pricing,
    on_text: Callable[[str], None] | None,
) -> ModelResponse:
    """Parse a Responses API event stream into an assembled ModelResponse."""
    text_parts: list[str] = []
    tool_calls: list[Message] = []
    tool_args: dict[str, list[str]] = {}
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    message_id = ""
    finish_reason: str | None = None

    async for event in stream:
        if isinstance(event, oai_responses.ResponseTextDeltaEvent):
            text_parts.append(event.delta)
            if on_text is not None:
                on_text(event.delta)
        elif isinstance(event, oai_responses.ResponseFunctionCallArgumentsDeltaEvent):
            tool_args.setdefault(event.item_id, []).append(event.delta)
        elif isinstance(event, oai_responses.ResponseOutputItemDoneEvent):
            item = event.item
            if item.type == "function_call":
                tc_id = str(item.call_id or "")  # ty: ignore[unresolved-attribute] -- narrowed by item.type == "function_call" but ty can't follow
                tc_name = str(item.name or "")  # ty: ignore[unresolved-attribute] -- narrowed by item.type == "function_call" but ty can't follow
                item_id = item.id or ""
                delta_args = "".join(tool_args.get(item_id, []))
                done_args = str(item.arguments or "")  # ty: ignore[unresolved-attribute] -- narrowed by item.type == "function_call" but ty can't follow
                args = _parse_tool_arguments(
                    delta_args,
                    done_args,
                    tool_name=tc_name,
                    call_id=tc_id,
                )
                tool_calls.append(tool_call_message(tc_id, tc_name, json_freeze(args)))
        elif isinstance(event, oai_responses.ResponseCompletedEvent):
            resp = event.response
            message_id = resp.id
            if resp.usage:
                input_tokens = resp.usage.input_tokens
                output_tokens = resp.usage.output_tokens
                if resp.usage.input_tokens_details:
                    cache_read = resp.usage.input_tokens_details.cached_tokens
            finish_reason = resp.status

    msg_parts: list[Message] = []
    if message_id:
        msg_parts.append(TextMessage(message_id, "text/x-queue-id"))
    if text_parts:
        msg_parts.append(TextMessage("".join(text_parts), "text/plain"))
    msg_parts.extend(tool_calls)

    has_tool_use = bool(tool_calls)
    raw_reason = _FINISH_MAP.get(finish_reason or "", finish_reason)

    in_cost, out_cost, total_cost = compute_cost(
        pricing,
        max(0, input_tokens - cache_read),
        output_tokens,
        cache_read=cache_read,
    )
    return ModelResponse(
        content=MultipartMessage(
            tuple(msg_parts),
            "multipart/x-model-message",
        ),
        tokens=TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
        ),
        stop_reason=normalize_stop_reason(
            raw_reason,
            kind="openai",
            has_tool_use=has_tool_use,
        ),
        message_id=message_id,
        request_id=message_id,
        input_cost=in_cost,
        output_cost=out_cost,
        total_cost=total_cost,
    )


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


# -- JWT helpers -------------------------------------------------------


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
