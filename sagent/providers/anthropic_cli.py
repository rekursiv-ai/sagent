"""``AnthropicCLI`` provider: wraps the user's installed ``claude`` CLI.

Speaks ``stream-json`` on stdin/stdout to a persistent ``claude --print``
subprocess that rides the user's CLI subscription. Tool calls are
routed through an in-process MCP bridge (``providers.lib.mcp_bridge``);
the CLI's own MCP client invokes the bridge, which dispatches to the
sagent ``Tool``s and returns results. The CLI handles the full
tool-use loop internally and emits a single ``result`` event per
sagent user turn.

See ``docs/private/cli_provider.md`` (§2) for the wire-protocol
reference, the spawn recipe, and the per-knob rationale.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast, override

import base64
import hashlib
import json
import logging
import os
import shutil
import tempfile

from sagent.lib import token_count
from sagent.lib.json import JSON, MutableJSON, int_val, validate_json_schema
from sagent.providers.anthropic import Anthropic
from sagent.providers.lib.cost import ModelProfile, Pricing
from sagent.providers.lib.hotspare import HotSpare
from sagent.providers.lib.mcp_bridge import ToolsBridge
from sagent.providers.lib.oauth import credentials_path
from sagent.providers.lib.stop_reason import normalize_stop_reason
from sagent.providers.lib.subproc import (
    Subproc,
    SubprocessTransportError,
)
from sagent.thinking import ThinkingCapability, valid_thinking_states
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
    TokenCount,
    UsageSnapshot,
    base_model_id,
)
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import TapeEvent


if TYPE_CHECKING:
    import sagent.lib.image as image_lib
else:
    from wrapt import lazy_import

    image_lib = lazy_import("sagent.lib.image")


logger = logging.getLogger(__name__)


_CREDS_PATH = Path.home() / ".claude" / ".credentials.json"
_TURN_RESPAWN_THRESHOLD = 100
_CONTEXT_FRACTION_RESPAWN_THRESHOLD = 0.5
_CREDENTIALS_SCHEMA: JSON = {
    "type": "object",
    "required": ["claudeAiOauth"],
    "properties": {
        "claudeAiOauth": {
            "type": "object",
            "required": ["accessToken", "refreshToken", "expiresAt"],
            "properties": {
                "accessToken": {"type": "string"},
                "refreshToken": {"type": "string"},
                "expiresAt": {"type": "number"},
            },
        },
    },
}


class AnthropicCLICredentials(TypedDict):
    """OAuth credentials from the Claude CLI credentials file."""

    access_token: str
    refresh_token: str
    expires_at: float
    scopes: NotRequired[list[str]]
    subscription_type: NotRequired[str | None]
    rate_limit_tier: NotRequired[str | None]
    account_uuid: NotRequired[str | None]
    email: NotRequired[str | None]
    organization_uuid: NotRequired[str | None]
    billing_type: NotRequired[str | None]
    account_created_at: NotRequired[str | None]
    subscription_created_at: NotRequired[str | None]
    has_extra_usage_enabled: NotRequired[bool | None]


class AnthropicCLI(Anthropic):
    """Provider that drives the user's installed ``claude`` CLI subprocess.

    Inherits ``KNOWN_MODELS`` (limits, pricing, tokenizer density) from
    :class:`Anthropic`. Auth is the CLI's own credentials file --
    ``~/.claude/.credentials.json`` for the default account or the
    per-account variant produced by ``providers.lib.oauth.credentials_path``.
    Cost figures are computed from the per-turn ``modelUsage`` summary
    the CLI emits on the terminal ``result`` event.
    """

    def __init__(self, *, account: str | None = None) -> None:
        super().__init__(api_key="")
        self._account = account

    @classmethod
    @override
    def from_key(
        cls,
        api_key: str,
        *,
        server_side_context_management: bool = False,
        redact_thinking: bool = False,
    ) -> Anthropic:
        """Create an API-key provider (delegates to :class:`Anthropic`).

        The CLI-wrapping provider is incompatible with API-key auth, so
        this returns a plain :class:`Anthropic` instance.

        Args:
          api_key: Anthropic API key (``sk-ant-...``).
          server_side_context_management: Forwarded to ``Anthropic.from_key``.
          redact_thinking: Forwarded to ``Anthropic.from_key``.

        Returns:
          provider: ``Anthropic`` provider instance.

        """
        return Anthropic.from_key(
            api_key,
            server_side_context_management=server_side_context_management,
            redact_thinking=redact_thinking,
        )

    @classmethod
    def from_credentials(cls, *, account: str | None = None) -> AnthropicCLI:
        """Build a provider that uses the local ``claude`` CLI's credentials.

        Args:
          account: Named credential slot. ``None`` reads the legacy
              unnamed credentials file.

        Returns:
          provider: Configured CLI-wrapping provider.

        Raises:
          FileNotFoundError: If the resolved credentials file is absent.
          RuntimeError: If ``claude`` is not on ``PATH``.

        """
        path = credentials_path(_CREDS_PATH, account)
        if not path.exists():
            raise FileNotFoundError(
                f"AnthropicCLI: no credentials at {path}; run `claude login`.",
            )
        if _load_cli_credentials_file(path) is None:
            raise ValueError(f"Invalid credentials file: {path}")
        if shutil.which("claude") is None:
            raise RuntimeError(
                "AnthropicCLI: `claude` is not on PATH; install the Claude CLI.",
            )
        return cls(account=account)

    @override
    def model(  # ty: ignore[invalid-method-override]  -- shared catalog, different transport; the CLI model can't subclass _AnthropicModel
        self,
        model_id: str | None = None,
        max_request_tokens: int | None = None,
    ) -> _AnthropicCLIModel:
        """Build a CLI-backed model.

        Args:
          model_id: Claude model id; ``None`` uses ``DEFAULT_MODEL``.
          max_request_tokens: Override the profile's input cap.

        Returns:
          model: Backend wrapping a managed ``claude`` subprocess.

        Raises:
          ValueError: If the resolved id is not in ``KNOWN_MODELS``.

        """
        mid = model_id if model_id is not None else self.DEFAULT_MODEL
        profile = self.KNOWN_MODELS.get(mid) or self.KNOWN_MODELS.get(
            base_model_id(mid),
        )
        if profile is None:
            known = ", ".join(sorted(self.KNOWN_MODELS))
            raise ValueError(
                f"Unknown model {mid!r} for AnthropicCLI. Known models: {known}",
            )
        return _AnthropicCLIModel(
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
    def utility_model(self) -> _AnthropicCLIModel:  # ty: ignore[invalid-method-override]  -- shared catalog, different transport
        """Return the cheapest CLI-backed model (Haiku by default).

        Returns:
          model: Utility model backend.

        """
        return self.model(self.DEFAULT_UTILITY_MODEL)

    @property
    def account(self) -> str | None:
        """Per-account credentials slot, ``None`` for the legacy file."""
        return self._account


class _AnthropicCLIModel:
    """``claude`` CLI subprocess wrapped as a sagent ``Model``.

    Args:
      provider: Owning :class:`AnthropicCLI`.
      model_id: Claude model id passed via ``--model``.
      profile: Resolved :class:`ModelProfile` for limits + pricing.
      max_request_tokens: Per-request input cap.

    """

    def __init__(
        self,
        *,
        provider: AnthropicCLI,
        model_id: str,
        profile: ModelProfile,
        max_request_tokens: int,
    ) -> None:
        self._provider = provider
        self._model_id = model_id
        self._profile = profile
        self._max_request_tokens = max_request_tokens
        self._last_sent_index = 0
        self._system_hash: str = ""
        self._turn_count = 0
        self._last_input_tokens = 0
        self._tools_bridge: ToolsBridge | None = None
        self._warming_proc: Subproc | None = None
        self._hot_spare = HotSpare(
            self._spawn_spare_initialized,
            close_partial=self._close_warming_proc,
        )
        # Set by ``stream`` before ``_spawn_initialized`` reads them.
        self._pending_system: str = ""
        self._sent_history_head: TapeEvent | None = None

    @property
    def max_request_tokens(self) -> int:
        """Per-request input token cap."""
        return self._max_request_tokens

    @property
    def model_id(self) -> str:
        """Model identifier passed to ``claude --model``."""
        return self._model_id

    @property
    def max_response_tokens(self) -> int:
        """Per-request output token cap from the profile."""
        return self._profile.max_response_tokens

    @property
    def supports_streaming(self) -> bool:
        """``True``: the wrapped CLI always streams."""
        return True

    @property
    def supports_thinking(self) -> bool:
        """Whether the active profile supports extended thinking."""
        return self._profile.supports_thinking

    @property
    def valid_thinking_states(self) -> tuple[str, ...]:
        """CLI transport returns readable thinking; redaction is inert here.

        The subprocess cannot send the redact-thinking beta header, so
        ``redact_thinking`` has no effect and no ``redact-hide`` mode is
        offered.
        """
        return valid_thinking_states(
            ThinkingCapability(supports_thinking=self.supports_thinking),
        )

    @property
    def supports_effort(self) -> bool:
        """``False``: the CLI does not expose the effort knob on stream-json."""
        return False

    @property
    def valid_efforts(self) -> tuple[str, ...]:
        """No effort knob on the CLI transport."""
        return ()

    @property
    def supports_cache_control(self) -> bool:
        """``False``: prompt cache is the CLI's concern, not ours."""
        return False

    @property
    def valid_service_tiers(self) -> tuple[str, ...]:
        """The CLI manages tier selection itself; no per-request knob."""
        return ()

    @property
    def valid_latency_modes(self) -> tuple[str, ...]:
        """The CLI exposes no per-request latency knob; fast mode unsupported."""
        return ()

    @property
    def supports_context_management(self) -> bool:
        """``True``: the CLI itself rolls history under quota pressure."""
        return True

    @property
    def supports_persistent_retry(self) -> bool:
        """``False``: persistent retry conflicts with subprocess lifecycle."""
        return False

    @property
    def supports_account_auth(self) -> bool:
        """``True``: the provider runs on the user's CLI subscription."""
        return True

    @property
    def pricing(self) -> Pricing:
        """Per-million-token pricing for the active profile."""
        return self._profile.pricing

    @property
    def max_image_dim(self) -> int:
        """Anthropic's documented vision pixel cap."""
        return 8000

    @property
    def max_image_bytes(self) -> int:
        """Anthropic's documented vision byte cap (5 MiB)."""
        return 5 * 1024 * 1024

    def approx_text_tokens(self, text: str) -> int:
        """Local estimate via ``chars_per_token``."""
        return int(len(text) / self._profile.chars_per_token)

    def approx_image_tokens(self, data: bytes) -> int:
        """Local estimate from image dimensions (``width*height/750``)."""
        dims = image_lib.get_dimensions(data)
        return dims[0] * dims[1] // 750 if dims is not None else 0

    def approx_request_tokens(self, request: ModelRequest) -> int:
        """Walk-and-sum every wire-bearing surface of ``request``."""
        return token_count.approx_request_tokens(request, self)

    async def actual_text_tokens(self, text: str) -> int:
        """Subprocess transport has no tokenizer access; falls back to approx."""
        return self.approx_text_tokens(text)

    async def actual_image_tokens(self, data: bytes) -> int:
        """Subprocess transport has no tokenizer access; falls back to approx."""
        return self.approx_image_tokens(data)

    async def actual_request_tokens(self, request: ModelRequest) -> int:
        """Subprocess transport has no tokenizer access; falls back to approx."""
        return self.approx_request_tokens(request)

    def is_context_overflow(self, error: Exception) -> bool:
        """Classify whether an error means the prompt exceeded the window.

        Args:
          error: Exception raised by the call path.

        Returns:
          overflow: ``True`` for known overflow markers in the message text.

        """
        msg = str(error).lower()
        return (
            "prompt is too long" in msg or "context window" in msg or "too_long" in msg
        )

    def is_retryable_provider_error(self, error: Exception) -> bool:
        """``False`` -- subprocess errors are handled via respawn, not retry."""
        del error
        return False

    def usage_snapshot(self) -> UsageSnapshot | None:
        """No rate-limit telemetry over the CLI subprocess transport."""
        return None

    async def buffer(self, request: ModelRequest) -> ModelResponse:
        """Run the request through the streaming path with no callbacks.

        Args:
          request: Fully-built model request.

        Returns:
          response: Translated model response.

        """
        return await self.stream(request, on_text=None, on_thinking=None)

    async def stream(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Drive one user turn through the ``claude`` subprocess.

        Sends each new history entry (skipping assistant turns the CLI
        already emitted itself), reads stream events until the terminal
        ``result``, and assembles a :class:`ModelResponse`.

        Args:
          request: Conversation + tools + system prompt for the turn.
          on_text: Per-chunk text callback; ``None`` disables live text.
          on_thinking: Per-chunk thinking callback; ``None`` disables it.

        Returns:
          response: Parsed model response with usage and cost filled in.

        Raises:
          RuntimeError: The subprocess exited before emitting a terminal
              ``result`` event, or the CLI surfaced an error result.

        """
        self._pending_system = request.system or ""
        if self._should_respawn(request):
            if _hash_system(request.system) != self._system_hash:
                await self._hot_spare.discard_spare()
            await self._hot_spare.respawn()
            self._reset_active_state()
        proc = await self._hot_spare.acquire()
        self._sync_tools_bridge(request)

        try:
            response = await self._exchange_turn(proc, request, on_text, on_thinking)
            self._system_hash = _hash_system(self._pending_system)
            self._last_sent_index = len(request.messages)
        except SubprocessTransportError:
            self._reset_active_state()
            await self._hot_spare.respawn_after_transport_failure()
            raise
        self._hot_spare.record_success()
        self._turn_count += 1
        return response

    async def close(self) -> None:
        """Tear down the subprocess pool and the MCP bridge."""
        await self._hot_spare.close()
        if self._tools_bridge is not None:
            await self._tools_bridge.stop()
            self._tools_bridge = None

    def _should_respawn(self, request: ModelRequest) -> bool:
        """Inspect the trigger list (§1.4) for this request."""
        if self._hot_spare.active is None:
            return False
        history = request.messages
        if not history:
            return True
        if self._last_sent_index > len(history):
            return True
        if self._sent_history_head is not None and history[0] is not (
            self._sent_history_head
        ):
            return True
        if _hash_system(request.system) != self._system_hash:
            return True
        if self._turn_count >= _TURN_RESPAWN_THRESHOLD:
            return True
        return (
            self._last_input_tokens
            > self._max_request_tokens * _CONTEXT_FRACTION_RESPAWN_THRESHOLD
        )

    def _sync_tools_bridge(self, request: ModelRequest) -> None:
        """Refresh the MCP bridge's tool registry to match the request."""
        if self._tools_bridge is not None:
            self._tools_bridge.update_tools(list(request.tools or []))

    async def _exchange_turn(
        self,
        proc: Subproc,
        request: ModelRequest,
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
    ) -> ModelResponse:
        """Replay prior entries quietly, then return the current turn result."""
        new_entries = request.messages[self._last_sent_index :]
        if self._last_sent_index == 0 and request.messages:
            self._sent_history_head = request.messages[0]
        user_like_entries = [
            entry for entry in new_entries if not isinstance(entry, AssistantMessage)
        ]
        for entry in user_like_entries[:-1]:
            await self._send_entry(proc, entry)
            _ = await self._drain_until_result(
                proc, on_text=None, on_thinking=None, update_input_tokens=False
            )
        if not user_like_entries:
            return await self._drain_until_result(proc, on_text, on_thinking)
        await self._send_entry(proc, user_like_entries[-1])
        return await self._drain_until_result(proc, on_text, on_thinking)

    async def _send_entry(self, proc: Subproc, entry: TapeEvent) -> None:
        """Write one history entry to stdin."""
        line = json.dumps(_serialize_for_stdin(entry, self.max_image_dim))
        await proc.write_line(line)

    async def _drain_until_result(
        self,
        proc: Subproc,
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
        *,
        update_input_tokens: bool = True,
    ) -> ModelResponse:
        """Read stream events until ``result``; assemble a ``ModelResponse``."""
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        # Thought-signature deltas accumulated alongside the thinking
        # body; carried into the response's thinking block so a later
        # wire re-send stays signed.
        signature_parts: list[str] = []
        usage_event: MutableJSON | None = None
        message_id = ""
        stop_reason: str | None = None
        while True:
            event = await proc.read_json_line(skip_non_json=False)
            if event is None:
                raise SubprocessTransportError(
                    "AnthropicCLI: subprocess stdout closed before result"
                )
            kind = event.get("type")
            if kind == "result":
                usage_event = event
                stop_reason = cast(str | None, event.get("stop_reason"))
                if event.get("is_error"):
                    raise SubprocessTransportError(
                        f"AnthropicCLI: result is_error: {event}"
                    )
                break
            if kind == "stream_event":
                _dispatch_stream_event(
                    cast(MutableJSON, event.get("event") or {}),
                    text_parts,
                    thinking_parts,
                    signature_parts,
                    on_text,
                    on_thinking,
                )
            elif kind == "system" and event.get("subtype") == "init":
                message_id = cast(str, event.get("session_id") or "")
        assert usage_event is not None
        if update_input_tokens:
            self._last_input_tokens = int_val(
                cast(MutableJSON, usage_event.get("usage") or {}).get("input_tokens"), 0
            )
        return _build_model_response(
            usage_event=usage_event,
            text="".join(text_parts),
            thinking_parts=thinking_parts,
            signature_parts=signature_parts,
            stop_reason=stop_reason,
            fallback_message_id=message_id,
        )

    async def _spawn_spare_initialized(self) -> Subproc:
        """Spawn a spare ``claude`` subprocess without touching active counters."""
        return await self._spawn_initialized()

    async def _spawn_initialized(self) -> Subproc:
        """Spawn a fresh ``claude`` subprocess ready to receive user lines."""
        if self._tools_bridge is None:
            self._tools_bridge = ToolsBridge(tools=[])
            await self._tools_bridge.start()
        tmpdir = Path(tempfile.mkdtemp(prefix="sagent-anthropic-cli-"))
        _populate_anthropic_tmpdir(tmpdir, self._provider.account)
        argv = _build_anthropic_argv(
            model_id=base_model_id(self._model_id),
            system_prompt=self._pending_system,
            bridge_url=self._tools_bridge.url,
            bridge_server_name=self._tools_bridge.server_name,
        )
        proc = Subproc(
            argv,
            env=_anthropic_subprocess_env(tmpdir),
            tmpdir=tmpdir,
            # ``claude --print --output-format stream-json`` emits ONE
            # NDJSON record per content block; a single Read of a large
            # file echoes the file's content verbatim into one line
            # (40+ KiB tool results are routine). Subproc's stock 64 KiB
            # per-line cap strands ``readline()`` with "ValueError:
            # Separator is found, but chunk is longer than limit" --
            # diagnosed live 2026-06-03 after back-to-back ~41 KiB +
            # ~22 KiB file reads tripped it. 16 MiB is comfortably above
            # any single stream-json record while keeping memory bounded.
            stream_limit=16 * 1024 * 1024,
        )
        self._warming_proc = proc
        try:
            await proc.start()
        except BaseException:
            await proc.close()
            self._warming_proc = None
            raise
        self._warming_proc = None
        return proc

    def _reset_active_state(self) -> None:
        """Reset active subprocess counters after a respawn boundary."""
        self._turn_count = 0
        self._last_input_tokens = 0
        self._reset_delta_state()

    def _reset_delta_state(self) -> None:
        """Reset sent-history delta tracking."""
        self._last_sent_index = 0
        self._sent_history_head = None

    async def _close_warming_proc(self) -> None:
        """Close the subprocess currently being warmed, if any."""
        if self._warming_proc is None:
            return
        proc = self._warming_proc
        self._warming_proc = None
        await proc.close()


def _hash_system(system: str | None) -> str:
    """Hash for cheap equality checks between system prompts."""
    return hashlib.sha256((system or "").encode()).hexdigest()


def _parse_cli_credentials(raw: MutableJSON) -> AnthropicCLICredentials:
    """Extract access/refresh/expiry from Claude CLI credential JSON."""
    oauth = cast(MutableJSON, raw["claudeAiOauth"])
    creds = AnthropicCLICredentials(
        access_token=cast(str, oauth["accessToken"]),
        refresh_token=cast(str, oauth["refreshToken"]),
        expires_at=cast(float, oauth["expiresAt"]) / 1000.0,
    )
    if "scopes" in oauth:
        creds["scopes"] = cast(list[str], oauth["scopes"])
    if "subscriptionType" in oauth:
        creds["subscription_type"] = cast(str | None, oauth["subscriptionType"])
    if "rateLimitTier" in oauth:
        creds["rate_limit_tier"] = cast(str | None, oauth["rateLimitTier"])
    token_account_raw = oauth.get("tokenAccount")
    if isinstance(token_account_raw, dict):
        token_account = cast(MutableJSON, token_account_raw)
        if token_account.get("uuid"):
            creds["account_uuid"] = cast(str, token_account["uuid"])
        if token_account.get("emailAddress"):
            creds["email"] = cast(str, token_account["emailAddress"])
        if token_account.get("organizationUuid"):
            creds["organization_uuid"] = cast(str, token_account["organizationUuid"])
    if "billingType" in oauth:
        creds["billing_type"] = cast(str | None, oauth["billingType"])
    if "accountCreatedAt" in oauth:
        creds["account_created_at"] = cast(str | None, oauth["accountCreatedAt"])
    if "subscriptionCreatedAt" in oauth:
        creds["subscription_created_at"] = cast(
            str | None, oauth["subscriptionCreatedAt"]
        )
    if "hasExtraUsageEnabled" in oauth:
        creds["has_extra_usage_enabled"] = cast(
            bool | None, oauth["hasExtraUsageEnabled"]
        )
    return creds


def _load_cli_credentials_file(path: Path) -> AnthropicCLICredentials | None:
    """Read credentials from a Claude CLI-format JSON file."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    raw = cast(MutableJSON, data)
    if validate_json_schema(_CREDENTIALS_SCHEMA, raw):
        return None
    return _parse_cli_credentials(raw)


def _populate_anthropic_tmpdir(tmpdir: Path, account: str | None) -> None:
    """Copy the user's credentials into a hermetic ``HOME`` for the CLI."""
    dot_claude = tmpdir / ".claude"
    dot_claude.mkdir(parents=True, exist_ok=True)
    source = credentials_path(_CREDS_PATH, account)
    if _load_cli_credentials_file(source) is None:
        raise ValueError(f"Invalid credentials file: {source}")
    target = dot_claude / _CREDS_PATH.name
    shutil.copyfile(source, target)
    target.chmod(0o600)


def _anthropic_subprocess_env(tmpdir: Path) -> dict[str, str]:
    """Build the env for the ``claude`` subprocess (telemetry off, hermetic HOME)."""
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env.update(
        {
            "HOME": str(tmpdir),
            "USERPROFILE": str(tmpdir),
            "DISABLE_AUTO_COMPACT": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_BUG_COMMAND": "1",
            "DISABLE_FEEDBACK_COMMAND": "1",
            "DISABLE_COST_WARNINGS": "1",
            "DISABLE_INSTALLATION_CHECKS": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_SKIP_PROMPT_HISTORY": "1",
            "CLAUDE_CODE_DISABLE_LEGACY_MODEL_REMAP": "1",
            "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
        }
    )
    return env


def _build_anthropic_argv(
    *,
    model_id: str,
    system_prompt: str,
    bridge_url: str,
    bridge_server_name: str,
) -> list[str]:
    """Assemble the ``claude --print --input-format stream-json ...`` argv."""
    mcp_config = json.dumps(
        {
            "mcpServers": {
                bridge_server_name: {"type": "http", "url": bridge_url},
            }
        }
    )
    return [
        "claude",
        "--print",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--model",
        model_id,
        "--system-prompt",
        system_prompt,
        "--no-session-persistence",
        "--setting-sources",
        "",
        "--mcp-config",
        mcp_config,
        "--strict-mcp-config",
        "--tools",
        "",
        "--disable-slash-commands",
        "--permission-mode",
        "bypassPermissions",
    ]


def _serialize_for_stdin(entry: TapeEvent, max_image_dim: int) -> MutableJSON:
    """Translate a non-assistant ``TapeEvent`` into the CLI's user-line shape."""
    if isinstance(entry, (AgentSendMessage, UserMessage)):
        return _user_line(entry, max_image_dim)
    assert isinstance(entry, ToolResult)
    # Tool results never traverse stdin: the CLI's MCP client handled
    # the tool_use round-trip internally. Surface mistakes loudly.
    raise RuntimeError(
        "AnthropicCLI: ToolResult in history -- tools must go through the MCP bridge",
    )


def _user_line(
    entry: AgentSendMessage | UserMessage, max_image_dim: int
) -> MutableJSON:
    """Build a ``{"type":"user", ...}`` stdin line, attaching images inline."""
    image_attachments = [
        att for att in entry.attachments if att.descriptor.startswith("image/")
    ]
    if not image_attachments:
        return cast(
            MutableJSON,
            {"type": "user", "message": {"role": "user", "content": entry.text}},
        )
    content: list[MutableJSON] = []
    for att in image_attachments:
        raw, mime = image_lib.resize(
            att.data, max_dim=max_image_dim, max_bytes=5 * 1024 * 1024
        )
        content.append(
            cast(
                MutableJSON,
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": base64.b64encode(raw).decode(),
                    },
                },
            )
        )
    if entry.text:
        content.append({"type": "text", "text": entry.text})
    return cast(
        MutableJSON,
        {"type": "user", "message": {"role": "user", "content": content}},
    )


def _dispatch_stream_event(
    event: MutableJSON,
    text_parts: list[str],
    thinking_parts: list[str],
    signature_parts: list[str],
    on_text: Callable[[str], None] | None,
    on_thinking: Callable[[str], None] | None,
) -> None:
    """Route one stream_event payload to ``on_text`` / ``on_thinking``."""
    delta = cast(MutableJSON, event.get("delta") or {})
    delta_type = delta.get("type")
    if delta_type == "text_delta":
        text = cast(str, delta.get("text") or "")
        if text:
            text_parts.append(text)
            if on_text is not None:
                on_text(text)
    elif delta_type == "thinking_delta":
        text = cast(str, delta.get("thinking") or "")
        if text:
            thinking_parts.append(text)
            if on_thinking is not None:
                on_thinking(text)
    elif delta_type == "signature_delta":
        # Per Anthropic's stream-json spec, ``signature_delta``
        # carries the opaque thought-signature in the ``signature``
        # field (mirrors ``thinking_delta`` for body text). The
        # final signature is the concatenation across deltas
        # (typically a single delta in practice). Required so any
        # downstream wire re-send embeds the signature in the
        # thinking block — Anthropic's API rejects unsigned thinking
        # with HTTP 400 ``thinking.signature: Field required``.
        sig = cast(str, delta.get("signature") or "")
        if sig:
            signature_parts.append(sig)


def _build_model_response(
    *,
    usage_event: MutableJSON,
    text: str,
    thinking_parts: list[str],
    signature_parts: list[str],
    stop_reason: str | None,
    fallback_message_id: str,
) -> ModelResponse:
    """Aggregate per-model usage rows and assemble a ``ModelResponse``."""
    model_usage = cast(MutableJSON, usage_event.get("modelUsage") or {})
    input_tokens = 0
    output_tokens = 0
    cache_creation = 0
    cache_read = 0
    total_cost = 0.0
    for row in model_usage.values():
        if not isinstance(row, dict):
            continue
        row_map = cast(MutableJSON, row)
        input_tokens += int_val(row_map.get("inputTokens"), 0)
        output_tokens += int_val(row_map.get("outputTokens"), 0)
        cache_creation += int_val(row_map.get("cacheCreationInputTokens"), 0)
        cache_read += int_val(row_map.get("cacheReadInputTokens"), 0)
        cost = row_map.get("costUSD")
        if isinstance(cost, (int, float)):
            total_cost += float(cost)
    if total_cost == 0.0:
        # Fall back to top-level total_cost_usd when modelUsage is absent.
        raw = usage_event.get("total_cost_usd")
        if isinstance(raw, (int, float)):
            total_cost = float(raw)
    # Build the single thinking block from the accumulated body + signature.
    # The signature MUST be present whenever the body is — otherwise a
    # subsequent wire send rejects with ``thinking.signature: Field required``.
    # We elide the block entirely if there's no body (no thinking happened).
    if thinking_parts:
        thinking_blocks: tuple[dict[str, object], ...] = (
            {
                "type": "thinking",
                "thinking": "".join(thinking_parts),
                "signature": "".join(signature_parts),
            },
        )
    else:
        thinking_blocks = ()
    message_id = cast(str, usage_event.get("session_id") or fallback_message_id)
    return ModelResponse(
        message=AssistantMessage(
            text=text,
            thinking_blocks=thinking_blocks,
            tool_calls=(),
        ),
        tokens=TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
        ),
        stop_reason=normalize_stop_reason(
            stop_reason,
            kind="anthropic",
            has_tool_use=False,
        ),
        message_id=message_id,
        request_id=message_id,
        input_cost=0.0,
        output_cost=0.0,
        total_cost=total_cost,
    )
