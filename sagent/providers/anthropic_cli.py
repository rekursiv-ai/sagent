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
from typing import TYPE_CHECKING, cast, override

import base64
import hashlib
import json
import logging
import os
import shutil
import tempfile

from sagent.agent.runtime import (
    AssistantMessage,
    HistoryEntry,
    ToolResult,
    UserMessage,
)
from sagent.custom_types import (
    ModelRequest,
    ModelResponse,
    TokenCount,
)
from sagent.lib.json import MutableJSON, int_val
from sagent.providers.anthropic import Anthropic, _strip_context_tag
from sagent.providers.lib.cost import ModelProfile, Pricing
from sagent.providers.lib.hotspare import _HotSpare
from sagent.providers.lib.mcp_bridge import _ToolsBridge
from sagent.providers.lib.oauth import credentials_path
from sagent.providers.lib.stop_reason import normalize_stop_reason
from sagent.providers.lib.subproc import _Subproc


if TYPE_CHECKING:
    import sagent.lib.image as image_lib
else:
    from sagent.lib.lazy_import import lazy_import

    image_lib = lazy_import("sagent.lib.image")


logger = logging.getLogger(__name__)


_CREDS_PATH = Path.home() / ".claude" / ".credentials.json"
_TURN_RESPAWN_THRESHOLD = 100
_CONTEXT_FRACTION_RESPAWN_THRESHOLD = 0.5


class AnthropicCLI(Anthropic):
    """Provider that drives the user's installed ``claude`` CLI subprocess.

    Inherits ``KNOWN_MODELS`` (limits, pricing, tokenizer density) from
    :class:`Anthropic`. Auth is the CLI's own credentials file --
    ``~/.claude/.credentials.json`` for the default account or the
    per-account variant produced by ``providers.lib.oauth.credentials_path``.
    Cost figures are computed from the per-turn ``modelUsage`` summary
    the CLI emits on the terminal ``result`` event.
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"
    DEFAULT_UTILITY_MODEL = "claude-haiku-4-5"

    def __init__(self, *, account: str | None = None) -> None:
        super().__init__(api_key="")
        self._account = account

    @classmethod
    @override
    def from_key(cls, api_key: str) -> Anthropic:
        """Create an API-key provider (delegates to :class:`Anthropic`).

        The CLI-wrapping provider is incompatible with API-key auth, so
        this returns a plain :class:`Anthropic` instance.

        Args:
          api_key: Anthropic API key (``sk-ant-...``).

        Returns:
          provider: ``Anthropic`` provider instance.

        """
        return Anthropic.from_key(api_key)

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
            _strip_context_tag(mid),
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
        self._tools_bridge: _ToolsBridge | None = None
        self._hot_spare = _HotSpare(self._spawn_initialized)
        # Set by ``stream`` before ``_spawn_initialized`` reads them.
        self._pending_system: str = ""
        self._sent_history_head: HistoryEntry | None = None

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
    def supports_effort(self) -> bool:
        """``False``: the CLI does not expose the effort knob on stream-json."""
        return False

    @property
    def supports_cache_control(self) -> bool:
        """``False``: prompt cache is the CLI's concern, not ours."""
        return False

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

    def estimate_text_token_count(self, text: str) -> int:
        """Estimate input tokens from ``len(text) / chars_per_token``.

        Args:
          text: Text to score.

        Returns:
          tokens: Integer token estimate.

        """
        return int(len(text) / self._profile.chars_per_token)

    def estimate_image_token_count(self, data: bytes) -> int:
        """Estimate tokens for an image using Anthropic's ``w*h/750`` formula.

        Args:
          data: Raw image bytes.

        Returns:
          tokens: Integer token estimate.

        """
        dims = image_lib.get_dimensions(data)
        return dims[0] * dims[1] // 750 if dims is not None else 0

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
            await self._hot_spare.respawn()
            self._last_sent_index = 0
            self._sent_history_head = None
        proc = await self._hot_spare.acquire()
        self._sync_tools_bridge(request)

        try:
            await self._send_new_entries(proc, request.messages)
            self._last_sent_index = len(request.messages)
            response = await self._drain_until_result(proc, on_text, on_thinking)
        except (RuntimeError, ValueError):
            await self._hot_spare.respawn()
            self._last_sent_index = 0
            self._sent_history_head = None
            raise
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

    async def _send_new_entries(
        self,
        proc: _Subproc,
        history: list[HistoryEntry],
    ) -> None:
        """Write each new history entry to stdin as a user-line."""
        if self._last_sent_index == 0 and history:
            self._sent_history_head = history[0]
        for entry in history[self._last_sent_index :]:
            if isinstance(entry, AssistantMessage):
                continue
            line = json.dumps(_serialize_for_stdin(entry, self.max_image_dim))
            await proc.write_line(line)

    async def _drain_until_result(
        self,
        proc: _Subproc,
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
    ) -> ModelResponse:
        """Read stream events until ``result``; assemble a ``ModelResponse``."""
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        usage_event: MutableJSON | None = None
        message_id = ""
        stop_reason: str | None = None
        while True:
            event = await proc.read_json_line(skip_non_json=False)
            if event is None:
                raise RuntimeError(
                    "AnthropicCLI: subprocess stdout closed before result"
                )
            kind = event.get("type")
            if kind == "result":
                usage_event = event
                stop_reason = cast(str | None, event.get("stop_reason"))
                if event.get("is_error"):
                    raise RuntimeError(f"AnthropicCLI: result is_error: {event}")
                break
            if kind == "stream_event":
                _dispatch_stream_event(
                    cast(MutableJSON, event.get("event") or {}),
                    text_parts,
                    thinking_parts,
                    on_text,
                    on_thinking,
                )
            elif kind == "system" and event.get("subtype") == "init":
                message_id = cast(str, event.get("session_id") or "")
        assert usage_event is not None
        self._last_input_tokens = int_val(
            cast(MutableJSON, usage_event.get("usage") or {}).get("input_tokens"), 0
        )
        return _build_model_response(
            usage_event=usage_event,
            text="".join(text_parts),
            thinking_parts=thinking_parts,
            stop_reason=stop_reason,
            fallback_message_id=message_id,
        )

    async def _spawn_initialized(self) -> _Subproc:
        """Spawn a fresh ``claude`` subprocess ready to receive user lines."""
        if self._tools_bridge is None:
            self._tools_bridge = _ToolsBridge(tools=[])
            await self._tools_bridge.start()
        tmpdir = Path(tempfile.mkdtemp(prefix="sagent-anthropic-cli-"))
        _populate_anthropic_tmpdir(tmpdir, self._provider.account)
        argv = _build_anthropic_argv(
            model_id=_strip_context_tag(self._model_id),
            system_prompt=self._pending_system,
            bridge_url=self._tools_bridge.url,
            bridge_server_name=self._tools_bridge.server_name,
        )
        proc = _Subproc(
            argv,
            env=_anthropic_subprocess_env(tmpdir),
            tmpdir=tmpdir,
        )
        await proc.start()
        self._system_hash = _hash_system(self._pending_system)
        self._turn_count = 0
        self._last_input_tokens = 0
        return proc


def _hash_system(system: str | None) -> str:
    """Hash for cheap equality checks between system prompts."""
    return hashlib.sha256((system or "").encode()).hexdigest()


def _populate_anthropic_tmpdir(tmpdir: Path, account: str | None) -> None:
    """Copy the user's credentials into a hermetic ``HOME`` for the CLI."""
    dot_claude = tmpdir / ".claude"
    dot_claude.mkdir(parents=True, exist_ok=True)
    source = credentials_path(_CREDS_PATH, account)
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


def _serialize_for_stdin(entry: HistoryEntry, max_image_dim: int) -> MutableJSON:
    """Translate a non-assistant ``HistoryEntry`` into the CLI's user-line shape."""
    if isinstance(entry, UserMessage):
        return _user_line(entry, max_image_dim)
    assert isinstance(entry, ToolResult)
    # Tool results never traverse stdin: the CLI's MCP client handled
    # the tool_use round-trip internally. Surface mistakes loudly.
    raise RuntimeError(
        "AnthropicCLI: ToolResult in history -- tools must go through the MCP bridge",
    )


def _user_line(entry: UserMessage, max_image_dim: int) -> MutableJSON:
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


def _build_model_response(
    *,
    usage_event: MutableJSON,
    text: str,
    thinking_parts: list[str],
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
    thinking_blocks = (
        ({"type": "thinking", "thinking": "".join(thinking_parts)},)
        if thinking_parts
        else ()
    )
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
