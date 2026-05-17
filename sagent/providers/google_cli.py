"""``GoogleCLI`` provider: wraps the user's installed ``gemini`` CLI in ACP mode.

Speaks the Agent Client Protocol (JSON-RPC over NDJSON) to a
persistent ``gemini --experimental-acp`` subprocess that rides the
user's Gemini Code Assist subscription. Tool calls flow through the
in-process MCP bridge (``providers.lib.mcp_bridge``); the CLI
registers the bridge via ``session/new.params.mcpServers`` with
``type: "http"`` and dispatches sagent tools by URL.

The terminal ACP event for each user turn is the JSON-RPC response to
``session/prompt`` (matched by id). Streaming chunks arrive as
``session/update`` notifications with ``agent_message_chunk`` /
``agent_thought_chunk`` payloads in between. See
``docs/private/cli_provider.md`` (§3) for protocol details, the spawn
recipe, and per-knob rationale.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
import shutil
import tempfile

from sagent.lib.json import MutableJSON
from sagent.providers.google import Google
from sagent.providers.lib.cost import (
    ModelProfile,
    Pricing,
    compute_cost,
)
from sagent.providers.lib.hotspare import HotSpare
from sagent.providers.lib.mcp_bridge import ToolsBridge
from sagent.providers.lib.oauth import credentials_path
from sagent.providers.lib.stop_reason import normalize_stop_reason
from sagent.providers.lib.subproc import Subproc
from sagent.types.history import (
    AssistantMessage,
    HistoryEntry,
    ToolResult,
    UserMessage,
)
from sagent.types.model import ModelRequest, ModelResponse, TokenCount


if TYPE_CHECKING:
    import sagent.lib.image as image_lib
else:
    from sagent.lib.lazy_import import lazy_import

    image_lib = lazy_import("sagent.lib.image")


logger = logging.getLogger(__name__)


_GEMINI_DIR = Path.home() / ".gemini"
_CREDS_PATH = _GEMINI_DIR / "oauth_creds.json"
_TURN_RESPAWN_THRESHOLD = 100
_CONTEXT_FRACTION_RESPAWN_THRESHOLD = 0.5
# Module-level mutable. Lazily-bound to the running loop's lock so a
# late-import session doesn't pin the asyncio.Lock to an event loop
# that doesn't exist yet.
_writeback_lock: asyncio.Lock | None = None


class GoogleCLI(Google):
    """Provider that drives the user's installed ``gemini`` CLI subprocess.

    Inherits ``KNOWN_MODELS`` (limits, pricing) from :class:`Google`.
    Auth is the CLI's own OAuth credentials at
    ``~/.gemini/oauth_creds.json`` (or the named-account variant).
    Cost figures are estimated from public per-token pricing since ACP
    does not surface per-turn usage on ``session/prompt`` responses.
    """

    def __init__(self, *, account: str | None = None) -> None:
        # Skip Google.__init__: the CLI provides auth, no api_key needed.
        self._account = account

    @property
    def api_key(self) -> str:  # pyright: ignore[reportImplicitOverride]  -- intentionally shadows parent attribute
        """Compatibility shim returning the empty string."""
        return ""

    @classmethod
    @override
    def from_key(cls, api_key: str) -> Google:
        """Create an API-key provider (delegates to :class:`Google`).

        The CLI-wrapping provider is incompatible with API-key auth, so
        this returns a plain :class:`Google` instance.

        Args:
          api_key: Google AI Studio API key.

        Returns:
          provider: ``Google`` provider instance.

        """
        return Google.from_key(api_key)

    @classmethod
    def from_credentials(cls, *, account: str | None = None) -> GoogleCLI:
        """Build a provider that uses the local ``gemini`` CLI's credentials.

        Args:
          account: Named credential slot. ``None`` reads the legacy
              unnamed credentials file.

        Returns:
          provider: Configured CLI-wrapping provider.

        Raises:
          FileNotFoundError: If the resolved credentials file is absent.
          RuntimeError: If ``gemini`` is not on ``PATH``.

        """
        path = credentials_path(_CREDS_PATH, account)
        if not path.exists():
            raise FileNotFoundError(
                f"GoogleCLI: no credentials at {path}; run `gemini` and authenticate.",
            )
        if shutil.which("gemini") is None:
            raise RuntimeError(
                "GoogleCLI: `gemini` is not on PATH; install the Gemini CLI.",
            )
        return cls(account=account)

    @override
    def model(  # ty: ignore[invalid-method-override]  -- shared catalog, different transport; the CLI model can't subclass _GeminiModel
        self,
        model_id: str | None = None,
        max_request_tokens: int | None = None,
    ) -> _GoogleCLIModel:
        """Build a CLI-backed model.

        Args:
          model_id: Gemini model id; ``None`` uses ``DEFAULT_MODEL``.
          max_request_tokens: Override the profile's input cap.

        Returns:
          model: Backend wrapping a managed ``gemini --experimental-acp`` subprocess.

        Raises:
          ValueError: If the resolved id is not in ``KNOWN_MODELS``.

        """
        mid = model_id if model_id is not None else self.DEFAULT_MODEL
        profile = self.KNOWN_MODELS.get(mid)
        if profile is None:
            known = ", ".join(sorted(self.KNOWN_MODELS))
            raise ValueError(
                f"Unknown model {mid!r} for GoogleCLI. Known models: {known}",
            )
        return _GoogleCLIModel(
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
    def utility_model(self) -> _GoogleCLIModel:  # ty: ignore[invalid-method-override]  -- shared catalog, different transport
        """Return the cheapest CLI-backed model."""
        return self.model(self.DEFAULT_UTILITY_MODEL)

    @property
    def account(self) -> str | None:
        """Per-account credentials slot, ``None`` for the legacy file."""
        return self._account


class _GoogleCLIModel:
    """``gemini --experimental-acp`` subprocess wrapped as a sagent ``Model``.

    Args:
      provider: Owning :class:`GoogleCLI`.
      model_id: Gemini model id passed via ``--model``.
      profile: Resolved :class:`ModelProfile` for limits + pricing.
      max_request_tokens: Per-request input cap.

    """

    def __init__(
        self,
        *,
        provider: GoogleCLI,
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
        self._next_rpc_id = 1
        self._session_id: str = ""
        self._tmpdir: Path | None = None
        self._hot_spare = HotSpare(self._spawn_initialized)
        self._pending_system: str = ""
        self._sent_history_head: HistoryEntry | None = None

    @property
    def max_request_tokens(self) -> int:
        """Per-request input token cap."""
        return self._max_request_tokens

    @property
    def model_id(self) -> str:
        """Model identifier passed to ``gemini --model``."""
        return self._model_id

    @property
    def max_response_tokens(self) -> int:
        """Per-request output token cap from the profile."""
        return self._profile.max_response_tokens

    @property
    def supports_streaming(self) -> bool:
        """``True``: ACP streams updates per turn."""
        return True

    @property
    def supports_thinking(self) -> bool:
        """``True``: ACP exposes ``agent_thought_chunk`` notifications."""
        return True

    @property
    def supports_effort(self) -> bool:
        """``False``: ACP has no effort knob on ``session/prompt``."""
        return False

    @property
    def supports_cache_control(self) -> bool:
        """``False``: prompt cache is the CLI's concern, not ours."""
        return False

    @property
    def valid_service_tiers(self) -> tuple[str, ...]:
        """The Gemini CLI manages tier selection itself."""
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
        """Gemini's documented vision pixel cap."""
        return 3072

    @property
    def max_image_bytes(self) -> int:
        """Gemini's documented vision byte cap (20 MiB)."""
        return 20 * 1024 * 1024

    def estimate_text_token_count(self, text: str) -> int:
        """Estimate input tokens from ``len(text) / 4`` (Gemini's heuristic).

        Args:
          text: Text to score.

        Returns:
          tokens: Integer token estimate.

        """
        return len(text) // 4

    def estimate_image_token_count(self, data: bytes) -> int:
        """Estimate tokens for an image using Gemini's tile heuristic.

        Args:
          data: Raw image bytes.

        Returns:
          tokens: Integer token estimate.

        """
        dims = image_lib.get_dimensions(data)
        if dims is None:
            return 0
        # 258 tokens per 512x512 tile (matches `Google._GeminiModel`).
        return ((dims[0] + 511) // 512) * ((dims[1] + 511) // 512) * 258

    def is_context_overflow(self, error: Exception) -> bool:
        """Classify whether an error means the prompt exceeded the window.

        Args:
          error: Exception raised by the call path.

        Returns:
          overflow: ``True`` for known overflow markers in the message text.

        """
        msg = str(error).lower()
        return "too large" in msg or "too long" in msg or "exceeds the maximum" in msg

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
        """Drive one user turn through the ``gemini`` ACP subprocess.

        Args:
          request: Conversation + tools + system prompt for the turn.
          on_text: Per-chunk text callback; ``None`` disables live text.
          on_thinking: Per-chunk thinking callback; ``None`` disables it.

        Returns:
          response: Parsed model response with cost estimated from pricing.

        Raises:
          RuntimeError: The subprocess closed before responding, or the
              server returned a JSON-RPC ``error`` field.

        """
        self._pending_system = request.system or ""
        if self._should_respawn(request):
            await self._hot_spare.respawn()
            self._last_sent_index = 0
            self._sent_history_head = None
        proc = await self._hot_spare.acquire()
        self._sync_tools_bridge(request)

        try:
            response = await self._exchange_turn(proc, request, on_text, on_thinking)
            self._last_sent_index = len(request.messages)
        except (RuntimeError, ValueError):
            await self._hot_spare.respawn()
            self._last_sent_index = 0
            self._sent_history_head = None
            raise
        self._turn_count += 1
        await self._writeback_credentials()
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
        """Send each new entry as ``session/prompt`` and assemble the reply."""
        new_entries = request.messages[self._last_sent_index :]
        if self._last_sent_index == 0 and request.messages:
            self._sent_history_head = request.messages[0]
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        stop_reason: str | None = None
        for entry in new_entries:
            if isinstance(entry, AssistantMessage):
                continue
            blocks = _serialize_prompt_blocks(entry, self.max_image_dim)
            stop_reason = await self._send_prompt(
                proc,
                blocks,
                text_parts,
                thinking_parts,
                on_text,
                on_thinking,
            )
        return self._build_response(text_parts, thinking_parts, stop_reason, request)

    async def _send_prompt(
        self,
        proc: Subproc,
        prompt_blocks: list[MutableJSON],
        text_parts: list[str],
        thinking_parts: list[str],
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
    ) -> str | None:
        """Issue one ``session/prompt`` RPC and drain until its response."""
        request_id = self._allocate_rpc_id()
        await _rpc_send(
            proc,
            request_id,
            "session/prompt",
            {"sessionId": self._session_id, "prompt": prompt_blocks},
        )
        while True:
            msg = await proc.read_json_line(skip_non_json=True)
            if msg is None:
                raise RuntimeError(
                    "GoogleCLI: subprocess stdout closed before response"
                )
            if msg.get("id") == request_id:
                if "error" in msg:
                    raise RuntimeError(f"GoogleCLI: JSON-RPC error: {msg['error']}")
                result = cast(MutableJSON, msg.get("result") or {})
                return cast(str | None, result.get("stopReason"))
            if msg.get("method") == "session/update":
                _dispatch_session_update(
                    cast(MutableJSON, msg.get("params") or {}),
                    text_parts,
                    thinking_parts,
                    on_text,
                    on_thinking,
                )

    def _build_response(
        self,
        text_parts: list[str],
        thinking_parts: list[str],
        stop_reason: str | None,
        request: ModelRequest,
    ) -> ModelResponse:
        """Assemble a ``ModelResponse`` with estimated token usage and cost."""
        full_text = "".join(text_parts)
        input_tokens = _estimate_input_tokens(request.messages, self)
        output_tokens = self.estimate_text_token_count(full_text)
        in_cost, out_cost, total_cost = compute_cost(
            self._profile.pricing, input_tokens, output_tokens
        )
        self._last_input_tokens = input_tokens
        thinking_blocks = (
            ({"type": "thinking", "thinking": "".join(thinking_parts)},)
            if thinking_parts
            else ()
        )
        return ModelResponse(
            message=AssistantMessage(
                text=full_text,
                thinking_blocks=thinking_blocks,
                tool_calls=(),
            ),
            tokens=TokenCount(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            stop_reason=normalize_stop_reason(
                stop_reason,
                kind="google",
                has_tool_use=False,
            ),
            message_id=self._session_id,
            request_id=self._session_id,
            input_cost=in_cost,
            output_cost=out_cost,
            total_cost=total_cost,
        )

    async def _spawn_initialized(self) -> Subproc:
        """Spawn ``gemini`` and run the ACP handshake to a live ``sessionId``."""
        if self._tools_bridge is None:
            self._tools_bridge = ToolsBridge(tools=[])
            await self._tools_bridge.start()
        tmpdir = Path(tempfile.mkdtemp(prefix="sagent-google-cli-"))
        _populate_google_tmpdir(tmpdir, self._provider.account, self._pending_system)
        workdir = tmpdir / "workdir"
        proc = Subproc(
            ["gemini", "--experimental-acp", "--model", self._model_id],
            env=_google_subprocess_env(tmpdir),
            tmpdir=tmpdir,
            cwd=workdir,
        )
        await proc.start()
        await self._acp_handshake(proc, workdir)
        self._system_hash = _hash_system(self._pending_system)
        self._turn_count = 0
        self._last_input_tokens = 0
        self._tmpdir = tmpdir
        return proc

    async def _acp_handshake(self, proc: Subproc, workdir: Path) -> None:
        """Run ``initialize`` → ``authenticate`` → ``session/new`` (§3.2)."""
        assert self._tools_bridge is not None
        await _rpc_call(
            proc,
            self._allocate_rpc_id(),
            "initialize",
            {"protocolVersion": 1, "clientCapabilities": {}},
        )
        await _rpc_call(
            proc,
            self._allocate_rpc_id(),
            "authenticate",
            {"methodId": "oauth-personal"},
        )
        result = await _rpc_call(
            proc,
            self._allocate_rpc_id(),
            "session/new",
            {
                "cwd": str(workdir),
                "mcpServers": [
                    {
                        "name": self._tools_bridge.server_name,
                        "type": "http",
                        "url": self._tools_bridge.url,
                        "headers": [],
                    }
                ],
            },
        )
        session_id = cast(str | None, result.get("sessionId"))
        if not session_id:
            raise RuntimeError(
                f"GoogleCLI: session/new returned no sessionId: {result}"
            )
        self._session_id = session_id

    def _allocate_rpc_id(self) -> int:
        """Allocate the next JSON-RPC id, incrementing in place."""
        rid = self._next_rpc_id
        self._next_rpc_id += 1
        return rid

    async def _writeback_credentials(self) -> None:
        """Copy refreshed tmpdir creds back to the user's home (newer-wins).

        Skipped if the user's copy is newer than the tmpdir's, so stale
        in-flight refreshes never clobber a fresher token from a sibling
        process or an interactive ``/login``.
        """
        if self._tmpdir is None:
            return
        global _writeback_lock  # noqa: PLW0603 -- lazy-init the lock at the running loop
        if _writeback_lock is None:
            _writeback_lock = asyncio.Lock()
        async with _writeback_lock:
            src = self._tmpdir / ".gemini" / "oauth_creds.json"
            if not src.exists():
                return
            target = credentials_path(_CREDS_PATH, self._provider.account)
            tmpdir_expiry = _read_expiry(src)
            user_expiry = _read_expiry(target) if target.exists() else 0.0
            if tmpdir_expiry <= user_expiry:
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
            target.chmod(0o600)


def _hash_system(system: str | None) -> str:
    """Hash for cheap equality checks between system prompts."""
    return hashlib.sha256((system or "").encode()).hexdigest()


def _read_expiry(path: Path) -> float:
    """Return the ``expiry_date`` (JS millis) in ``path``, or ``0`` on any error."""
    with contextlib.suppress(OSError, json.JSONDecodeError, KeyError, ValueError):
        return float(json.loads(path.read_text(encoding="utf-8"))["expiry_date"])
    return 0.0


def _populate_google_tmpdir(
    tmpdir: Path,
    account: str | None,
    system_prompt: str,
) -> None:
    """Lay out the hermetic ``HOME`` the spawn recipe (§3.1) expects."""
    dot_gemini = tmpdir / ".gemini"
    workdir = tmpdir / "workdir"
    dot_gemini.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    creds_src = credentials_path(_CREDS_PATH, account)
    creds_dst = dot_gemini / "oauth_creds.json"
    shutil.copyfile(creds_src, creds_dst)
    creds_dst.chmod(0o600)
    for name in ("google_accounts.json", "installation_id"):
        src = _GEMINI_DIR / name
        if src.exists():
            shutil.copyfile(src, dot_gemini / name)
    (dot_gemini / "settings.json").write_text(
        json.dumps(_GEMINI_SETTINGS), encoding="utf-8"
    )
    (tmpdir / "system.md").write_text(system_prompt, encoding="utf-8")


def _google_subprocess_env(tmpdir: Path) -> dict[str, str]:
    """Build the env for the ``gemini`` subprocess (hermetic + telemetry off)."""
    drop = {
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_GENAI_USE_GCA",
        "GEMINI_CLI_USE_COMPUTE_ADC",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(
        {
            "HOME": str(tmpdir),
            "USERPROFILE": str(tmpdir),
            "GEMINI_SYSTEM_MD": str(tmpdir / "system.md"),
            "GEMINI_DEFAULT_AUTH_TYPE": "oauth-personal",
            "GEMINI_FORCE_FILE_STORAGE": "true",
            "GEMINI_YOLO_MODE": "true",
            "GEMINI_FOLDER_TRUST": "true",
            "GEMINI_CLI_TRUST_WORKSPACE": "true",
            "GEMINI_CLI_NO_RELAUNCH": "true",
            "GEMINI_STRICT_TELEMETRY_LIMITS": "true",
        }
    )
    return env


_GEMINI_SETTINGS: MutableJSON = cast(
    MutableJSON,
    {
        "security": {"auth": {"selectedType": "oauth-personal"}},
        "privacy": {"usageStatisticsEnabled": False},
        "telemetry": {
            "enabled": False,
            "logPrompts": False,
            "useCollector": False,
        },
        "general": {
            "checkpointing": {"enabled": False},
            "enableAutoUpdate": False,
            "enableAutoUpdateNotification": False,
            "enableNotifications": False,
        },
        "ui": {
            "hideTips": True,
            "hideBanner": True,
            "hideContextSummary": True,
            "hideSandboxStatus": True,
            "hideModelInfo": True,
            "showMemoryUsage": False,
        },
        "context": {
            "loadMemoryFromIncludeDirectories": False,
            "discoveryMaxDirs": 0,
        },
        "tools": {
            "excludeTools": ["*"],
            "useWriteTodos": False,
            "toolSandboxing": False,
            "blockGitExtensions": True,
            "allowedExtensions": [],
        },
        "mcp": {"allowed": [], "excluded": ["*"]},
        "advanced": {
            "autoConfigureMemory": False,
            "agentSessionNoninteractiveEnabled": False,
            "agentSessionInteractiveEnabled": False,
            "extensionManagement": False,
            "extensionConfig": False,
            "extensionRegistry": False,
            "extensionReloading": False,
            "jitContext": False,
            "taskTracker": False,
            "modelSteering": False,
            "memoryV2": False,
            "autoMemory": False,
            "contextManagement": False,
        },
        "experimental": {"compressionThreshold": 1.0},
    },
)


async def _rpc_call(
    proc: Subproc,
    request_id: int,
    method: str,
    params: dict[str, object],
) -> MutableJSON:
    """Send one JSON-RPC request and return the matching ``result`` payload."""
    await _rpc_send(proc, request_id, method, params)
    while True:
        msg = await proc.read_json_line(skip_non_json=True)
        if msg is None:
            raise RuntimeError(f"GoogleCLI: stdout closed waiting for {method}")
        if msg.get("id") == request_id:
            if "error" in msg:
                raise RuntimeError(f"GoogleCLI: {method} error: {msg['error']}")
            return cast(MutableJSON, msg.get("result") or {})


async def _rpc_send(
    proc: Subproc,
    request_id: int,
    method: str,
    params: dict[str, object],
) -> None:
    """Serialise one JSON-RPC request and write it to the subprocess stdin."""
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }
    await proc.write_line(json.dumps(payload))


def _serialize_prompt_blocks(
    entry: HistoryEntry,
    max_image_dim: int,
) -> list[MutableJSON]:
    """Translate one non-assistant ``HistoryEntry`` into ACP prompt blocks."""
    if isinstance(entry, UserMessage):
        return _user_prompt_blocks(entry, max_image_dim)
    assert isinstance(entry, ToolResult)
    raise RuntimeError(
        "GoogleCLI: ToolResult in history -- tools must go through the MCP bridge",
    )


def _user_prompt_blocks(entry: UserMessage, max_image_dim: int) -> list[MutableJSON]:
    """Build ACP ``[{type:text}|{type:image}]`` blocks for a ``UserMessage``."""
    blocks: list[MutableJSON] = []
    if entry.text:
        blocks.append(cast(MutableJSON, {"type": "text", "text": entry.text}))
    for att in entry.attachments:
        if not att.descriptor.startswith("image/"):
            continue
        raw, mime = image_lib.resize(
            att.data, max_dim=max_image_dim, max_bytes=20 * 1024 * 1024
        )
        blocks.append(
            cast(
                MutableJSON,
                {
                    "type": "image",
                    "data": base64.b64encode(raw).decode(),
                    "mimeType": mime,
                },
            )
        )
    if not blocks:
        blocks.append(cast(MutableJSON, {"type": "text", "text": ""}))
    return blocks


def _dispatch_session_update(
    params: MutableJSON,
    text_parts: list[str],
    thinking_parts: list[str],
    on_text: Callable[[str], None] | None,
    on_thinking: Callable[[str], None] | None,
) -> None:
    """Route one ``session/update`` notification payload."""
    update = cast(MutableJSON, params.get("update") or {})
    kind = update.get("sessionUpdate")
    if kind == "agent_message_chunk":
        text = cast(
            str, cast(MutableJSON, update.get("content") or {}).get("text") or ""
        )
        if text:
            text_parts.append(text)
            if on_text is not None:
                on_text(text)
    elif kind == "agent_thought_chunk":
        text = cast(
            str, cast(MutableJSON, update.get("content") or {}).get("text") or ""
        )
        if text:
            thinking_parts.append(text)
            if on_thinking is not None:
                on_thinking(text)


def _estimate_input_tokens(
    history: list[HistoryEntry],
    model: _GoogleCLIModel,
) -> int:
    """Approximate input tokens from history text via the model's heuristic."""
    total = 0
    for entry in history:
        if isinstance(entry, UserMessage):
            total += model.estimate_text_token_count(entry.text)
            for att in entry.attachments:
                if att.descriptor.startswith("image/"):
                    total += model.estimate_image_token_count(att.data)
        elif isinstance(entry, AssistantMessage):
            total += model.estimate_text_token_count(entry.text)
    return total
