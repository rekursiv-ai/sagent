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
    _READ_IDLE_TIMEOUT_SEC,
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


class AnthropicCLIRetryableError(SubprocessTransportError):
    """The CLI returned a transient ``is_error`` ``result`` event.

    Distinguished from :class:`SubprocessTransportError` so the model's
    :meth:`is_retryable_provider_error` can flag the call for in-place
    retry by ``send_with_retry`` instead of propagating to the runtime
    as a fatal ``ModelResponseError`` (which would burn a turn boundary
    + pollute history with a synthetic ``[Error: ...]`` UserMessage).

    **Naming heritage and corrected understanding (2026-06-04 evening).**
    The ``aborted_streaming`` / ``ede_diagnostic`` shape was originally
    framed as "Anthropic-side stream instability under load." Live
    cross-correlation with peer-message timestamps that evening showed
    the dominant cause is actually our own SIGINT preempt
    (``runtime.preempt_in_flight=True``, override #2): when a peer or
    operator message arrives while a ``claude --print`` subprocess is
    mid-turn, we ``model.cancel_in_flight()`` to make room for the new
    inbound. The CLI subprocess dies and emits a final ``result`` event
    with ``terminal_reason="aborted_streaming"`` + an ede_diagnostic
    line -- exactly the shape we'd been attributing to upstream
    instability. The smoking gun was the 0-input-token events: an API
    can't abort a request that never reached it; only a local SIGINT
    can.

    So the dominant retry path here is: "operator typed a correction
    mid-turn, our preempt killed the in-flight subprocess, retry the
    delivery via ``--resume`` so the correction lands cleanly without
    burning a runtime turn boundary." Genuine upstream stream cuts
    exist but are a minority. The retry behaviour is correct either
    way; only the framing matters for future debugging.

    See :func:`_is_event_retryable` for the catalog of shapes that are
    treated as transient. Carries an optional ``retry_after_ms`` hint
    extracted from the CLI event when present; ``send_with_retry`` will
    fall back to standard exponential backoff if ``None``.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after_ms: float | None = None,
        event: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_ms = retry_after_ms
        self.event = event


def _is_event_retryable(event: dict) -> bool:
    """Classify a CLI ``result`` event as transient.

    Validated against ~60 ``is_error`` events captured on 2026-06-03/04
    from the live multi-agent server. The retryable shapes share two
    attributes: (a) claude's session JSONL on disk remains consistent
    (the next ``--resume`` will pick up cleanly), and (b) re-running
    the same request usually succeeds without operator intervention.

    Retryable shapes:

    * ``terminal_reason == "aborted_streaming"``: dominant pattern,
      often paired with ``stop_reason == "tool_use"`` and an
      ``[ede_diagnostic]`` entry in ``errors``. **Historically read as
      "Anthropic-side mid-stream cut"; the dominant cause is actually
      our own preempt SIGINT** (see :class:`AnthropicCLIRetryableError`
      for the diagnosis). 0-input-token instances are unambiguous
      preempt kills (the API never saw the request).
    * ``errors`` contains an ``[ede_diagnostic]`` line: same surface
      shape, occasionally reported with ``terminal_reason ==
      "completed"`` when the SIGINT raced the natural stop.

    NOT retryable:

    * ``terminal_reason == "blocking_limit"``: context overflow. The
      next attempt would hit the same wall; operator must clear the
      session. Surfacing as a hard error gets the runtime to publish
      ``ModelResponseError`` so the operator sees ``status=hung``.
    * ``api_error_status`` in non-429 4xx: client-side issue, retry
      won't help.

    The classification doesn't depend on root cause: retryable shapes
    recover cleanly via ``--resume`` regardless of whether the abort
    came from upstream or from our own SIGINT. Only the framing in
    docs/logs matters for future debugging.
    """
    terminal_reason = event.get("terminal_reason")
    if terminal_reason == "aborted_streaming":
        return True
    if terminal_reason == "blocking_limit":
        return False
    errors = event.get("errors", [])
    if isinstance(errors, list):
        for err in errors:
            if "ede_diagnostic" in str(err):
                return True
    return False


def _extract_retry_after_ms(event: dict) -> float | None:
    """Pull a millisecond retry hint out of the CLI ``result`` event.

    The CLI sometimes embeds ``retry_after_ms`` / ``retry_delay_ms``
    keys in the result envelope (mirroring the underlying API
    ``retry-after`` header). When present we forward it to the retry
    layer so backoff respects the server's hint; when absent
    ``send_with_retry`` falls back to its standard exponential schedule.
    """
    for key in ("retry_after_ms", "retry_delay_ms"):
        val = event.get(key)
        if isinstance(val, (int, float)) and val >= 0:
            return float(val)
    return None


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
        *,
        extra_mcp_servers: dict[str, dict] | None = None,
        session_id: str | None = None,
        subprocess_read_timeout_sec: float | None = None,
    ) -> _AnthropicCLIModel:
        """Build a CLI-backed model.

        Args:
          model_id: Claude model id; ``None`` uses ``DEFAULT_MODEL``.
          max_request_tokens: Override the profile's input cap.
          extra_mcp_servers: Additional MCP servers (stdio or HTTP)
            merged into the CLI's ``--mcp-config`` at subprocess spawn
            time. Each entry follows Claude Code's mcp.json shape:
            ``{"command": "...", "args": [...], "env": {...}}`` for
            stdio or ``{"type": "http", "url": "..."}`` for HTTP. Keys
            colliding with sagent's own bridge server name are dropped.
          session_id: When set, every ``claude`` subprocess is spawned
            with ``--session-id <uuid>`` (first turn) or
            ``--resume <uuid>`` (subsequent turns) instead of
            ``--no-session-persistence``. Sagent stops re-feeding
            history via stdin: only the latest user-like inbound is
            sent each turn, and ``claude`` itself owns the
            conversation transcript at
            ``~/.claude/projects/-<encoded-cwd>/<uuid>.jsonl``. This
            mode is intended for chat-channel use cases where
            ``aborted_streaming`` recoveries must NOT lose
            ``AssistantMessage`` content — see
            ``plugin/blackjax-chat/README.md`` for context.
          subprocess_read_timeout_sec: Stdout-idle timeout (seconds)
            for the ``claude`` subprocess transport. ``None`` keeps
            the ``Subproc`` default (60s). Set higher when the
            agent's tools include long-running synchronous Bash
            commands (e.g. ``pre-commit run --files ...``,
            ``ty check``, big test suites) that legitimately go
            silent for >60s while claude awaits the result —
            without a bump, the transport reads silence as a hang,
            raises ``SubprocessTransportError``, and
            ``send_with_retry`` may emit a divergence marker that
            eats the closing assistant message. Verified live
            2026-06-09 on SWE's PR3 ``pre-commit + git commit``
            chain.

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
            extra_mcp_servers=extra_mcp_servers,
            session_id=session_id,
            subprocess_read_timeout_sec=subprocess_read_timeout_sec,
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
        extra_mcp_servers: dict[str, dict] | None = None,
        session_id: str | None = None,
        subprocess_read_timeout_sec: float | None = None,
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
        # Stdout-idle timeout (seconds) for the ``claude`` subprocess
        # transport. ``None`` defers to the ``Subproc`` default (60s).
        # Bumped to ~3-5min for v2 plugin agents whose tools include
        # long-running ``pre-commit run`` / ``ty check`` / heavy test
        # invocations; without the bump, a >60s tool wait makes the
        # transport read claude's silence as a hang and triggers
        # ``send_with_retry``, whose retried response often diverges
        # from the cached partial and eats the closing assistant
        # message. See worklog ``v2.1-cli-session-materialize`` →
        # 2026-06-09 SWE PR3 divergence diagnosis.
        self._subprocess_read_timeout_sec: float | None = subprocess_read_timeout_sec
        # Session-persistence mode (see ``AnthropicCLI.model``'s
        # ``session_id`` arg). When set:
        #   * ``--session-id <uuid>`` is passed on the first turn,
        #     ``--resume <uuid>`` thereafter. ``claude`` owns history
        #     at ``~/.claude/projects/-<encoded-cwd>/<uuid>.jsonl``.
        #   * Sagent no longer re-feeds history via stdin: only the
        #     latest user-like inbound is sent per turn.
        #   * HotSpare is bypassed -- each ``stream`` call spawns a
        #     fresh subprocess. Pre-warming a spare with
        #     ``--resume <same-uuid>`` would race against in-flight
        #     turns updating the session file (see the
        #     ``/tmp/resume_probe/`` test A from 2026-06-02 evening:
        #     concurrent ``--resume`` branches the conversation tree).
        self._session_id: str | None = session_id
        # ``_session_initialized = True`` means "claude already has a
        # session JSONL for this uuid on disk, use ``--resume``"; False
        # means "first time we've seen this uuid, use ``--session-id``".
        # We probe the operator's real ``~/.claude/projects/`` at
        # construction time so server restarts pick up prior
        # conversations transparently.
        self._session_initialized: bool = (
            session_id is not None and _session_jsonl_exists(session_id)
        )
        if session_id is None:
            self._hot_spare: HotSpare | None = HotSpare(
                self._spawn_spare_initialized,
                close_partial=self._close_warming_proc,
            )
            # Stateless mode mints a fresh tmpdir per spawn for
            # credential isolation; the per-spawn tmpdir is local to
            # ``_spawn_initialized``.
            self._persistent_tmpdir: Path | None = None
        else:
            self._hot_spare = None
            # Session-persistence mode tradeoff on the ``HOME`` env var:
            #
            # If we override HOME to a hermetic tmpdir (as stateless
            # mode does for credential isolation), then claude's
            # NATIVE tools -- ``Bash``, ``Read``, ``Write``, etc. --
            # run with that hermetic HOME inside the ``claude --print``
            # subprocess. They lose visibility into the operator's
            # ``~/.config/gh/hosts.yml``, ``~/.gitconfig``, ssh keys,
            # etc. This shows up as ``gh auth status`` reporting "not
            # authenticated" even though the operator IS authenticated
            # on the host. (In stateless mode the same tools were
            # mounted via sagent's HTTP bridge, so their handler ran
            # IN THE SAGENT SERVER PROCESS with the operator's real
            # HOME -- mooting the issue.)
            #
            # In session-persistence mode + single-account
            # (``provider.account is None``), we drop the HOME
            # override entirely. ``claude`` reads its credentials
            # from the operator's real ``~/.claude/.credentials.json``,
            # native tools find ``~/.config/gh/``, and session
            # JSONLs land at the operator's real
            # ``~/.claude/projects/-<encoded-cwd>/<uuid>.jsonl``,
            # which survives ``serve.py`` restarts -- so re-running
            # the server now ``--resume``s the prior conversation
            # transcripts cleanly.
            #
            # For per-account use (``provider.account is not None``)
            # we still mint a tmpdir + populate the renamed
            # credentials file, because ``claude``'s creds path is
            # hardcoded to ``$HOME/.claude/.credentials.json`` --
            # there's no env var to redirect it. That case keeps the
            # stateless-mode HOME-override behaviour.
            if self._provider.account is None:
                self._persistent_tmpdir = None
            else:
                self._persistent_tmpdir = Path(
                    tempfile.mkdtemp(prefix="sagent-anthropic-cli-resume-"),
                )
                _populate_anthropic_tmpdir(
                    self._persistent_tmpdir,
                    self._provider.account,
                )
        self._active_proc: Subproc | None = None
        # Set by ``stream`` before ``_spawn_initialized`` reads them.
        self._pending_system: str = ""
        self._sent_history_head: TapeEvent | None = None
        # External MCP servers (stdio or HTTP) merged into the CLI's
        # ``--mcp-config`` at subprocess spawn time. See
        # :func:`_build_anthropic_argv`.
        self._extra_mcp_servers = extra_mcp_servers

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
        """Session-persistent mode flags transient ``is_error`` results
        as retryable so ``send_with_retry`` performs an in-place retry
        (sleep → spawn fresh ``claude --print --resume`` → process only
        the entries the per-entry-advance ``_last_sent_index`` hasn't
        delivered yet) instead of letting the error propagate up to
        the runtime as a ``ModelResponseError`` (which appends a
        synthetic ``[Error: …]`` UserMessage to history and costs a
        full turn boundary).

        Stateless mode keeps the historical ``return False`` because
        its warm ``HotSpare`` subprocess has already consumed the
        stdin lines we wrote; same-subprocess retry would either
        duplicate inbound messages or stall waiting on a CLI that no
        longer expects more input. The runtime's respawn path handles
        that case correctly by resetting ``_last_sent_index = 0`` and
        re-feeding history from scratch.

        Only :class:`AnthropicCLIRetryableError` qualifies; plain
        :class:`SubprocessTransportError` (subprocess died, stdout
        closed, blocking_limit context overflow) still propagates so
        the operator sees the failure.
        """
        if isinstance(error, AnthropicCLIRetryableError):
            return self._session_id is not None
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
        if self._session_id is not None:
            return await self._stream_session_persistent(
                request,
                on_text,
                on_thinking,
            )
        assert self._hot_spare is not None  # stateless path
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

    async def _stream_session_persistent(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None,
        on_thinking: Callable[[str], None] | None,
    ) -> ModelResponse:
        """Drive one turn through a ``--session-id`` / ``--resume`` subprocess.

        Spawn-on-demand (no HotSpare) because pre-warming a spare with
        ``--resume <same-uuid>`` would race against the active
        subprocess updating the session file (see
        ``/tmp/resume_probe/`` test A from 2026-06-02: concurrent
        ``--resume`` produces a branched conversation tree).

        Only the newest user-like entries are written to stdin --
        ``claude`` already has the rest in its session file.
        ``_last_sent_index`` is the cumulative count of messages we've
        delivered to ``claude`` across this session_id, NOT a per-
        subprocess counter -- so it's preserved across transport-error
        respawns. On the first turn we use ``--session-id``; on every
        subsequent turn (including respawns) we use ``--resume``.
        """
        # ``agent.clear()`` (driven by ``/api/restart``, ``Clear`` event,
        # or context-overflow recovery) wipes ``runtime.context().messages``
        # but doesn't reach into the provider's ``_last_sent_index`` /
        # ``_session_initialized``. After clear, the runtime keeps calling
        # ``stream()`` -- with an empty ``request.messages`` until new
        # input arrives. The stateless path handles this in
        # ``_should_respawn`` (history shrunk → respawn → reset
        # counters); we have to do the equivalent here. Otherwise:
        #
        # * The defensive "no new user-like entries" guard fires and
        #   crashes the runtime turn.
        # * Even if we let an empty turn through, the next real
        #   ``--session-id <uuid>`` call would fail with "Session ID
        #   is already in use" because claude's prior session JSONL
        #   is still on disk.
        #
        # Detect via ``self._last_sent_index > len(request.messages)``
        # (cumulative-count > current-history-length is only possible
        # after a clear). Reset state, delete the stale on-disk JSONL,
        # then continue as if this is a fresh session.
        if self._last_sent_index > len(request.messages):
            self._reset_for_clear()

        new_entries = request.messages[self._last_sent_index :]
        # user-like entries only: filter out AssistantMessage / ToolResult
        # (sagent's own history bookkeeping, never written to stdin).
        # We build the index map below at the same time we'd otherwise
        # build the list.
        if not any(
            not isinstance(e, (AssistantMessage, ToolResult)) for e in new_entries
        ):
            # No new input to feed. Return a no-op response: empty
            # assistant message + zero usage. The runtime treats this
            # as a finished turn with no output; the next real inbound
            # will spawn the next subprocess. Cheaper than spawning a
            # claude --print just to wait for stdin EOF.
            return ModelResponse(
                message=AssistantMessage(text="", tool_calls=()),
                stop_reason="model_finished",
            )
        # Bridge MUST be populated before the subprocess spawns: the
        # CLI issues ``ListToolsRequest`` against the bridge soon
        # after launch, and if our tool catalog isn't there at that
        # moment, opus falls back to emitting tool calls as plain
        # text inside the assistant message (Episode 2.7 pathology
        # — observed 2026-06-02 23:16 when TL produced
        # "Bash {command: ls -la …}" as text instead of a tool_use
        # block on the first turn after refactor).
        # Per-entry index map: each user-like entry's position in
        # ``new_entries`` (which is ``request.messages[base:]``). We
        # use this to advance ``_last_sent_index`` per-entry as the
        # writes succeed, so a mid-loop ``SubprocessTransportError``
        # doesn't lose the entries we already wrote and doesn't force
        # the next retry to re-send them.
        new_entries_idx: list[tuple[int, TapeEvent]] = []
        for i, entry in enumerate(new_entries):
            if not isinstance(entry, (AssistantMessage, ToolResult)):
                new_entries_idx.append((i, entry))
        base = self._last_sent_index

        await self._ensure_tools_bridge()
        self._sync_tools_bridge(request)
        proc = await self._spawn_initialized()
        self._active_proc = proc
        try:
            for rel_idx, entry in new_entries_idx[:-1]:
                await self._send_entry(proc, entry)
                # Advance per-entry BEFORE the drain: once a line is on
                # claude's stdin, claude will consume it + persist it to
                # the session JSONL even if the resulting model_call
                # aborts. If the drain fails, the next ``--resume`` MUST
                # NOT re-write this entry (that produced the "Great
                # smoke 3×" duplication observed in SWE's session log
                # on 2026-06-03 around 14:30, when each retry re-wrote
                # the earliest pending entry AND failed to reach the
                # later entries that contained TL's STOP directives).
                self._last_sent_index = base + rel_idx + 1
                _ = await self._drain_until_result(
                    proc,
                    on_text=None,
                    on_thinking=None,
                    update_input_tokens=False,
                )
            last_rel_idx, last_entry = new_entries_idx[-1]
            await self._send_entry(proc, last_entry)
            self._last_sent_index = base + last_rel_idx + 1
            response = await self._drain_until_result(
                proc,
                on_text,
                on_thinking,
            )
        except SubprocessTransportError:
            # ``claude`` died mid-turn. ``_last_sent_index`` already
            # reflects every entry we wrote to stdin; the next
            # ``--resume`` will skip them and pick up from the first
            # entry we hadn't reached yet. The session JSONL on disk
            # was updated by claude as it processed each line, so the
            # next turn's view is consistent.
            await proc.close()
            self._active_proc = None
            raise
        else:
            # All entries delivered + final drain returned cleanly.
            # Advance past any trailing AssistantMessage / ToolResult
            # entries (sagent's own bookkeeping; we never write them).
            self._last_sent_index = len(request.messages)
            # First successful turn established the session on disk;
            # all future spawns use ``--resume``.
            self._session_initialized = True
            await proc.close()
            self._active_proc = None
            self._turn_count += 1
            return response

    async def close(self) -> None:
        """Tear down the subprocess pool and the MCP bridge."""
        if self._hot_spare is not None:
            await self._hot_spare.close()
        if self._active_proc is not None:
            await self._active_proc.close()
            self._active_proc = None
        if self._tools_bridge is not None:
            await self._tools_bridge.stop()
            self._tools_bridge = None
        if self._persistent_tmpdir is not None and self._persistent_tmpdir.exists():
            shutil.rmtree(self._persistent_tmpdir, ignore_errors=True)
            self._persistent_tmpdir = None

    def _should_respawn(self, request: ModelRequest) -> bool:
        """Inspect the trigger list (§1.4) for this request."""
        assert self._hot_spare is not None  # caller is the stateless path
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

    async def _ensure_tools_bridge(self) -> None:
        """Lazily create the MCP bridge (without spawning the CLI).

        The bridge MUST exist before the first ``claude --print``
        subprocess starts -- the CLI does ``ListToolsRequest`` against
        the bridge URL soon after launch, and an empty bridge produces
        an empty tool catalog (which then makes opus emit tool calls
        as plain text). In the stateless path this is implicit because
        the HotSpare's first spawn calls ``_spawn_initialized`` which
        creates the bridge; in session-persistent mode we must hoist
        the bridge creation out so the spawn argv can use a real
        ``--mcp-config`` URL.
        """
        if self._tools_bridge is None:
            self._tools_bridge = ToolsBridge(tools=[])
            await self._tools_bridge.start()

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
        # Aggregated thinking-block signature. Anthropic's stream
        # emits a ``signature_delta`` event alongside ``thinking_delta``;
        # the final signature is the concatenation of those deltas
        # (typically a single delta). Required because Anthropic's API
        # rejects an assistant message with a signature-less thinking
        # block (HTTP 400 ``thinking.signature: Field required``).
        # Pre-2026-06-09 this parser ignored signature_delta and v2's
        # session-persistent path didn't notice — claude owned the
        # JSONL and never re-sent unsigned thinking blocks via wire.
        # v2.1-α materialize mode IS the wire-resender, so the
        # signature became load-bearing.
        signature_parts: list[str] = []
        # Per-block-index state for tool_use accumulators. Each
        # ``content_block_start`` for a ``tool_use`` registers
        # ``{name, id, json_parts: list[str]}`` keyed by block index;
        # streamed ``input_json_delta`` deltas append to ``json_parts``;
        # ``content_block_stop`` finalises and emits a ToolLabel with
        # the parsed args.
        tool_use_blocks: dict[int, dict[str, object]] = {}
        usage_event: MutableJSON | None = None
        # Usage of the LAST internal round's request (raw Anthropic API
        # shape, snake_case). One ``claude --print`` turn runs the whole
        # tool loop inside the subprocess -- N internal API rounds -- and
        # the terminal ``result`` event sums input + cache tokens across
        # ALL of them, so the result usage over-counts the live context
        # window by ~the round count (observed 2026-06-09: a 69-round
        # turn reported 5.6M "input" tokens against a 200k window,
        # spuriously tripping the Agent's proactive compaction gate).
        # Each round's ``message_start`` carries that round's request
        # usage; the last one IS the current context footprint -- the
        # same thing the direct-API provider reports. Captured here and
        # used to normalize ``ModelResponse.tokens``'s input side; the
        # output side stays cumulative (output genuinely accumulates
        # across rounds), and billing is unaffected either way
        # (``total_cost`` comes from ``modelUsage.costUSD``, summed).
        last_round_usage: MutableJSON | None = None
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
                    # Classify transient shapes for in-place retry vs
                    # fatal shapes for runtime escalation. The
                    # dominant transient shape in practice is the
                    # ``aborted_streaming`` event the CLI emits when
                    # we ``model.cancel_in_flight()`` to preempt a
                    # mid-turn subprocess for an incoming peer/operator
                    # message; the retry then re-delivers via
                    # ``--resume``. See :func:`_is_event_retryable`
                    # for the catalog.
                    if _is_event_retryable(cast(dict, event)):
                        raise AnthropicCLIRetryableError(
                            f"AnthropicCLI: retryable result is_error: {event}",
                            retry_after_ms=_extract_retry_after_ms(
                                cast(dict, event),
                            ),
                            event=cast(dict, event),
                        )
                    raise SubprocessTransportError(
                        f"AnthropicCLI: result is_error: {event}"
                    )
                break
            if kind == "stream_event":
                inner = cast(MutableJSON, event.get("event") or {})
                if inner.get("type") == "message_start":
                    msg = cast(MutableJSON, inner.get("message") or {})
                    usage = msg.get("usage")
                    if isinstance(usage, dict):
                        last_round_usage = cast(MutableJSON, usage)
                _dispatch_stream_event(
                    inner,
                    text_parts,
                    thinking_parts,
                    signature_parts,
                    tool_use_blocks,
                    on_text,
                    on_thinking,
                )
            elif kind == "system" and event.get("subtype") == "init":
                message_id = cast(str, event.get("session_id") or "")
        assert usage_event is not None
        if update_input_tokens:
            # Cache-inclusive context footprint of the last internal
            # round; feeds the context-fraction respawn heuristic.
            self._last_input_tokens = _round_context_tokens(last_round_usage)
        return _build_model_response(
            usage_event=usage_event,
            last_round_usage=last_round_usage,
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
        spawn_owned_tmpdir: Path | None
        tmpdir: Path | None
        if self._persistent_tmpdir is not None:
            # Session-persistence + per-account: reuse the
            # construction-time tmpdir so the renamed credentials
            # file is found by claude (its creds path is hardcoded
            # to $HOME/.claude/.credentials.json).
            tmpdir = self._persistent_tmpdir
            spawn_owned_tmpdir = None
        elif self._session_id is not None:
            # Session-persistence + single account: NO HOME override.
            # The subprocess inherits the operator's real HOME so
            # native tools (``Bash``-from-shell, ``gh``, ``git``,
            # ssh, ...) find ``~/.config/``, ``~/.gitconfig``, etc.,
            # AND ``claude`` reads + writes session JSONLs at the
            # operator's real ``~/.claude/projects/`` (which
            # survives ``serve.py`` restarts -- a free upgrade).
            tmpdir = None
            spawn_owned_tmpdir = None
        else:
            # Stateless mode: hermetic per-spawn tmpdir for
            # credential isolation. The Subproc wrapper deletes it
            # on close.
            tmpdir = Path(tempfile.mkdtemp(prefix="sagent-anthropic-cli-"))
            _populate_anthropic_tmpdir(tmpdir, self._provider.account)
            spawn_owned_tmpdir = tmpdir
        argv = _build_anthropic_argv(
            model_id=base_model_id(self._model_id),
            system_prompt=self._pending_system,
            bridge_url=self._tools_bridge.url,
            bridge_server_name=self._tools_bridge.server_name,
            extra_mcp_servers=self._extra_mcp_servers,
            session_id=self._session_id,
            # ``--resume`` once we've successfully spawned and ack'd
            # at least one turn under this session_id; ``--session-id``
            # otherwise. Updated in the session-persistence stream
            # path on first successful drain.
            resume_existing=self._session_initialized,
        )
        env = _anthropic_subprocess_env(
            tmpdir,
            persist_session=self._session_id is not None,
        )
        # ``None`` defers to ``Subproc``'s own default (60s); an explicit
        # value (set by the plugin for long pre-commit/ty/JAX tool calls)
        # overrides it. Pass the resolved value either way.
        read_timeout = (
            self._subprocess_read_timeout_sec
            if self._subprocess_read_timeout_sec is not None
            else _READ_IDLE_TIMEOUT_SEC
        )
        proc = Subproc(
            argv,
            env=env,
            tmpdir=spawn_owned_tmpdir,
            read_timeout_sec=read_timeout,
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

    def _reset_for_clear(self) -> None:
        """Reset session-persistent state after ``agent.clear()``.

        Wipes the cumulative-sent counter, marks the session as
        uninitialised (next spawn will use ``--session-id`` again),
        and DELETES the on-disk session JSONL so the next
        ``--session-id <same-uuid>`` call doesn't error with
        "Session ID is already in use".

        Called from :meth:`_stream_session_persistent` when it
        detects ``_last_sent_index > len(request.messages)``, which
        is the post-``Clear`` shape.
        """
        if self._session_id is None:
            return
        self._last_sent_index = 0
        self._session_initialized = False
        # Find + delete the session JSONL. Path:
        #   <HOME>/.claude/projects/-<encoded-cwd>/<uuid>.jsonl
        # where HOME is either the persistent tmpdir (per-account
        # mode) or the operator's real $HOME (single-account mode).
        home = self._persistent_tmpdir or Path(os.environ.get("HOME", "~")).expanduser()
        projects = home / ".claude" / "projects"
        if not projects.exists():
            return
        # The cwd-encoded subdir is opaque to us (claude picks the
        # encoding); glob across all subdirs for safety.
        for jsonl in projects.glob(f"*/{self._session_id}.jsonl"):
            try:
                jsonl.unlink()
                logger.info(
                    "AnthropicCLI(session_persistent): cleared session "
                    "JSONL at %s after agent.clear()",
                    jsonl,
                )
            except OSError as exc:
                logger.warning(
                    "AnthropicCLI(session_persistent): failed to delete %s: %s",
                    jsonl,
                    exc,
                )

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


def _session_jsonl_exists(session_id: str) -> bool:
    """Check whether claude already has a session transcript for this
    uuid on disk, under THIS cwd's encoded project dir.

    Used at construction time so that on a host-application restart we
    pick the right initial flag: ``--resume`` (the JSONL exists, we
    continue the prior conversation) vs ``--session-id`` (no prior
    session, we establish a fresh one). Picking the wrong one makes
    ``claude --print`` exit non-zero before consuming stdin --
    ``--session-id`` on an existing session errors with "Session ID
    is already in use", ``--resume`` on a nonexistent one errors
    with "No conversation found".

    The check must be cwd-AWARE: claude indexes sessions per
    encoded-cwd project dir (``~/.claude/projects/-<encoded-cwd>/``)
    and ``--resume`` cannot see a session recorded under a different
    cwd. The original implementation globbed ACROSS project dirs
    ("the encoding is opaque to us") -- a latent bug once two
    deployments derive the same deterministic session uuid: live repro
    2026-06-09, where a second server instance launched from a scratch
    cwd globbed the primary deployment's JSONL into ``True``, spawned
    ``--resume``, and claude exited ``No conversation found``, wedging
    warmup for all five agents. The encoding mirrors the CLI's:
    every character of the absolute cwd outside ``[A-Za-z0-9-]``
    becomes ``-``; ask the question claude will answer: does the
    JSONL exist under THIS cwd's encoded project dir.
    """
    home = Path(os.environ.get("HOME", "~")).expanduser()
    encoded = "".join(
        ch if (ch.isalnum() or ch == "-") else "-" for ch in str(Path.cwd())
    )
    return (home / ".claude" / "projects" / encoded / f"{session_id}.jsonl").exists()


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


def _anthropic_subprocess_env(
    tmpdir: Path | None,
    *,
    persist_session: bool = False,
) -> dict[str, str]:
    """Build the env for the ``claude`` subprocess (telemetry off, hermetic HOME).

    When ``persist_session=True`` we keep ``CLAUDE_CODE_SKIP_PROMPT_HISTORY``
    unset -- that env var (verified by bisect 2026-06-02) causes the CLI
    to skip writing its session JSONL even when ``--session-id`` /
    ``--resume`` are passed, which makes session-persistence mode silently
    no-op and the next ``--resume`` fails with "No conversation found".

    When ``tmpdir is None`` we don't override ``HOME`` -- the subprocess
    inherits the operator's real HOME so claude finds its real
    credentials + project session JSONLs AND its native tools (Bash,
    gh, git, ...) find ``~/.config/`` and ``~/.gitconfig`` naturally.
    Used by the session-persistent + single-account path.

    """
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    if tmpdir is not None:
        env["HOME"] = str(tmpdir)
        env["USERPROFILE"] = str(tmpdir)
    env.update(
        {
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_BUG_COMMAND": "1",
            "DISABLE_FEEDBACK_COMMAND": "1",
            "DISABLE_COST_WARNINGS": "1",
            "DISABLE_INSTALLATION_CHECKS": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_LEGACY_MODEL_REMAP": "1",
            "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
        }
    )
    # Auto-compact: disable in stateless mode where sagent owns
    # history; enable in session-persistent mode where claude is the
    # sole writer.
    sagent_owns_history = not persist_session
    if sagent_owns_history:
        env["DISABLE_AUTO_COMPACT"] = "1"
    if not persist_session:
        env["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] = "1"
    return env


def _build_anthropic_argv(
    *,
    model_id: str,
    system_prompt: str,
    bridge_url: str,
    bridge_server_name: str,
    extra_mcp_servers: dict[str, dict] | None = None,
    session_id: str | None = None,
    resume_existing: bool = False,
) -> list[str]:
    """Assemble the ``claude --print --input-format stream-json ...`` argv.

    When ``session_id`` is ``None`` (default), passes
    ``--no-session-persistence`` -- the historical behaviour. Otherwise
    passes ``--session-id <uuid>`` (``resume_existing=False``) or
    ``--resume <uuid>`` (``resume_existing=True``). Misusing this
    (``--session-id`` on an existing session, ``--resume`` on a
    nonexistent one) makes ``claude`` exit non-zero before consuming
    stdin, which sagent surfaces as ``SubprocessTransportError``.

    ``extra_mcp_servers``, when provided, is merged into the
    ``mcpServers`` block of the JSON written to ``--mcp-config``. Use
    this to register stdio/HTTP MCP servers alongside sagent's own
    in-process tool bridge. Caller is responsible for ensuring the
    keys don't collide with ``bridge_server_name``; sagent's bridge
    wins on conflict.
    """
    servers: dict[str, dict] = {
        bridge_server_name: {"type": "http", "url": bridge_url},
    }
    if extra_mcp_servers:
        for name, entry in extra_mcp_servers.items():
            if name == bridge_server_name:
                # Don't let an external entry stomp on the bridge that
                # exposes sagent-native tools (Read/Bash/etc.).
                continue
            servers[name] = entry
    mcp_config = json.dumps({"mcpServers": servers})
    base = [
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
    ]
    if session_id is None:
        # Stateless: every spawn is a fresh session, history is fed
        # via stdin by ``_exchange_turn``.
        base.append("--no-session-persistence")
    elif resume_existing:
        # Session-persistence mode, claude has the session on disk.
        base.extend(["--resume", session_id])
    else:
        # Session-persistence mode, first time we see this UUID.
        base.extend(["--session-id", session_id])
    base.extend(
        [
            "--setting-sources",
            "",
            "--mcp-config",
            mcp_config,
            "--strict-mcp-config",
        ]
    )
    # ``--tools ""`` historically meant "use the default allowlist" in
    # stateless mode. In session-persistence mode (``--session-id`` /
    # ``--resume``) the CLI re-interprets empty-string as "ALLOW NO
    # TOOLS, including MCP ones" -- bisect probe 2026-06-03 found:
    #
    #   --tools ""              → no tool_use; opus replies "I'll run
    #                             `ls /tmp`" as plain text.
    #   (omit --tools entirely) → tool_use(name=Bash) emitted cleanly.
    #   --tools "Bash"          → tool_use(name=Bash) emitted cleanly.
    #
    # We omit ``--tools`` in session-persistence mode and keep the
    # empty-string in stateless mode (no observed regressions there).
    if session_id is None:
        base.extend(["--tools", ""])
    base.extend(
        [
            "--disable-slash-commands",
            "--permission-mode",
            "bypassPermissions",
        ]
    )
    return base


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
    tool_use_blocks: dict[int, dict[str, object]],
    on_text: Callable[[str], None] | None,
    on_thinking: Callable[[str], None] | None,
) -> None:
    """Route one stream_event payload to ``on_text`` / ``on_thinking``.

    Also accumulates ``tool_use`` content blocks across their start /
    streamed ``input_json_delta`` chunks / stop events, and emits one
    ``ToolLabel`` per tool call at block-stop with ``name`` plus a
    short rendering of the JSON args (e.g. ``Bash ls -la`` or
    ``Read foo.py``). Published via :data:`cli_publish_var` so the
    trace panel surfaces what tools the model is invoking. Covers
    both bridge-mounted tools (Bash, Read, ...) and external-MCP
    tools (``mcp__sagent_chat__sagent_send``, ...) uniformly --
    in stateless mode the bridge ALSO publishes labels for its own
    tools, so bridge-mounted tools get logged twice; the trace
    renderer treats each ToolLabel as a separate event.
    """
    event_type = event.get("type")
    if event_type == "content_block_start":
        idx = int(cast(int, event.get("index") or 0))
        block = cast(MutableJSON, event.get("content_block") or {})
        if block.get("type") == "tool_use":
            tool_use_blocks[idx] = {
                "name": cast(str, block.get("name") or "?"),
                "id": cast(str, block.get("id") or ""),
                "json_parts": [],
            }
        return
    if event_type == "content_block_delta":
        delta = cast(MutableJSON, event.get("delta") or {})
        delta_type = delta.get("type")
        if delta_type == "input_json_delta":
            idx = int(cast(int, event.get("index") or 0))
            state = tool_use_blocks.get(idx)
            if state is not None:
                partial = cast(str, delta.get("partial_json") or "")
                cast(list, state["json_parts"]).append(partial)
            return
        if delta_type == "text_delta":
            text = cast(str, delta.get("text") or "")
            if text:
                text_parts.append(text)
                if on_text is not None:
                    on_text(text)
            return
        if delta_type == "thinking_delta":
            text = cast(str, delta.get("thinking") or "")
            if text:
                thinking_parts.append(text)
                if on_thinking is not None:
                    on_thinking(text)
            return
        if delta_type == "signature_delta":
            # Per Anthropic's stream-json spec, ``signature_delta``
            # carries the opaque thought-signature in the ``signature``
            # field (mirrors ``thinking_delta`` for body text). The
            # final signature is the concatenation across deltas
            # (typically a single delta in practice). Required so
            # downstream wire sends (v2.1-α materializer mode) embed
            # the signature in the thinking block — Anthropic's API
            # rejects unsigned thinking with HTTP 400
            # ``thinking.signature: Field required``.
            sig = cast(str, delta.get("signature") or "")
            if sig:
                signature_parts.append(sig)
            return
        return
    if event_type == "content_block_stop":
        idx = int(cast(int, event.get("index") or 0))
        state = tool_use_blocks.pop(idx, None)
        if state is None:
            return
        tool_name = cast(str, state["name"])
        tool_id = cast(str, state["id"])
        json_parts = cast(list, state["json_parts"])
        args_summary = _render_tool_args(tool_name, "".join(json_parts))
        label_text = f"{tool_name} {args_summary}".rstrip()
        try:
            # Lazy: a module-level import would cycle (agent.runtime
            # transitively imports the providers package).
            from sagent.agent.runtime import cli_publish_var  # noqa: PLC0415
            from sagent.types.runtime import ToolLabel  # noqa: PLC0415

            publish = cli_publish_var.get()
            if publish is not None:
                publish(ToolLabel(call_id=tool_id, text=label_text))
        except Exception:  # noqa: BLE001 -- never let a label publish break the stream
            logger.debug("failed to publish ToolLabel for %r", tool_name, exc_info=True)
        return


def _render_tool_args(name: str, raw_json: str) -> str:
    """Render tool input JSON as a short label suffix.

    Best-effort: if the JSON is incomplete (streaming aborted mid-flight)
    or unparseable, falls back to a truncated raw form. Common args
    (``command``, ``file_path``, ``pattern``, ``query``, ``to``,
    ``content``) get a friendly rendering; unknown tools fall back to
    the raw arg dict.
    """
    del name  # reserved for future per-tool renderings
    if not raw_json:
        return ""
    try:
        args = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError):
        snippet = raw_json[:80].replace("\n", " ")
        return f"({snippet}…)" if len(raw_json) > 80 else f"({snippet})"
    if not isinstance(args, dict):
        return str(args)[:80]
    for key in ("command", "file_path", "path", "pattern", "query", "to"):
        if key in args and isinstance(args[key], str):
            val = args[key]
            return val if len(val) <= 120 else val[:120] + "…"
    if "content" in args and isinstance(args["content"], str):
        val = args["content"]
        return val if len(val) <= 120 else val[:120] + "…"
    rendered = ", ".join(f"{k}={str(v)[:40]!r}" for k, v in list(args.items())[:3])
    return rendered[:120]


def _round_context_tokens(round_usage: MutableJSON | None) -> int:
    """Cache-inclusive input footprint of one internal round's request.

    ``round_usage`` is the raw Anthropic API ``usage`` object off a
    ``message_start`` stream event (snake_case keys — unlike the
    camelCase rows in the CLI's terminal ``result.modelUsage``). The
    sum of non-cached input plus both cache pools is the full prompt
    size the server counted for that request — the same number the
    direct-API provider's per-request usage reports. Returns 0 when no
    round was observed (defensive; a successful drain always sees at
    least one ``message_start``), which downstream consumers treat as
    "unknown — estimate instead".
    """
    if round_usage is None:
        return 0
    return (
        int_val(round_usage.get("input_tokens"), 0)
        + int_val(round_usage.get("cache_creation_input_tokens"), 0)
        + int_val(round_usage.get("cache_read_input_tokens"), 0)
    )


def _build_model_response(
    *,
    usage_event: MutableJSON,
    last_round_usage: MutableJSON | None,
    text: str,
    thinking_parts: list[str],
    signature_parts: list[str],
    stop_reason: str | None,
    fallback_message_id: str,
) -> ModelResponse:
    """Assemble a ``ModelResponse`` with normalized token semantics.

    One ``claude --print`` turn is N internal API rounds, so the
    terminal ``result`` event's usage is CUMULATIVE across rounds while
    the ``Model`` contract (and every direct-API provider) reports
    per-request numbers. Mixing the two poisons context-size consumers:
    the Agent's proactive compaction gate anchors on
    ``tokens.input_tokens + cache_*`` as "how full is the window" and a
    69-round turn summing to 5.6M against a 200k window trips it
    spuriously (live 2026-06-09). Normalization at this boundary:

    - **input side** (``input_tokens``, ``cache_creation_tokens``,
      ``cache_read_tokens``): the LAST round's request usage — the
      true context footprint, matching direct-API semantics. Zeros
      when no round was observed (consumers fall back to estimates).
    - **output side** (``output_tokens``): cumulative across rounds —
      output genuinely accumulates (every internal round's generation
      was produced and billed).
    - **billing**: unaffected — ``total_cost`` sums
      ``modelUsage.costUSD``, which the CLI computes from the full
      cumulative usage, so under-reporting cumulative input *tokens*
      here loses no cost fidelity.
    """
    model_usage = cast(MutableJSON, usage_event.get("modelUsage") or {})
    output_tokens = 0
    total_cost = 0.0
    for row in model_usage.values():
        if not isinstance(row, dict):
            continue
        row_map = cast(MutableJSON, row)
        output_tokens += int_val(row_map.get("outputTokens"), 0)
        cost = row_map.get("costUSD")
        if isinstance(cost, (int, float)):
            total_cost += float(cost)
    if total_cost == 0.0:
        # Fall back to top-level total_cost_usd when modelUsage is absent.
        raw = usage_event.get("total_cost_usd")
        if isinstance(raw, (int, float)):
            total_cost = float(raw)
    input_tokens = 0
    cache_creation = 0
    cache_read = 0
    if last_round_usage is not None:
        input_tokens = int_val(last_round_usage.get("input_tokens"), 0)
        cache_creation = int_val(
            last_round_usage.get("cache_creation_input_tokens"), 0
        )
        cache_read = int_val(last_round_usage.get("cache_read_input_tokens"), 0)
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
