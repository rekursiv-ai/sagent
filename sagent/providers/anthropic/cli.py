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

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    NotRequired,
    TypedDict,
    cast,
    override,
)

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile

from sagent import types
from sagent.catalog import anthropic as anthropic_catalog
from sagent.catalog.cost import TokenCost
from sagent.lib.custom_json import (
    JSON,
    FloatCodec,
    IntCodec,
    MutableJSON,
    validate_json_schema,
)
from sagent.providers.anthropic.api import Anthropic
from sagent.providers.lib.cli_respawn import respawn_for_cadence
from sagent.providers.lib.errors import (
    error_status_code,
    is_context_overflow_text,
    is_request_too_large,
)
from sagent.providers.lib.hotspare import HotSpare
from sagent.providers.lib.mcp_bridge import ToolsBridge
from sagent.providers.lib.model_base import ModelDefaults
from sagent.providers.lib.oauth import (
    credentials_path,
    resolve_account,
)
from sagent.providers.lib.stop_reason import normalize_stop_reason
from sagent.providers.lib.subproc import (
    _READ_IDLE_TIMEOUT_SEC,
    Subproc,
    SubprocessTransportError,
)
from sagent.types.model import (
    ModelCapability,
    ModelRequest,
    ModelResponse,
    ModelSpec,
    TokenCount,
    base_model_id,
    latency_from_model_id,
)
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ModelResponsePartial,
    ModelResponseThinking,
    RuntimeEvent,
    ToolLabel,
    ToolResult,
    UserMessage,
)
from sagent.types.tape import TapeEvent


if TYPE_CHECKING:
    from sagent.types.tools import Tool

    import sagent.lib.image as image_lib
else:
    from wrapt import lazy_import

    image_lib = lazy_import("sagent.lib.image")


logger = logging.getLogger(__name__)


_CREDS_PATH = Path.home() / ".claude" / ".credentials.json"
_AUTH_STATUS_TIMEOUT_SEC = (
    5.0  # config-globals: ignore -- retunable auth-status probe timeout
)
_NON_SUBSCRIPTION_AUTH_ENV = frozenset(
    {
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_AWS_API_KEY",
        "ANTHROPIC_AWS_BASE_URL",
        "ANTHROPIC_AWS_WORKSPACE_ID",
        "ANTHROPIC_BEDROCK_BASE_URL",
        "ANTHROPIC_BEDROCK_MANTLE_API_KEY",
        "ANTHROPIC_BEDROCK_MANTLE_BASE_URL",
        "ANTHROPIC_CUSTOM_HEADERS",
        "ANTHROPIC_FOUNDRY_API_KEY",
        "ANTHROPIC_FOUNDRY_AUTH_TOKEN",
        "ANTHROPIC_FOUNDRY_BASE_URL",
        "ANTHROPIC_FOUNDRY_RESOURCE",
        "ANTHROPIC_IDENTITY_TOKEN",
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
        "ANTHROPIC_UNIX_SOCKET",
        "ANTHROPIC_VERTEX_BASE_URL",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "AWS_BEARER_TOKEN_BEDROCK",
        "CLAUDE_CODE_API_BASE_URL",
        "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
        "CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL",
        "CLAUDE_CODE_CUSTOM_OAUTH_URL",
        "CLAUDE_CODE_ENABLE_PROXY_AUTH_HELPER",
        "CLAUDE_CODE_HFI_BEARER_TOKEN",
        "CLAUDE_CODE_HOST_AUTH_ENV_VAR",
        "CLAUDE_CODE_OAUTH_CLIENT_ID",
        "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
        "CLAUDE_CODE_OAUTH_SCOPES",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
        "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST",
        "CLAUDE_CODE_PROXY_AUTHENTICATE",
        "CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH",
        "CLAUDE_CODE_SDK_HAS_OAUTH_REFRESH",
        "CLAUDE_CODE_SESSION_ACCESS_TOKEN",
        "CLAUDE_CODE_WEBSOCKET_AUTH_FILE_DESCRIPTOR",
        "CLAUDE_CODE_USE_ANTHROPIC_AWS",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_GATEWAY",
        "CLAUDE_CODE_USE_MANTLE",
        "CLAUDE_CODE_USE_VERTEX",
    },
)
_CREDENTIALS_SCHEMA: Final[JSON] = {
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


def _claude_auth_status(binary: str) -> bool | None:
    """Return whether the native CLI has a Claude.ai subscription login.

    Current Claude Code stores credentials in the macOS Keychain, so the
    historical ``~/.claude/.credentials.json`` file is not an authoritative
    login check there. ``None`` means the installed CLI predates
    ``claude auth status --json`` or did not identify its auth method; callers
    may then fall back to the legacy file check. Console/API/cloud auth returns
    ``False`` because ``AnthropicCLI`` is the subscription provider.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _NON_SUBSCRIPTION_AUTH_ENV
    }
    try:
        proc = subprocess.run(  # noqa: S603 -- binary resolved by shutil.which
            [binary, "auth", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=_AUTH_STATUS_TIMEOUT_SEC,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        decoded: object = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict):
        return None
    status = cast(MutableJSON, decoded)
    logged_in = status.get("loggedIn")
    if logged_in is False:
        return False
    if logged_in is not True:
        return None
    auth_method = status.get("authMethod")
    if not isinstance(auth_method, str):
        # Older CLIs that expose only ``loggedIn`` are not enough to prove
        # subscription billing. Let the caller try the legacy OAuth file.
        return None
    api_provider = status.get("apiProvider")
    return auth_method == "claude.ai" and (
        api_provider is None or api_provider == "firstParty"
    )


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
    the dominant cause is actually our own SIGINT preempt: when the
    runtime cancels the in-flight model-call task (a peer or operator
    message arrives while a ``claude --print`` subprocess is mid-turn),
    ``CancelledError`` reaches ``stream`` and ``_interrupt_active_proc``
    SIGINTs the subprocess. The CLI subprocess dies and emits a final
    ``result`` event
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
        event: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_ms = retry_after_ms
        self.event = event


def _is_event_retryable(event: Mapping[str, object]) -> bool:
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
        for err in cast(list[object], errors):
            if "ede_diagnostic" in str(err):
                return True
    return False


def _extract_retry_after_ms(event: Mapping[str, object]) -> float | None:
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

    Inherits ``CAPABILITIES`` (limits, pricing, tokenizer density) from
    :class:`Anthropic`. The default account uses the CLI's native login
    (including macOS Keychain storage); named accounts use the file variant
    produced by ``providers.lib.oauth.credentials_path``.
    Cost figures are computed from the per-turn ``modelUsage`` summary
    the CLI emits on the terminal ``result`` event.
    """

    TRANSPORT: ClassVar[ModelCapability] = anthropic_catalog.CLI
    """The subprocess exposes no effort, latency, cache, or redaction knob."""

    supported_options: ClassVar[frozenset[str]] = frozenset[str]()
    """``from_credentials`` (the CLI wrapper) takes no construction options.

    Declared for the class's primary (credentials) auth. ``from_key``
    delegates to plain :class:`Anthropic` and technically accepts its
    options, but that degenerate spelling is not worth an auth-scoped
    capability surface -- use ``--provider Anthropic`` for key auth.
    """

    def __init__(self, *, account: str | None = None) -> None:
        super().__init__(api_key="")
        account_name = resolve_account(account)
        self._account = None if account_name == "default" else account_name

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
        """Build a provider that uses the local ``claude`` CLI login.

        Args:
          account: Named credential slot. ``None`` reads the legacy
              native CLI login, including the macOS Keychain. Named accounts
              use Sagent's legacy file-based credential slots.

        Returns:
          provider: Configured CLI-wrapping provider.

        Raises:
          FileNotFoundError: If the native CLI is logged out or a named
              credentials file is absent.
          RuntimeError: If ``claude`` is not on ``PATH``.

        """
        binary = shutil.which("claude")
        if binary is None:
            raise RuntimeError(
                "AnthropicCLI: `claude` is not on PATH; install the Claude CLI.",
            )
        account_name = resolve_account(account)
        is_default_account = account_name == "default"
        if is_default_account:
            logged_in = _claude_auth_status(binary)
            if logged_in is True:
                return cls(account=None)
            if logged_in is False:
                raise FileNotFoundError(
                    "AnthropicCLI: Claude CLI has no active Claude.ai "
                    "subscription login; run `claude auth login --claudeai`.",
                )
        path = credentials_path(_CREDS_PATH, account)
        if not path.exists():
            if not is_default_account:
                raise FileNotFoundError(
                    f"AnthropicCLI: no named-account credentials at {path}; "
                    "named accounts require a legacy Claude credentials file "
                    "at that path.",
                )
            raise FileNotFoundError(
                f"AnthropicCLI: no credentials at {path}; run "
                "`claude auth login --claudeai`.",
            )
        if _load_cli_credentials_file(path) is None:
            raise ValueError(f"Invalid credentials file: {path}")
        return cls(account=account)

    @classmethod
    def login(cls, *, account: str | None = None) -> None:
        """Run Claude Code's native Claude.ai login for the default account.

        Named AnthropicCLI accounts are legacy file-backed slots and cannot be
        targeted by Claude Code's interactive login command.

        Args:
          account: Account slot. ``None`` or ``"default"`` selects the native
              Claude Code account.

        Raises:
          RuntimeError: If the Claude CLI is missing, its login command fails,
              or the resulting Claude.ai login cannot be verified.
          ValueError: If a named account is requested.

        """
        account_name = resolve_account(account)
        if account_name != "default":
            raise ValueError(
                "AnthropicCLI named accounts use legacy credential files and "
                "do not support interactive login.",
            )
        binary = shutil.which("claude")
        if binary is None:
            raise RuntimeError(
                "AnthropicCLI: `claude` is not on PATH; install the Claude CLI.",
            )
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in _NON_SUBSCRIPTION_AUTH_ENV
        }
        proc = subprocess.run(  # noqa: S603 -- binary resolved by shutil.which
            [binary, "auth", "login", "--claudeai"],
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Claude CLI login failed with exit code {proc.returncode}.",
            )
        if _claude_auth_status(binary) is not True:
            raise RuntimeError(
                "Claude CLI login completed but no active Claude.ai "
                "subscription login could be verified.",
            )

    @override
    def model(  # ty: ignore[invalid-method-override]  -- subclasses Anthropic for the shared model catalog + auth, but the CLI transport returns a different Model and accepts provider-specific options; both still satisfy the Provider protocol's ``model(..., **provider_options)`` shape
        self,
        model_id: str | None = None,
        max_request_tokens: int | None = None,
        *,
        extra_mcp_servers: dict[str, dict[str, object]] | None = None,
        session_id: str | None = None,
        subprocess_read_timeout_sec: float | None = None,
        mcp_connect_timeout_sec: float = 8.0,
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
          session_id: When set, the CLI runs in session-persistence
            mode: ``--session-id <uuid>`` on the first turn, ``--resume
            <uuid>`` thereafter, so ``claude`` keeps the transcript on
            disk and the prompt cache stays warm across turns. Sagent's
            tape remains the single source of truth: on the first call
            (including the first call after a host restart, when the
            tape has been rehydrated via ``Agent.resume``) sagent feeds
            the full history via stdin, which rebuilds the on-disk
            session; every subsequent turn feeds only the new entries
            and resumes. ``None`` (default) uses the stateless
            ``--no-session-persistence`` path. The on-disk file is an
            internal cache the provider rebuilds from the request --
            it is never parsed back or treated as a source of truth, so
            no separate session-file format coupling exists.
          subprocess_read_timeout_sec: Stdout-idle timeout (seconds)
            for the ``claude`` subprocess transport. ``None`` keeps
            the ``Subproc`` default (60s). Set higher when the
            agent's tools include long-running synchronous Bash
            commands (e.g. ``pre-commit run --files ...``,
            ``ty check``, big test suites) that legitimately go
            silent for >60s while claude awaits the result.
          mcp_connect_timeout_sec: Cap (seconds) on waiting for the CLI to
            fetch the MCP catalog (proof the in-process bridge connected)
            before feeding the first user line. The CLI's MCP connect +
            catalog fetch is ~2-3s on a cold subprocess locally; this bounds
            a pathologically slow connect before we give up. Only paid once,
            on a cold spawn's first turn -- warm subprocesses keep the
            connection and skip the wait entirely.

        Returns:
          model: Backend wrapping a managed ``claude`` subprocess.

        Raises:
          ValueError: If the resolved id is not in ``CAPABILITIES``, or
              it carries a ``+fast`` tag (the CLI has no fast path).

        """
        mid = model_id if model_id is not None else "default"
        if latency_from_model_id(mid) is not None:
            raise ValueError(
                f"Model {mid!r}: fast mode (+fast) is unsupported via the CLI",
            )
        spec = types.providers.resolve(
            mid, models=self.CAPABILITIES, roles=self.ROLES, transport=self.TRANSPORT
        )
        return _AnthropicCLIModel(
            provider=self,
            # ``mid`` may be a role name; the spec carries the resolved id.
            model_id=spec.tagged_model_id,
            spec=spec,
            max_request_tokens=(
                max_request_tokens
                if max_request_tokens is not None
                else spec.context_limits.max_request_tokens
            ),
            extra_mcp_servers=extra_mcp_servers,
            session_id=session_id,
            subprocess_read_timeout_sec=subprocess_read_timeout_sec,
            mcp_connect_timeout_sec=mcp_connect_timeout_sec,
        )

    @override
    def utility_model(self) -> _AnthropicCLIModel:  # ty: ignore[invalid-method-override]  -- shared catalog, different transport
        """Return the cheapest CLI-backed model (Haiku by default).

        Returns:
          model: Utility model backend.

        """
        return self.model("utility")

    @property
    def account(self) -> str | None:
        """Per-account credentials slot, ``None`` for the legacy file."""
        return self._account


class _AnthropicCLIModel(ModelDefaults):
    """``claude`` CLI subprocess wrapped as a sagent ``Model``.

    Args:
      provider: Owning :class:`AnthropicCLI`.
      model_id: Claude model id passed via ``--model``.
      max_request_tokens: Per-request input cap.

    """

    spec: ModelSpec = ModelSpec()
    """What this model can do; replaced from the catalog at construction."""

    def __init__(
        self,
        *,
        provider: AnthropicCLI,
        model_id: str,
        max_request_tokens: int,
        spec: ModelSpec | None = None,
        extra_mcp_servers: dict[str, dict[str, object]] | None = None,
        session_id: str | None = None,
        subprocess_read_timeout_sec: float | None = None,
        mcp_connect_timeout_sec: float = 8.0,
    ) -> None:
        self._provider = provider
        self._model_id = model_id
        self._max_request_tokens = max_request_tokens
        self.spec = spec or ModelSpec()
        # Cap (seconds) on waiting for the CLI to fetch the MCP catalog
        # before feeding the first user line. See ``AnthropicCLI.model``.
        self._mcp_connect_timeout_sec = mcp_connect_timeout_sec
        self._last_sent_index = 0
        self._system_hash: str = ""
        self._turn_count = 0
        self._last_input_tokens = 0
        self._tools_bridge: ToolsBridge | None = None
        self._warming_proc: Subproc | None = None
        # Stdout-idle timeout (seconds) for the ``claude`` subprocess
        # transport. ``None`` defers to the ``Subproc`` default (60s).
        # Set higher (~3-5min) for agents whose tools include
        # long-running synchronous Bash calls (``pre-commit run`` /
        # ``ty check`` / heavy tests); without the bump a >60s tool wait
        # makes the transport read claude's silence as a hang and
        # raises ``SubprocessTransportError`` on a healthy turn.
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
        # ``_session_initialized = True`` means "this PROCESS has spawned
        # at least one successful turn under this uuid, so the on-disk
        # session is ours to ``--resume``". Always False at construction:
        # implicit session mode never trusts a pre-existing on-disk file
        # it didn't write this process. On the first call -- including
        # the first call after a host restart, when ``Agent.resume`` has
        # rehydrated the tape so ``request.messages`` carries the full
        # history -- sagent re-feeds that history via stdin under
        # ``--session-id``, which REBUILDS the on-disk session from
        # sagent's canonical tape. Every later turn feeds only new
        # entries and ``--resume``s. This makes the on-disk file a pure
        # cache derived from the request, never a source of truth, so
        # there is no session-file format to parse or keep in sync.
        # The one cost is a single cold (full-history) turn after a
        # restart; thereafter the prompt cache is warm.
        self._session_initialized: bool = False
        if session_id is None:
            self._hot_spare: HotSpare | None = HotSpare(
                self._spawn_spare_initialized,
                close_partial=self._close_warming_proc,
            )
            # Stateless named accounts mint an isolated HOME per spawn.
            # The default account inherits real HOME so current macOS
            # Claude Code can read its Keychain-backed login.
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
        # Tools the stateless ``stream`` stashes so ``_spawn_initialized``
        # can populate the bridge before launching ``claude`` (the CLI
        # lists tools right after launch). Empty in session mode, which
        # syncs the bridge before its own spawn.
        self._pending_tools: list[Tool] = []
        # Bridge ``list_tools`` count snapshotted per spawned proc (keyed
        # by ``id(proc)``); the turn's first ``_send_entry`` waits for the
        # count to exceed that proc's snapshot (proof THIS subprocess's MCP
        # client connected). Per-proc, not a single field, so a concurrent
        # HotSpare spare-warm can't clobber the active turn's baseline.
        # Pruned after the wait succeeds, on spawn failure, and in ``close``.
        self._mcp_baseline_by_proc: dict[int, int] = {}
        # Folded text of drained detached tool results, held across a
        # failed-then-retried delivery turn (``drain_detached_results`` is
        # destructive). Cleared only when the delivering turn succeeds, so
        # a transport/retryable failure of that turn's final drain does not
        # lose the results. See ``_detached_delivery_entry``.
        self._pending_detached_text: str | None = None
        self._sent_history_head: TapeEvent | None = None
        # External MCP servers (stdio or HTTP) merged into the CLI's
        # ``--mcp-config`` at subprocess spawn time. See
        # :func:`_build_anthropic_argv`.
        self._extra_mcp_servers = extra_mcp_servers
        # Session mode: drop any session file a prior process left on
        # disk. We don't trust it -- the first turn rebuilds the session
        # from the (possibly rehydrated) tape via ``--session-id``. A
        # leftover file would make that spawn error "Session ID already
        # in use".
        if session_id is not None:
            self._delete_session_jsonl(reason="construction (rebuild from tape)")

    @property
    def max_request_tokens(self) -> int:
        """Per-request input token cap."""
        return self._max_request_tokens

    @property
    def model_id(self) -> str:
        """Model identifier passed to ``claude --model``."""
        return self._model_id

    def _interrupt_active_proc(self) -> bool:
        """SIGINT the active CLI subprocess to abort the current turn.

        Called from ``stream``'s ``CancelledError`` handler: when the
        runtime cancels the in-flight model-call task (peer/operator
        preempt), ``CancelledError`` propagates into the awaiting
        ``stream``; we translate it into a subprocess SIGINT so the
        opaque CLI tool loop actually stops, then re-raise so the
        runtime's standard cancellation path (``_stream_and_post`` ->
        ``ModelResponseCancelled``) runs. The full CLI tool loop runs
        in-process inside the subprocess, so the runtime cannot
        ``Detach``/``Kill`` individual tool calls; SIGINT is the only
        mid-turn cancellation surface.

        Returns True if a live subprocess was signalled, False otherwise
        (idle, never started, already closed). Non-blocking. Partial
        assistant text + usage telemetry for the cancelled turn are lost
        -- intentional, the caller is preempting precisely because that
        work is no longer wanted.
        """
        if self._hot_spare is not None:
            active = self._hot_spare.active
            if active is None:
                return False
            return active.interrupt()
        # Session-persistence mode: HotSpare is bypassed; the active
        # subprocess (if any) is on ``_active_proc``.
        if self._active_proc is None:
            return False
        return self._active_proc.interrupt()

    @property
    def max_response_tokens(self) -> int:
        """Per-request output token cap from the profile."""
        return self.spec.context_limits.max_response_tokens

    @property
    def supports_streaming(self) -> bool:
        """``True``: the wrapped CLI always streams."""
        return True

    @property
    def supports_thinking(self) -> bool:
        """Whether the active spec supports extended thinking."""
        return bool(self.spec.supported_thinking_budgets)

    @property
    def supports_effort(self) -> bool:
        """``False``: the CLI does not expose the effort knob on stream-json."""
        return bool(self.spec.supported_thinking_efforts)

    @property
    def valid_efforts(self) -> tuple[str, ...]:
        """No effort knob on the CLI transport."""
        return tuple(self.spec.supported_thinking_efforts.values())

    @property
    def supports_cache_control(self) -> bool:
        """``False``: prompt cache is the CLI's concern, not ours."""
        return self.spec.prompt_cache_breakpoints

    @property
    def supports_context_management(self) -> bool:
        """``True``: the CLI itself rolls history under quota pressure."""
        return True

    @property
    def supports_persistent_retry(self) -> bool:
        """``False``: persistent retry conflicts with subprocess lifecycle."""
        return self.spec.retries_internally

    @property
    def supports_account_auth(self) -> bool:
        """``True``: the provider runs on the user's CLI subscription."""
        return True

    @property
    def max_image_dim(self) -> int:
        """Maximum image edge (pixels) accepted, from the model profile."""
        return self.spec.context_limits.max_image_edge_px

    @property
    def max_image_bytes(self) -> int:
        """Maximum size (bytes) of a single image, from the model profile."""
        return self.spec.context_limits.max_image_bytes

    @property
    def max_request_bytes(self) -> int:
        """Maximum request-body size (bytes), from the model profile."""
        return self.spec.context_limits.max_request_bytes

    @override
    def approx_text_tokens(self, text: str) -> int:
        """Local estimate via ``chars_per_token``."""
        return int(len(text) / self.spec.chars_per_token)

    @override
    def approx_image_tokens(self, data: bytes) -> int:
        """Local estimate from image dimensions (``width*height/750``)."""
        dims = image_lib.get_dimensions(data)
        return dims[0] * dims[1] // 750 if dims is not None else 0

    def is_context_overflow(self, error: Exception) -> bool:
        """Classify whether an error means the prompt exceeded the token window.

        Excludes the request-byte wire-limit via the shared classifier first,
        uniform with the HTTP providers and ``GoogleCLI``: a byte-limit error
        routes to byte-overflow recovery, not the ``/model`` larger-window
        remediation the byte ceiling ignores.

        Args:
          error: Exception raised by the call path.

        Returns:
          overflow: ``True`` for known token-overflow markers in the message.

        """
        if is_request_too_large(error_status_code(error), str(error)):
            return False
        return is_context_overflow_text(str(error))

    @override
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

    @override
    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        """Drive one user turn through the ``claude`` subprocess.

        Sends each new history entry (skipping assistant turns the CLI
        already emitted itself), reads stream events until the terminal
        ``result``, and assembles a :class:`ModelResponse`.

        Args:
          request: Conversation + tools + system prompt for the turn.
          publish: Runtime event sink. Text chunks are emitted as
              ``ModelResponsePartial``, thinking as
              ``ModelResponseThinking``; tool calls routed through the
              in-process MCP bridge surface as ``ToolLabel``. ``None``
              disables streaming output.

        Returns:
          response: Parsed model response with usage and cost filled in.

        Raises:
          RuntimeError: The subprocess exited before emitting a terminal
              ``result`` event, or the CLI surfaced an error result.

        """
        self._pending_system = request.system or ""
        # Clear the bridge sink when the turn ends (any exit) so a
        # stale runtime publisher never lingers on the long-lived bridge
        # -- e.g. a detached background ``call_tool`` firing between
        # turns must not label against a finished turn's publisher.
        try:
            if self._session_id is not None:
                return await self._stream_session_persistent(request, publish)
            assert self._hot_spare is not None  # stateless path
            # Stash the request's tools so the spawn factory
            # (``_spawn_initialized``) can populate the bridge BEFORE it
            # launches ``claude`` -- the CLI issues ``ListToolsRequest``
            # right after launch, and an empty registry at that moment
            # makes the model emit "no tools have been provided" instead
            # of a ``tool_use`` (live 2026-06-16
            # ``test_bridge_tool_round_trips``). The post-acquire
            # ``_sync_tools_bridge`` below still refreshes the live
            # registry for the already-warm subprocess.
            self._pending_tools = list(request.tools or [])
            if self._should_respawn(request):
                if _hash_system(request.system) != self._system_hash:
                    await self._hot_spare.discard_spare()
                await self._hot_spare.respawn()
                self._reset_active_state()
            proc = await self._hot_spare.acquire()
            self._sync_tools_bridge(request, publish)

            try:
                response = await self._exchange_turn(proc, request, publish)
                self._system_hash = _hash_system(self._pending_system)
                self._last_sent_index = len(request.messages)
                # Detached results (if any) were delivered + answered; drop
                # the held buffer so they aren't redelivered next turn.
                self._pending_detached_text = None
            except asyncio.CancelledError:
                # Runtime cancelled the model-call task (preempt). Translate
                # into a subprocess SIGINT so the opaque CLI tool loop
                # stops, then re-raise so the runtime's cancellation path
                # runs.
                self._interrupt_active_proc()
                self._reset_active_state()
                await self._hot_spare.respawn_after_transport_failure()
                raise
            except SubprocessTransportError:
                self._reset_active_state()
                await self._hot_spare.respawn_after_transport_failure()
                raise
            self._hot_spare.record_success()
            self._turn_count += 1
            return response
        finally:
            if self._tools_bridge is not None:
                self._tools_bridge.set_publish(None)

    async def _stream_session_persistent(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None,
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
        #
        # ``_last_sent_index``-advancing entries from the tape come first;
        # a synthetic detached-tool-result delivery (``rel_idx is None``)
        # is appended LAST and never advances the cumulative counter (it
        # is not part of ``request.messages``). The detached delivery is
        # drained only from an already-running bridge -- if no bridge
        # exists yet, no detached run can have completed.
        new_entries_idx: list[tuple[int | None, TapeEvent]] = []
        for i, entry in enumerate(new_entries):
            if not isinstance(entry, (AssistantMessage, ToolResult)):
                new_entries_idx.append((i, entry))
        detached = self._detached_delivery_entry()
        if detached is not None:
            new_entries_idx.append((None, detached))

        if not new_entries_idx:
            # No new input to feed and no detached results to deliver.
            # Return a no-op response: empty assistant message + zero
            # usage. The runtime treats this as a finished turn with no
            # output; the next real inbound will spawn the next
            # subprocess. (Don't start the bridge for a no-op turn.)
            return ModelResponse(
                message=AssistantMessage(text="", tool_calls=()),
                stop_reason="model_finished",
            )
        base = self._last_sent_index

        # Bridge MUST be populated before the subprocess spawns: the
        # CLI issues ``ListToolsRequest`` against the bridge soon
        # after launch, and if our tool catalog isn't there at that
        # moment, opus falls back to emitting tool calls as plain
        # text inside the assistant message (Episode 2.7 pathology
        # -- observed 2026-06-02 23:16 when TL produced
        # "Bash {command: ls -la …}" as text instead of a tool_use
        # block on the first turn after refactor).
        await self._ensure_tools_bridge()
        self._sync_tools_bridge(request, publish)
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
                assert rel_idx is not None  # only the trailing entry may be synthetic
                self._last_sent_index = base + rel_idx + 1
                _ = await self._drain_until_result(
                    proc,
                    publish=None,
                    update_input_tokens=False,
                )
            last_rel_idx, last_entry = new_entries_idx[-1]
            await self._send_entry(proc, last_entry)
            # ``rel_idx is None`` for a synthetic detached-result delivery:
            # it is not part of ``request.messages``, so it must not
            # advance the cumulative sent counter.
            if last_rel_idx is not None:
                self._last_sent_index = base + last_rel_idx + 1
            response = await self._drain_until_result(proc, publish)
        except asyncio.CancelledError:
            # Runtime cancelled the model-call task (preempt). SIGINT the
            # subprocess so the opaque CLI tool loop stops, then re-raise
            # so the runtime's cancellation path runs. ``_last_sent_index``
            # already reflects every entry written to stdin, so the next
            # ``--resume`` skips them.
            self._interrupt_active_proc()
            await proc.close()
            self._active_proc = None
            raise
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
            # Detached results were delivered + answered this turn; drop
            # the held buffer so they aren't redelivered next turn.
            self._pending_detached_text = None
            await proc.close()
            self._active_proc = None
            self._turn_count += 1
            return response

    @override
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
        self._mcp_baseline_by_proc.clear()

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
        return respawn_for_cadence(
            turn_count=self._turn_count,
            last_input_tokens=self._last_input_tokens,
            max_request_tokens=self._max_request_tokens,
        )

    def _sync_tools_bridge(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> None:
        """Refresh the MCP bridge's tool registry and runtime event sink.

        ``publish`` is the runtime sink for this turn; the bridge emits a
        ``ToolLabel`` through it for each tool call routed through the
        subprocess. Passed afresh every turn so a stale runtime publisher
        never lingers on the long-lived bridge.
        """
        if self._tools_bridge is not None:
            self._tools_bridge.update_tools(list(request.tools or []))
            self._tools_bridge.set_publish(publish)

    def _detached_delivery_entry(self) -> UserMessage | None:
        """Drain the bridge's completed detached tool runs into one entry.

        The bridge runs background tool calls (those the model requested
        with ``background``/``delay``) as tasks in this Model's loop and
        hands back finished results here. We fold them into a single
        user-side message so the next ``claude --print`` turn delivers
        the results the model was promised. ``None`` when nothing is
        pending. This is the Model side of the bridge's internal cohort:
        detached tool results come back as ordinary turn input.

        ``drain_detached_results`` is a destructive read, so the folded
        text is held in ``_pending_detached_text`` until the turn that
        delivers it completes successfully (cleared in the ``else`` branch
        of the stream paths). If that turn's final drain fails and
        ``send_with_retry`` re-invokes ``stream``, this returns the SAME
        buffered entry rather than ``None`` -- otherwise the results, no
        longer in ``_bg_done``, would vanish (the model was promised them).
        """
        if self._pending_detached_text is None:
            if self._tools_bridge is None:
                return None
            results = self._tools_bridge.drain_detached_results()
            if not results:
                return None
            self._pending_detached_text = "\n\n".join(
                f"[detached tool result{f' {r.call_id}' if r.call_id else ''}]"
                f" {r.content}" + (" (error)" if r.is_error else "")
                for r in results
            )
        return UserMessage(text=self._pending_detached_text)

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
        publish: Callable[[RuntimeEvent], None] | None,
    ) -> ModelResponse:
        """Replay prior entries quietly, then return the current turn result."""
        new_entries = request.messages[self._last_sent_index :]
        if self._last_sent_index == 0 and request.messages:
            self._sent_history_head = request.messages[0]
        user_like_entries: list[TapeEvent] = [
            entry for entry in new_entries if not isinstance(entry, AssistantMessage)
        ]
        # Deliver any completed detached (background) tool results as a
        # trailing synthetic user entry, same as the session path -- the
        # bridge advertises ``background``/``delay`` in BOTH modes, so a
        # stateless turn can stage a detached run whose result must be fed
        # back here. Without this the model is promised a later delivery
        # that never arrives and ``_bg_done`` leaks unboundedly.
        detached = self._detached_delivery_entry()
        if detached is not None:
            user_like_entries.append(detached)
        for entry in user_like_entries[:-1]:
            await self._send_entry(proc, entry)
            _ = await self._drain_until_result(
                proc, publish=None, update_input_tokens=False
            )
        if not user_like_entries:
            return await self._drain_until_result(proc, publish)
        await self._send_entry(proc, user_like_entries[-1])
        return await self._drain_until_result(proc, publish)

    async def _send_entry(self, proc: Subproc, entry: TapeEvent) -> None:
        """Write one history entry to stdin.

        Gated on ``proc`` having fetched the bridge catalog (instant after
        the first fetch): a cold subprocess's first user line must not
        race ahead of claude's still-connecting MCP client, or the model
        sees no tools and answers "no tools have been provided".
        """
        await self._await_mcp_listed(proc)
        line = json.dumps(
            _serialize_for_stdin(entry, self.max_image_dim, self.max_image_bytes)
        )
        await proc.write_line(line)

    async def _drain_until_result(
        self,
        proc: Subproc,
        publish: Callable[[RuntimeEvent], None] | None,
        *,
        update_input_tokens: bool = True,
    ) -> ModelResponse:
        """Read stream events until ``result``; assemble a ``ModelResponse``."""
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        # Aggregated thinking-block signature. Anthropic's stream
        # emits a ``signature_delta`` event alongside ``thinking_delta``;
        # the final signature is the concatenation of those deltas
        # (typically a single delta). Required because the assistant
        # message is re-sent on the wire when a session-mode turn
        # rebuilds history, and Anthropic's API rejects a signature-less
        # thinking block (HTTP 400 ``thinking.signature: Field required``).
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
                    # ``aborted_streaming`` event the CLI emits when the
                    # runtime cancels the model-call task and we SIGINT
                    # the mid-turn subprocess (``_interrupt_active_proc``)
                    # for an incoming peer/operator message; the retry
                    # then re-delivers via ``--resume``. See
                    # :func:`_is_event_retryable`
                    # for the catalog.
                    if _is_event_retryable(cast(Mapping[str, object], event)):
                        raise AnthropicCLIRetryableError(
                            f"AnthropicCLI: retryable result is_error: {event}",
                            retry_after_ms=_extract_retry_after_ms(
                                cast(Mapping[str, object], event),
                            ),
                            event=cast(Mapping[str, object], event),
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
                    signature_parts=signature_parts,
                    tool_use_blocks=tool_use_blocks,
                    publish=publish,
                )
            elif kind == "system" and event.get("subtype") == "init":
                message_id = cast(str, event.get("session_id") or "")
        assert usage_event is not None
        if update_input_tokens and last_round_usage is not None:
            # Cache-inclusive context footprint of the last internal round;
            # feeds the context-fraction respawn heuristic. Only overwrite when
            # a round was actually observed: a zero-round drain (no
            # ``message_start``) carries NO footprint signal, and clobbering the
            # last known value with 0 would read as "context is empty" and
            # suppress a respawn that a genuinely full context still needs.
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
        # Populate the bridge registry BEFORE launching ``claude``: the
        # CLI issues ``ListToolsRequest`` right after launch, so an empty
        # registry at this moment makes the model see no tools. The
        # stateless ``stream`` stashes the turn's tools in
        # ``_pending_tools`` for exactly this; session mode hoists its
        # own ``_sync_tools_bridge`` before the spawn, leaving
        # ``_pending_tools`` empty (a no-op refresh).
        if self._pending_tools:
            self._tools_bridge.update_tools(list(self._pending_tools))
        spawn_owned_tmpdir: Path | None
        tmpdir: Path | None
        if self._persistent_tmpdir is not None:
            # Session-persistence + per-account: reuse the
            # construction-time tmpdir so the renamed credentials
            # file is found by claude (its creds path is hardcoded
            # to $HOME/.claude/.credentials.json).
            tmpdir = self._persistent_tmpdir
            spawn_owned_tmpdir = None
        elif self._provider.account is None:
            # Default account, stateless or persistent: NO HOME override.
            # On macOS the Claude CLI stores its subscription login in the
            # Keychain rather than ``~/.claude/.credentials.json``. Inheriting
            # HOME lets the CLI use that native login and also keeps native
            # tools (``gh``, ``git``, ssh) on the operator's normal config.
            tmpdir = None
            spawn_owned_tmpdir = None
        else:
            # Named-account stateless mode: hermetic per-spawn HOME holding
            # the selected legacy file credential. Subproc deletes it on close.
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
        # Snapshot the bridge's ``list_tools`` count BEFORE launch, keyed
        # to THIS proc, so the turn can later confirm THIS subprocess's
        # MCP client connected (its fetch bumps the count past the
        # snapshot). Keyed per-proc (not a single shared field) so a
        # concurrent HotSpare spare-warm -- which runs this same factory
        # and would otherwise clobber a shared baseline -- can't make the
        # active turn wait for the SPARE's connect instead of its own. A
        # warm spare connects during background warm-up, so by the time it
        # is promoted its count already exceeds its snapshot and
        # ``_await_mcp_listed`` returns instantly -- only a genuinely cold
        # first spawn pays the ~3s connect.
        # Only record a baseline when the bridge actually advertises tools
        # -- ``_await_mcp_listed`` is a no-op otherwise and would never prune
        # the entry, so a long-lived tool-less session would accrete them.
        if self._tools_bridge.has_tools:
            self._mcp_baseline_by_proc[id(proc)] = self._tools_bridge.listed_snapshot()
        self._warming_proc = proc
        try:
            await proc.start()
        except BaseException:
            self._mcp_baseline_by_proc.pop(id(proc), None)
            await proc.close()
            self._warming_proc = None
            raise
        self._warming_proc = None
        return proc

    async def _await_mcp_listed(self, proc: Subproc) -> None:
        """Wait for ``proc`` to fetch the bridge's tool catalog.

        The CLI connects to MCP servers asynchronously after launch and
        does NOT block its first turn on that handshake; a cold
        subprocess that generates before the ``sagent`` server flips from
        ``pending`` to ``connected`` sees an empty catalog and answers "no
        tools have been provided" (live 2026-06-16). Called before the
        first user line is written, so the connect overlaps no extra
        latency; instant after the first fetch (the per-proc baseline is
        already exceeded).

        No-op when the bridge has no tools. When tools ARE expected but
        the catalog is not fetched within ``_mcp_connect_timeout_sec``,
        raise :class:`SubprocessTransportError` rather than silently
        feeding the turn a tool-less context: a degraded "no tools have
        been provided" answer is worse than a respawn. The standard
        transport-failure path (HotSpare respawn in stateless mode,
        in-place ``--resume`` retry in session mode) then re-attempts the
        connect on a fresh subprocess.
        """
        if self._tools_bridge is None or not self._tools_bridge.has_tools:
            return
        baseline = self._mcp_baseline_by_proc.get(id(proc), 0)
        listed = await self._tools_bridge.wait_listed(
            baseline, self._mcp_connect_timeout_sec
        )
        if not listed:
            raise SubprocessTransportError(
                "AnthropicCLI: MCP bridge catalog not fetched within "
                f"{self._mcp_connect_timeout_sec:.1f}s (CLI MCP client never "
                "connected); respawning rather than running a tool-less turn",
            )
        # Consumed: this proc has connected. Drop the baseline so the map
        # stays bounded; any later ``_send_entry`` for the same proc reads
        # the ``0`` default and returns instantly (the count already moved
        # past it). New spawns re-snapshot under their own ``id(proc)``.
        self._mcp_baseline_by_proc.pop(id(proc), None)

    def _reset_active_state(self) -> None:
        """Reset active subprocess counters after a respawn boundary."""
        self._turn_count = 0
        self._last_input_tokens = 0
        # A buffered detached result is promised to the OUTGOING context; a
        # respawn starts fresh, so drop it rather than inject a phantom
        # ``[detached tool result]`` referencing calls the new process never saw.
        self._pending_detached_text = None
        self._reset_delta_state()

    def _reset_for_clear(self) -> None:
        """Reset session-persistent state after ``agent.clear()``.

        Wipes the cumulative-sent counter, marks the session as
        uninitialised (next spawn will use ``--session-id`` again),
        DELETES the on-disk session JSONL so the next
        ``--session-id <same-uuid>`` call doesn't error with
        "Session ID is already in use", and drops any buffered detached
        result -- the cleared context no longer knows the tool calls it
        answered, so re-injecting it would be a phantom result.

        Called from :meth:`_stream_session_persistent` when it
        detects ``_last_sent_index > len(request.messages)``, which
        is the post-``Clear`` shape.
        """
        # The detached buffer is context state, cleared regardless of session
        # mode (the early-return below only gates the session-file teardown).
        self._pending_detached_text = None
        if self._session_id is None:
            return
        self._last_sent_index = 0
        self._session_initialized = False
        self._delete_session_jsonl(reason="agent.clear()")

    def _delete_session_jsonl(self, *, reason: str) -> None:
        """Delete the on-disk session JSONL for this uuid, if present.

        The on-disk file is a pure cache the provider rebuilds from the
        request; a stale copy (left by a prior process, or invalidated
        by ``agent.clear()``) must be removed so the next
        ``--session-id <same-uuid>`` spawn doesn't error with "Session
        ID is already in use". Called at construction (drop any prior
        process's file -- we rebuild from the rehydrated tape) and from
        ``_reset_for_clear``.
        """
        if self._session_id is None:
            return
        path = self._session_jsonl_path()
        if path is None or not path.exists():
            return
        try:
            path.unlink()
            logger.info(
                "AnthropicCLI(session): cleared session JSONL at %s (%s)",
                path,
                reason,
            )
        except OSError as exc:
            logger.warning(
                "AnthropicCLI(session): failed to delete %s: %s",
                path,
                exc,
            )

    def _session_jsonl_path(self) -> Path | None:
        """On-disk session-file path for this uuid under the spawn cwd.

        ``None`` in stateless mode. Resolves HOME the way the spawn does
        (per-account tmpdir, else the operator's real HOME honoring
        ``CLAUDE_CONFIG_DIR``) and the cwd the way claude encodes it
        (canonicalized, non-alnum -> ``-``), so the path the provider
        computes is the path claude reads/writes.
        """
        if self._session_id is None:
            return None
        try:
            cwd = Path.cwd()
        except OSError:
            return None
        return _session_jsonl_path(self._session_id, cwd=cwd, home=self._claude_home())

    def _claude_home(self) -> Path:
        """Resolve the HOME the ``claude`` subprocess uses.

        Per-account mode pins a hermetic tmpdir; single-account mode
        inherits the operator's real HOME.
        """
        return self._persistent_tmpdir or _real_home()

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
        access_token=str(oauth["accessToken"]),
        refresh_token=str(oauth["refreshToken"]),
        expires_at=FloatCodec.coerce(oauth["expiresAt"]) / 1000.0,
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
            creds["account_uuid"] = str(token_account["uuid"])
        if token_account.get("emailAddress"):
            creds["email"] = str(token_account["emailAddress"])
        if token_account.get("organizationUuid"):
            creds["organization_uuid"] = str(token_account["organizationUuid"])
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


def _real_home() -> Path:
    """The HOME claude uses when sagent doesn't override it.

    Honors ``CLAUDE_CONFIG_DIR`` the way the CLI does: when set, claude
    stores ``projects/`` under it rather than ``$HOME/.claude``. We
    return a path such that ``<return>/.claude/projects`` equals claude's
    projects root in both cases.
    """
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        # claude treats CLAUDE_CONFIG_DIR as the ``.claude`` dir itself;
        # return its parent so the shared ``/.claude/projects`` suffix
        # in ``_session_jsonl_path`` resolves correctly.
        return Path(config_dir).expanduser().parent
    return Path(os.environ.get("HOME", "~")).expanduser()


def _session_jsonl_path(session_id: str, *, cwd: Path, home: Path) -> Path:
    """On-disk session-file path claude uses for ``(session_id, cwd)``.

    Mirrors the CLI's encoding so the path sagent computes is the path
    claude reads/writes: the cwd is canonicalized (symlinks resolved --
    e.g. macOS ``/tmp`` -> ``/private/tmp``) and every non-``[A-Za-z0-9-]``
    character becomes ``-``. Must stay cwd-aware: claude indexes sessions
    per encoded-cwd project dir and ``--resume`` cannot see a session
    recorded under a different cwd.
    """
    try:
        resolved = cwd.resolve()
    except OSError:
        resolved = cwd
    encoded = re.sub(r"[^A-Za-z0-9-]", "-", str(resolved))
    return home / ".claude" / "projects" / encoded / f"{session_id}.jsonl"


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
    """Build the ``claude`` subprocess env with native or isolated auth.

    When ``persist_session=True`` we keep ``CLAUDE_CODE_SKIP_PROMPT_HISTORY``
    unset -- that env var (verified by bisect 2026-06-02) causes the CLI
    to skip writing its session JSONL even when ``--session-id`` /
    ``--resume`` are passed, which makes session-persistence mode silently
    no-op and the next ``--resume`` fails with "No conversation found".

    When ``tmpdir is None`` we don't override ``HOME`` -- the subprocess
    inherits the operator's real HOME so Claude finds macOS Keychain auth,
    project session JSONLs, and native-tool config. This is the default-account
    path in both stateless and persistent modes. Named accounts use ``tmpdir``
    because the CLI has no credential-file override.

    Auto-compact is always disabled: sagent owns history in both modes
    (stateless re-feeds it each turn; session mode rebuilds the on-disk
    file from the tape on the first turn and feeds deltas thereafter).
    Claude's auto-compact would write a ``system/compact_boundary`` to a
    file sagent overwrites next turn, so sagent's own ``SummaryCompactor``
    is the sole compaction authority.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _NON_SUBSCRIPTION_AUTH_ENV
    }
    if tmpdir is not None:
        env["HOME"] = str(tmpdir)
        env["USERPROFILE"] = str(tmpdir)
        env["CLAUDE_CONFIG_DIR"] = str(tmpdir / ".claude")
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
            "DISABLE_AUTO_COMPACT": "1",
        }
    )
    if not persist_session:
        env["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] = "1"
    return env


def _build_anthropic_argv(
    *,
    model_id: str,
    system_prompt: str,
    bridge_url: str,
    bridge_server_name: str,
    extra_mcp_servers: dict[str, dict[str, object]] | None = None,
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
    servers: dict[str, dict[str, object]] = {
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
    # Never pass ``--tools ""``: the CLI reads empty-string as "ALLOW NO
    # TOOLS, including MCP ones" -- bisect probe 2026-06-03 found:
    #
    #   --tools ""              → no tool_use; opus replies "I'll run
    #                             `ls /tmp`" as plain text.
    #   (omit --tools entirely) → tool_use(name=Bash) emitted cleanly.
    #   --tools "Bash"          → tool_use(name=Bash) emitted cleanly.
    #
    # This was originally believed to bite only session-persistence mode,
    # but the stateless path routes tools through the same MCP bridge, so
    # ``--tools ""`` silently disabled bridge tools there too -- the model
    # answered "no tools have been provided" for a bridge-mounted tool
    # (live 2026-06-16 ``test_bridge_tool_round_trips``). Omit ``--tools``
    # in BOTH modes so the bridge's MCP catalog is always honored.
    base.extend(
        [
            "--disable-slash-commands",
            "--permission-mode",
            "bypassPermissions",
        ]
    )
    # Defensively deny the Claude-Teams / Agent-SDK ``SendMessage``
    # built-in. Per the CLI source it is gated behind
    # ``isAgentSwarmsEnabled()`` (``USER_TYPE=ant`` / ``--agent-teams`` /
    # the experimental env flag) and is therefore NOT in the default
    # allowlist for ordinary external builds -- so this is usually a
    # no-op. We keep it because if the flag ever IS on, ``SendMessage``
    # routes to a private Teams registry that is EMPTY in the sagent
    # context, silently dropping the message (observed losing a "PR is
    # open" notification when an ant-flavored CLI was in use). The only
    # working peer channel is an external MCP tool such as
    # ``mcp__sagent_chat__sagent_send``.
    base.extend(["--disallowedTools", "SendMessage"])
    return base


def _serialize_for_stdin(
    entry: TapeEvent, max_image_dim: int, max_image_bytes: int
) -> MutableJSON:
    """Translate a non-assistant ``TapeEvent`` into the CLI's user-line shape."""
    if isinstance(entry, (AgentSendMessage, UserMessage)):
        return _user_line(entry, max_image_dim, max_image_bytes)
    assert isinstance(entry, ToolResult)
    # Tool results never traverse stdin: the CLI's MCP client handled
    # the tool_use round-trip internally. Surface mistakes loudly.
    raise RuntimeError(
        "AnthropicCLI: ToolResult in history -- tools must go through the MCP bridge",
    )


def _user_line(
    entry: AgentSendMessage | UserMessage,
    max_image_dim: int,
    max_image_bytes: int,
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
            att.data, max_dim=max_image_dim, max_bytes=max_image_bytes
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
    *,
    signature_parts: list[str],
    tool_use_blocks: dict[int, dict[str, object]],
    publish: Callable[[RuntimeEvent], None] | None,
) -> None:
    """Route one stream_event payload to the runtime ``publish`` sink.

    Text deltas publish ``ModelResponsePartial``; thinking deltas
    publish ``ModelResponseThinking``. Also accumulates ``tool_use``
    content blocks across their start / streamed ``input_json_delta``
    chunks / stop events, and emits one ``ToolLabel`` per tool call at
    block-stop with ``name`` plus a short rendering of the JSON args
    (e.g. ``Bash ls -la`` or ``Read foo.py``) so the trace panel
    surfaces what tools the model is invoking. Covers both
    bridge-mounted tools (Bash, Read, ...) and external-MCP tools
    (``mcp__sagent_chat__sagent_send``, ...) uniformly -- in stateless
    mode the bridge ALSO publishes labels for its own tools, so
    bridge-mounted tools get logged twice; the trace renderer treats
    each ToolLabel as a separate event.
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
                cast(list[str], state["json_parts"]).append(partial)
            return
        if delta_type == "text_delta":
            text = cast(str, delta.get("text") or "")
            if text:
                text_parts.append(text)
                if publish is not None:
                    publish(ModelResponsePartial(text))
            return
        if delta_type == "thinking_delta":
            text = cast(str, delta.get("thinking") or "")
            if text:
                thinking_parts.append(text)
                if publish is not None:
                    publish(ModelResponseThinking(text))
            return
        if delta_type == "signature_delta":
            # Per Anthropic's stream-json spec, ``signature_delta``
            # carries the opaque thought-signature in the ``signature``
            # field (mirrors ``thinking_delta`` for body text). The
            # final signature is the concatenation across deltas
            # (typically a single delta in practice). Required so a
            # downstream wire re-send (session-mode history rebuild)
            # embeds the signature in the thinking block -- Anthropic's
            # API rejects unsigned thinking with HTTP 400
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
        tool_name = str(state["name"])
        tool_id = str(state["id"])
        json_parts = cast(list[str], state["json_parts"])
        args_summary = _render_tool_args("".join(json_parts))
        label_text = f"{tool_name} {args_summary}".rstrip()
        if publish is not None:
            try:
                publish(ToolLabel(call_id=tool_id, text=label_text))
            except Exception:
                logger.debug(
                    "failed to publish ToolLabel for %r", tool_name, exc_info=True
                )
        return


def _render_tool_args(raw_json: str) -> str:
    """Render tool input JSON as a short label suffix.

    Best-effort: if the JSON is incomplete (streaming aborted mid-flight)
    or unparseable, falls back to the raw form. Common args
    (``command``, ``file_path``, ``pattern``, ``query``, ``to``,
    ``content``) get a friendly rendering; unknown tools fall back to
    the raw arg dict.

    Known-arg branches return the value whole -- the renderer
    (``console_pane.write_tool_label``) owns wrapping and the line cap.
    The unknown-tool fallback keeps its per-value clamp: an arbitrary
    arg dict can carry a multi-megabyte blob (the CLI's own stdout line
    limit is 16 MiB for exactly this reason), and unlike the named args
    it is not a value the operator asked to see.
    """
    if not raw_json:
        return ""
    try:
        args = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError):
        return f"({raw_json.replace(chr(10), ' ')})"
    if not isinstance(args, dict):
        return str(args)
    arg_map = cast(dict[object, object], args)
    for key in ("command", "file_path", "path", "pattern", "query", "to"):
        val = arg_map.get(key)
        if isinstance(val, str):
            return val
    content = arg_map.get("content")
    if isinstance(content, str):
        return content
    return ", ".join(f"{k}={str(v)[:40]!r}" for k, v in list(arg_map.items())[:3])


def _round_context_tokens(round_usage: MutableJSON | None) -> int:
    """Cache-inclusive input footprint of one internal round's request.

    ``round_usage`` is the raw Anthropic API ``usage`` object off a
    ``message_start`` stream event (snake_case keys -- unlike the
    camelCase rows in the CLI's terminal ``result.modelUsage``). The
    sum of non-cached input plus both cache pools is the full prompt
    size the server counted for that request -- the same number the
    direct-API provider's per-request usage reports. Returns 0 when no
    round was observed (defensive; a successful drain always sees at
    least one ``message_start``), which downstream consumers treat as
    "unknown -- estimate instead".
    """
    if round_usage is None:
        return 0
    return (
        IntCodec.coerce(round_usage.get("input_tokens"), 0)
        + IntCodec.coerce(round_usage.get("cache_creation_input_tokens"), 0)
        + IntCodec.coerce(round_usage.get("cache_read_input_tokens"), 0)
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
    ``tokens.request + cache_*`` as "how full is the window" and a
    69-round turn summing to 5.6M against a 200k window trips it
    spuriously (live 2026-06-09). Normalization at this boundary:

    - **input side** (``input_tokens``, ``cache_creation_tokens``,
      ``cache_read_tokens``): the LAST round's request usage -- the
      true context footprint, matching direct-API semantics. Zeros
      when no round was observed (consumers fall back to estimates).
    - **output side** (``output_tokens``): cumulative across rounds --
      output genuinely accumulates (every internal round's generation
      was produced and billed).
    - **billing**: unaffected -- ``total_cost`` sums
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
        output_tokens += IntCodec.coerce(row_map.get("outputTokens"), 0)
        cost = row_map.get("costUSD")
        if isinstance(cost, (int, float)):
            total_cost += float(cost)
    if total_cost == 0.0:
        # Fall back to top-level total_cost_usd when modelUsage is absent.
        raw = usage_event.get("total_cost_usd")
        if isinstance(raw, (int, float)):
            total_cost = float(raw)
        elif not model_usage:
            # Neither cost source present: a genuinely-free turn and a dropped
            # usage summary both read as 0.0 here. Log so the latter is not
            # silently indistinguishable from the former.
            logger.debug(
                "no cost signal in result event (modelUsage and total_cost_usd"
                " both absent); reporting total_cost=0.0"
            )
    input_tokens = 0
    cache_creation = 0
    cache_read = 0
    if last_round_usage is not None:
        input_tokens = IntCodec.coerce(last_round_usage.get("input_tokens"), 0)
        cache_creation = IntCodec.coerce(
            last_round_usage.get("cache_creation_input_tokens"), 0
        )
        cache_read = IntCodec.coerce(last_round_usage.get("cache_read_input_tokens"), 0)
    # Build the single thinking block from the accumulated body + signature.
    # The signature MUST be present whenever the body is -- otherwise a
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
            request=input_tokens,
            response=output_tokens,
            cache_write=cache_creation,
            cache_read=cache_read,
        ),
        stop_reason=normalize_stop_reason(
            stop_reason,
            kind="anthropic",
            has_tool_use=False,
        ),
        message_id=message_id,
        request_id=message_id,
        # The CLI reports one server-computed total per turn with no bucket
        # breakdown. Booking it to ``request`` keeps ``spend.total`` exact;
        # the per-bucket split is simply not observable on this transport.
        spend=TokenCost(request=total_cost),
    )
