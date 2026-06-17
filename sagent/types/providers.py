"""Provider factory contract and the optional auth-reload hook.

A ``Provider`` builds ``Model`` instances. ``AuthReloadable`` is the
hot-reload-credentials Protocol opt-in for OAuth-backed providers.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from sagent.types.model import Model


__all__ = [
    "AuthReloadable",
    "Provider",
]


@runtime_checkable
class Provider(Protocol):
    """Factory for model backends. ``None`` -> provider's ``DEFAULT_MODEL``."""

    def model(
        self,
        model_id: str | None = None,
        max_request_tokens: int | None = None,
        **provider_options: Any,
    ) -> Model:
        """Build a model backend.

        Args:
          model_id: Provider-specific id; ``None`` selects the
              provider's ``DEFAULT_MODEL``.
          max_request_tokens: Override for the model's input cap.
          provider_options: Provider-specific construction knobs. Each
              concrete provider declares these as real, typed keyword
              args on its own ``model(...)`` (e.g. ``AnthropicCLI``
              accepts ``extra_mcp_servers`` / ``subprocess_read_timeout_sec``);
              the protocol acknowledges the nonstandard tail here so the
              uniform call site holds. A provider rejects keys it does
              not recognize rather than silently ignoring them.

        Returns:
          model: A ``Model`` ready to handle requests.

        """
        ...

    def utility_model(self) -> Model:
        """Build the cheapest/fastest model for utility tasks.

        Returns:
          model: A low-cost ``Model`` for internal use (summarizers, etc.).

        """
        ...


@runtime_checkable
class AuthReloadable(Protocol):
    """Provider that can hot-reload OAuth credentials after a re-login.

    Implementations re-read the credential file from disk and refresh
    any in-memory token state. The contract is reused by the auth-error
    retry path (mid-call 401) and by ``Agent.relogin`` (explicit user-
    triggered re-auth), since both need the same "freshen the running
    provider's tokens" semantic.

    Anthropic-family and Google-family Subscription providers satisfy
    this Protocol today; API-key providers don't (no tokens to refresh).
    """

    async def handle_auth_error(self) -> None:
        """Hot-reload credentials from disk into the running provider."""
        ...
