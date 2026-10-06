"""Provider factory contract and the auth hook.

Which model a name means, and what it can do over a transport, is
``sagent.catalog.table``'s; a provider only holds its
``ModelCatalog`` and builds ``Model`` instances from what it resolves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from sagent.catalog.table import ModelCatalog
    from sagent.types.model import Model


__all__ = [
    "AuthReloadable",
    "ModelResolver",
    "Provider",
]


@runtime_checkable
class ModelResolver(Protocol):
    """Provider class or instance exposing its resolved catalog directly."""

    catalog: ModelCatalog


@runtime_checkable
class Provider(Protocol):
    """Factory for model backends. ``None`` selects catalog key ``default``.

    Whoever builds a provider owns it and calls :meth:`close_sdk` once. One
    provider can back several models (``sagent --advisor`` builds two), so a
    client the provider shares is closed here, never by ``Model.close``.
    """

    def model(
        self,
        model_id: str | None = None,
    ) -> Model:
        """Build a model backend.

        Args:
          model_id: Provider-specific id; ``None`` selects ``default``.

        Returns:
          model: A ``Model`` ready to handle requests.

        """
        ...

    async def close_sdk(self) -> None:
        """Close any client this provider shares across its models.

        Total and idempotent: a provider whose models own their resources
        (CLI subprocesses, per-model HTTP clients) returns immediately.
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
