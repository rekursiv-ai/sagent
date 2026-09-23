"""Provider factory contract, model-id resolution, and the auth hook.

A ``Provider`` builds ``Model`` instances from a catalog of
``ModelCapability`` rows. ``resolve`` is the one place a tagged model id
becomes a capability met with its transport plus the settings that id
selected.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelSettings,
)
from sagent.types.model import (
    Model,
    split_model_id,
)


__all__ = [
    "AuthReloadable",
    "ModelCatalog",
    "ModelResolver",
    "Provider",
    "ProviderCloseable",
    "UnknownModelError",
    "UnsupportedTagError",
]


class UnknownModelError(ValueError):
    """The base id is absent from the provider's catalog."""


class UnsupportedTagError(ValueError):
    """The id carries a context tag the model does not offer."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCatalog:
    """Resolved model rows and the transport restrictions applied to them."""

    rows: Mapping[str, ModelCapability]
    transport: ModelCapability

    def __post_init__(self) -> None:
        """Snapshot rows so a global provider catalog cannot be mutated."""
        object.__setattr__(self, "rows", MappingProxyType(dict(self.rows)))

    def resolve(self, model_id: str) -> tuple[ModelCapability, ModelSettings]:
        """Resolve ``model_id`` against this catalog's transport.

        Args:
          model_id: Catalog key or vendor wire id, optionally context-tagged.

        Returns:
          capability: The catalog capability narrowed by the transport.
          settings: Settings selecting the requested context.

        Raises:
          UnknownModelError: The base id is absent.
          UnsupportedTagError: Context tags conflict or are unavailable.

        """
        base, id_tags = split_model_id(model_id)
        tags: list[ContextTag] = sorted(id_tags)
        if len(tags) > 1:
            joined = ", ".join(tags)
            raise UnsupportedTagError(
                f"Model {model_id!r} selects conflicting contexts: {joined}",
            )
        context: ContextTag = tags[0] if tags else ""
        row = self.rows.get(base)
        if row is None:
            row = next(
                (
                    candidate
                    for candidate in self.rows.values()
                    if candidate.wire_model_id == base
                ),
                None,
            )
        if row is None:
            known = ", ".join(sorted(self.rows))
            raise UnknownModelError(
                f"Unknown model {model_id!r}. Known models: {known}",
            )
        capability = row & self.transport
        if (
            context == "+1m"
            and context not in capability.context
            and capability.context[""].max_request_tokens >= 1_000_000
        ):
            capability = replace(
                capability,
                context=MappingProxyType(
                    {**capability.context, "+1m": capability.context[""]},
                ),
            )
        if context not in capability.context:
            offered = ", ".join(sorted(t for t in capability.context if t)) or "(none)"
            raise UnsupportedTagError(
                f"Unknown model {model_id!r}: {base} has no {context} context;"
                f" offers: {offered}",
            )
        return capability, ModelSettings.narrowest(capability, context=context)


@runtime_checkable
class ModelResolver(Protocol):
    """Provider class or instance exposing its resolved catalog directly."""

    catalog: ModelCatalog


@runtime_checkable
class Provider(Protocol):
    """Factory for model backends. ``None`` selects catalog key ``default``."""

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


@runtime_checkable
class ProviderCloseable(Protocol):
    """Provider that owns a client its models share.

    Teardown belongs here, not on ``Model.close``: one provider can back
    several models (``sagent --advisor`` builds two), so a model closing
    the shared client strands its siblings. Whoever built the provider
    closes it, once.

    API-family providers holding an SDK or HTTP client satisfy this;
    CLI-family providers own their resources per model and don't.
    """

    async def close_sdk(self) -> None:
        """Close the client this provider opened on the running loop."""
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
