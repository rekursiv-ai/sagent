"""Provider factory contract, model-id resolution, and the auth hook.

A ``Provider`` builds ``Model`` instances from a catalog of
``ModelCapability`` rows. ``resolve`` is the one place a tagged model id
becomes a capability met with its transport plus the settings that id
selected.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, cast, runtime_checkable

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
    "ModelRole",
    "Provider",
    "ProviderCloseable",
    "UnknownModelError",
    "UnsupportedTagError",
    "resolve",
]


type ModelRole = Literal["default", "utility"]
"""A model selected by what it is FOR, not by id."""


class UnknownModelError(ValueError):
    """The base id is absent from the provider's catalog."""


class UnsupportedTagError(ValueError):
    """The id carries a context tag the model does not offer."""


def resolve(
    model_id: str | ModelRole,
    *,
    models: Mapping[str, ModelCapability],
    roles: Mapping[ModelRole, str],
    transport: ModelCapability,
) -> tuple[ModelCapability, ModelSettings]:
    """Turn a tagged model id into a capability and the settings it selects.

    Args:
      model_id: Catalog id with an optional ``+1m`` / ``+200k`` tag, or a
          role name resolved through ``roles``.
      models: The provider's capability catalog, keyed by base id.
      roles: Role name to base id (``default``, ``utility``).
      transport: What this transport lets through.

    Returns:
      capability: The catalog row met with ``transport``.
      settings: The choices the id encoded.

    Raises:
      UnknownModelError: The base id is not in ``models``.
      UnsupportedTagError: The id asks for a context the model does not
          offer. Silently serving the base window would understate the
          caller's budget by up to 4x.

    """
    tagged: str = roles.get(cast(ModelRole, model_id), model_id)
    base, id_tags = split_model_id(tagged)
    tags: list[ContextTag] = sorted(id_tags)
    context: ContextTag = tags[0] if tags else ""
    row = models.get(base)
    if row is None:
        known = ", ".join(sorted(models))
        raise UnknownModelError(f"Unknown model {model_id!r}. Known models: {known}")
    capability = row & transport
    if context not in capability.context:
        offered = ", ".join(sorted(t for t in capability.context if t)) or "(none)"
        raise UnsupportedTagError(
            f"Unknown model {model_id!r}: {base} has no {context} context;"
            f" offers: {offered}",
        )
    return capability, ModelSettings.narrowest(capability, context=context)


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
