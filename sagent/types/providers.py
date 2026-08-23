"""Provider factory contract, model-id resolution, and the auth hook.

A ``Provider`` builds ``Model`` instances from a ``CAPABILITIES``
catalog. ``resolve`` is the one place a tagged model id (``+1m``,
``+fast``) becomes a narrowed ``ModelSpec``. ``AuthReloadable`` is the
hot-reload-credentials Protocol opt-in for OAuth-backed providers.
``ProviderOptions`` is the typed, exhaustive set of construction-time
provider knobs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, cast, runtime_checkable

import dataclasses

from sagent.types.model import (
    CONTEXT_TAGS,
    LATENCY_TAGS,
    Limits,
    Model,
    ModelCapability,
    ModelSpec,
    split_model_id,
)


__all__ = [
    "AuthReloadable",
    "ModelRole",
    "Provider",
    "ProviderCloseable",
    "ProviderOptions",
    "UnknownModelError",
    "UnsupportedTagError",
    "resolve",
]


type ModelRole = Literal["default", "utility"]
"""A model selected by what it is FOR, not by id."""


class UnknownModelError(ValueError):
    """The base id is absent from the provider's catalog."""


class UnsupportedTagError(ValueError):
    """The id carries an option tag the model does not offer."""


def resolve(
    model_id: str | ModelRole,
    *,
    models: Mapping[str, ModelCapability],
    roles: Mapping[ModelRole, str],
    transport: ModelCapability | None = None,
) -> ModelSpec:
    """Turn a tagged model id into a narrowed ``ModelSpec``.

    The single resolution path: strip the option tags, look the base id
    up, reject a tag the model does not offer, meet the row with the
    transport, and narrow to the selected context.

    Args:
      model_id: Catalog id with optional ``+1m`` / ``+fast`` tags, or a
          role name resolved through ``roles``.
      models: The provider's capability catalog, keyed by base id.
      roles: Role name to base id (``default``, ``utility``).
      transport: What this transport lets through; ``None`` restricts
          nothing.

    Returns:
      spec: The narrowed spec for this id on this transport.

    Raises:
      UnknownModelError: The base id is not in ``models``.
      UnsupportedTagError: The id asks for a context or fast tier the
          model does not offer. Silently serving the base window would
          understate the caller's budget by up to 4x.

    """
    tagged: str = roles.get(cast(ModelRole, model_id), model_id)
    # ``split_model_id`` is the one tag parser; it accepts the tags in any
    # order, which a hand-rolled suffix strip here did not.
    base, id_tags = split_model_id(tagged)
    fast = any(t in LATENCY_TAGS for t in id_tags)
    context = next((t for t in id_tags if t in CONTEXT_TAGS), "")
    cap = models.get(base)
    if cap is None:
        known = ", ".join(sorted(models))
        raise UnknownModelError(f"Unknown model {model_id!r}. Known models: {known}")
    limits = cap.context_limits
    offered_contexts = cast(
        Mapping[str, Limits],
        ({} if isinstance(limits, Limits) else limits),
    )
    if context and context not in offered_contexts:
        offered = ", ".join(sorted(t for t in offered_contexts if t)) or "(none)"
        raise UnsupportedTagError(
            f"Unknown model {model_id!r}: {base} has no {context} context;"
            f" offers: {offered}",
        )
    if transport is not None:
        cap = cap & transport
    # Check fast AFTER the meet: a transport that strips fast (the CLI
    # shape) would otherwise yield ``serve_fast=True`` on a spec whose
    # fast price row is gone -- sagent believing it serves fast, sending
    # no fast flag, and billing standard.
    if fast and not cap.serves_fast:
        raise UnsupportedTagError(f"Model {model_id!r} does not support fast mode")
    return ModelSpec.narrow(cap, context=context, fast=fast)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProviderOptions:
    """Construction-time provider knobs forwarded to provider factories.

    Every knob a provider factory accepts beyond auth/account lives
    here as a typed field -- there is no untyped passthrough. Each
    field is tri-state: ``None`` (the default) defers to the factory's
    own default, while an explicit value is forwarded and must be
    declared in the target provider class's ``supported_options``
    (``build_provider`` raises on an unsupported set field rather than
    silently dropping it).
    """

    redact_thinking: bool | None = None
    """Request redacted thinking blocks (Anthropic family)."""

    server_side_context_management: bool | None = None
    """Opt in to Anthropic's server-side ``clear_tool_uses`` beta."""

    def set_fields(self) -> dict[str, bool]:
        """Return the explicitly set (non-``None``) fields as factory kwargs.

        Returns:
          kwargs: ``{field_name: value}`` for every non-``None`` field.

        """
        return {
            field.name: value
            for field in dataclasses.fields(self)
            if (value := getattr(self, field.name)) is not None
        }


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
