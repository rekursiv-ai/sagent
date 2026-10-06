"""Provider dispatch and model-id inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, cast, get_args

import inspect
import sys


if TYPE_CHECKING:
    from sagent.types.providers import Provider


type ProviderName = Literal[
    "Anthropic",
    "AnthropicCLI",
    "DashScope",
    "Google",
    "GoogleCLI",
    "LlamaCpp",
    "MiniMax",
    "Moonshot",
    "OpenAI",
    "OpenAICompat",
    "OpenAISubscription",
    "SelfHosted",
]

PROVIDER_NAMES: Final[tuple[ProviderName, ...]] = cast(
    tuple[ProviderName, ...],
    get_args(ProviderName.__value__),
)

# Maps API-key provider names to their account-auth (subscription /
# credentials-file) variant. When ``infer_provider`` sees a model_id
# whose catalog row maps to an API-key provider (e.g. ``opus-4.8`` →
# ``Anthropic``) AND the user's *current* provider is the account
# variant (e.g. ``AnthropicCLI``), the inference resolves to the
# account variant + ``credentials`` auth instead of the API-key path.
# Without this, ``AgentSelf(model_id="sonnet-4.6")`` from an
# AnthropicCLI-backed agent would silently try to build a fresh
# Anthropic API provider, requiring ``ANTHROPIC_API_KEY`` to be set.
_ACCOUNT_OVERRIDES: Final[dict[str, str]] = {
    "Anthropic": "AnthropicCLI",
}
_ACCOUNT_PROVIDERS: Final[frozenset[str]] = frozenset(_ACCOUNT_OVERRIDES.values())


def infer_provider(
    model_id: str,
    current_provider: str,
) -> tuple[str, str] | None:
    """Infer ``(provider, auth)`` from a local path or catalog membership.

    Args:
      model_id: Model identifier (e.g. ``"sonnet-4.6"``).
      current_provider: Name of the currently active provider.

    Returns:
      provider_auth: ``(provider_name, auth_method)`` tuple;
          ``("SelfHosted", model_id)`` for a filesystem path, whose second
          element is the path rather than an auth method; or ``None`` when
          the current provider's catalog already accepts the model or no
          single API provider's catalog does.

    """
    if model_id.startswith(("/", "./", "../", "~/")):
        return ("SelfHosted", model_id)

    if _provider_accepts(current_provider, model_id):
        return None
    prefer_account = current_provider in _ACCOUNT_PROVIDERS
    base_prov = _catalog_provider(model_id)
    if base_prov is not None:
        target = (
            _ACCOUNT_OVERRIDES.get(base_prov, base_prov)
            if prefer_account
            else base_prov
        )
        if target == current_provider:
            return None
        auth = "credentials" if target in _ACCOUNT_PROVIDERS else "env"
        return (target, auth)
    return None


def provider_class(provider_name: str) -> type | None:
    """Look a provider class up by name on the ``providers`` package facade.

    The facade is the registry: every provider is re-exported there, and
    reading it through ``sys.modules`` keeps this module -- and the agent
    layer that calls it -- below the facade in the import graph.

    Args:
      provider_name: Provider class name (e.g. ``"Anthropic"``).

    Returns:
      cls: The provider class, or ``None`` when the name is unknown.

    """
    providers = sys.modules["sagent.providers"]
    cls = getattr(providers, provider_name, None)
    return cls if isinstance(cls, type) else None


def build_provider(
    provider_name: str,
    auth: str = "env",
    *,
    account: str | None = None,
) -> Provider:
    """Dispatch ``<provider>.from_<auth>()``.

    Args:
      provider_name: Provider class name (e.g. ``"Anthropic"``).
      auth: Auth method suffix such as ``"env"`` or ``"credentials"``.
      account: Credential slot forwarded to providers that accept it.
          Ignored by providers without an ``account`` parameter.

    Returns:
      provider: Constructed provider instance.

    Raises:
      AttributeError: If the provider class is unknown or has no
          matching auth method.

    """
    cls = provider_class(provider_name)
    if cls is None:
        raise AttributeError(f"unknown provider {provider_name!r}")
    factory = getattr(cls, f"from_{auth}", None)
    if factory is None:
        raise AttributeError(
            f"provider {provider_name!r} has no ``from_{auth}`` method",
        )
    kwargs: dict[str, object] = {}
    if "account" in inspect.signature(factory).parameters:
        kwargs["account"] = account
    return factory(**kwargs)


def build_provider_with_account_fallback(
    provider_name: str,
    auth: str,
    *,
    account: str | None,
    fallback_to_default: bool,
) -> tuple[Provider, str | None]:
    """Build a provider, retrying its default account when permitted.

    Args:
      provider_name: Provider class name.
      auth: Auth method suffix.
      account: Credential account attempted first.
      fallback_to_default: Whether a missing named account may retry default.

    Returns:
      provider_account: Constructed provider and the account that succeeded.

    """
    try:
        return build_provider(provider_name, auth, account=account), account
    except FileNotFoundError:
        if not fallback_to_default or account in (None, "", "default"):
            raise
    return build_provider(provider_name, auth, account=None), None


def default_auth_for_provider(provider_name: str) -> str:
    """Return the conventional auth method for ``provider_name``.

    Args:
      provider_name: Provider class name.

    Returns:
      auth: Default auth suffix for that provider.

    Raises:
      AttributeError: If the provider class is unknown or has no defaultable auth.

    """
    cls = provider_class(provider_name)
    if cls is None:
        raise AttributeError(f"unknown provider {provider_name!r}")
    if provider_name.endswith(("CLI", "Subscription")) and hasattr(
        cls,
        "from_credentials",
    ):
        return "credentials"
    if hasattr(cls, "from_env"):
        return "env"
    raise AttributeError(f"provider {provider_name!r} has no default auth method")


def _catalog_provider(model_id: str) -> str | None:
    """Return the unique API provider whose catalog accepts ``model_id``."""
    matches: list[str] = []
    for provider in PROVIDER_NAMES:
        if provider.endswith(("CLI", "Subscription")):
            continue
        cls = provider_class(provider)
        catalog = getattr(cls, "catalog", None)
        if catalog is None:
            continue
        try:
            catalog.resolve(model_id)
        except ValueError:
            continue
        matches.append(provider)
    return matches[0] if len(matches) == 1 else None


def _provider_accepts(provider: str, model_id: str) -> bool:
    """Return whether the provider catalog resolves the model id."""
    cls = provider_class(provider)
    catalog = getattr(cls, "catalog", None)
    if catalog is None:
        return False
    try:
        catalog.resolve(model_id)
    except ValueError:
        return False
    return True
