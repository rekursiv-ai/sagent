"""Provider dispatch and model-id inference."""

from __future__ import annotations

from typing import Final, cast

import inspect
import sys

from sagent.types.providers import Provider, ProviderOptions


_MODEL_PROVIDER_MAP: Final[list[tuple[str, str]]] = [
    ("claude", "Anthropic"),
    ("gemini", "Google"),
    ("gpt", "OpenAI"),
    ("chatgpt", "OpenAI"),
    ("o1", "OpenAI"),
    ("o3", "OpenAI"),
    ("o4", "OpenAI"),
    ("codex", "OpenAI"),
    ("kimi", "Moonshot"),
    ("moonshot", "Moonshot"),
    ("qwen", "DashScope"),
    ("minimax", "MiniMax"),
]

# Maps API-key provider names to their account-auth (subscription /
# credentials-file) variant. When ``infer_provider`` sees a model_id
# whose prefix maps to an API-key provider (e.g. ``claude-...`` →
# ``Anthropic``) AND the user's *current* provider is the account
# variant (e.g. ``AnthropicCLI``), the inference resolves to the
# account variant + ``credentials`` auth instead of the API-key path.
# Without this, ``AgentSelf(model_id="claude-sonnet-4-6")`` from an
# AnthropicCLI-backed agent would silently try to build a fresh
# Anthropic API provider, requiring ``ANTHROPIC_API_KEY`` to be set.
_ACCOUNT_OVERRIDES: dict[str, str] = {
    "Anthropic": "AnthropicCLI",
}
_ACCOUNT_PROVIDERS: set[str] = set(_ACCOUNT_OVERRIDES.values())


def infer_provider(
    model_id: str,
    current_provider: str,
) -> tuple[str, str] | None:
    """Infer ``(provider, auth)`` from a model-id prefix.

    Args:
      model_id: Model identifier (e.g. ``"claude-sonnet-4-6"``).
      current_provider: Name of the currently active provider.

    Returns:
      provider_auth: ``(provider_name, auth_method)`` tuple, or ``None``
          when the model already matches ``current_provider`` or can't
          be mapped.

    """
    if _is_local_model_path(model_id):
        return ("SelfHosted", model_id)

    for prefix, base_prov in _MODEL_PROVIDER_MAP:
        if model_id.startswith(prefix):
            if current_provider.startswith(base_prov):
                # Same vendor family: the current provider is either the base
                # itself or one of its variants (``OpenAISubscription``,
                # ``AnthropicCLI``, ...). Preserve it rather than re-infer to
                # the bare API-key sibling, which would demand a key the
                # operator never set. This guard is independent of the account
                # override table, so it holds for variants that table does not
                # enumerate (e.g. ``OpenAISubscription`` in the public build).
                return None

            account_target = _ACCOUNT_OVERRIDES.get(base_prov)
            target = (
                account_target
                if account_target is not None and current_provider == account_target
                else base_prov
            )
            if target == current_provider:
                return None
            auth = "credentials" if target in _ACCOUNT_PROVIDERS else "env"
            return (target, auth)
    return None


def _is_local_model_path(model_id: str) -> bool:
    """Return whether ``model_id`` looks like a local HF snapshot path."""
    return model_id.startswith(("/", "./", "../", "~/"))


def build_provider(
    provider_name: str,
    auth: str = "env",
    *,
    account: str | None = None,
    options: ProviderOptions | None = None,
) -> Provider:
    """Dispatch ``<provider>.from_<auth>()``.

    Args:
      provider_name: Provider class name (e.g. ``"Anthropic"``).
      auth: Auth method suffix (``"env"``, ``"credentials"``, ``"key"``).
      account: Credential slot forwarded to providers that accept it.
          Ignored by providers without an ``account`` parameter.
      options: Construction-time knobs. Every explicitly set field must
          appear in the provider class's ``supported_options``
          declaration; an unsupported set field raises rather than
          being silently dropped.

    Returns:
      provider: Constructed provider instance.

    Raises:
      AttributeError: If the provider class is unknown or has no
          matching auth method.
      ValueError: If ``options`` sets a field the provider does not
          declare in ``supported_options``.

    """
    providers = sys.modules["sagent.providers"]
    cls = getattr(providers, provider_name, None)
    if cls is None:
        raise AttributeError(f"unknown provider {provider_name!r}")
    factory = getattr(cls, f"from_{auth}", None)
    if factory is None:
        raise AttributeError(
            f"provider {provider_name!r} has no ``from_{auth}`` method",
        )
    kwargs: dict[str, object] = dict((options or ProviderOptions()).set_fields())
    supported = supported_provider_options(provider_name)
    unsupported = sorted(kwargs.keys() - supported)
    if unsupported:
        raise ValueError(
            f"provider {provider_name!r} does not support option(s) "
            f"{', '.join(unsupported)}; supported: "
            f"{', '.join(sorted(supported)) or '(none)'}"
        )
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        sig = None
    if sig is not None and "account" in sig.parameters:
        kwargs["account"] = account
    return factory(**kwargs)


def supported_provider_options(provider_name: str) -> frozenset[str]:
    """Return the ``ProviderOptions`` field names ``provider_name`` supports.

    Provider classes declare support via a ``supported_options`` class
    attribute; a class without the attribute takes no construction
    options.

    Args:
      provider_name: Provider class name (e.g. ``"Anthropic"``).

    Returns:
      supported: Declared ``ProviderOptions`` field names.

    Raises:
      AttributeError: If the provider class is unknown.

    """
    providers = sys.modules["sagent.providers"]
    cls = getattr(providers, provider_name, None)
    if cls is None:
        raise AttributeError(f"unknown provider {provider_name!r}")
    return cast("frozenset[str]", getattr(cls, "supported_options", frozenset[str]()))


def default_auth_for_provider(provider_name: str) -> str:
    """Return the conventional auth method for ``provider_name``.

    Args:
      provider_name: Provider class name.

    Returns:
      auth: Default auth suffix for that provider.

    Raises:
      AttributeError: If the provider class is unknown or has no defaultable auth.

    """
    providers = sys.modules["sagent.providers"]
    cls = getattr(providers, provider_name, None)
    if cls is None:
        raise AttributeError(f"unknown provider {provider_name!r}")
    if provider_name.endswith(("CLI", "Subscription")) and hasattr(
        cls, "from_credentials"
    ):
        return "credentials"
    if hasattr(cls, "from_env"):
        return "env"
    if hasattr(cls, "from_key"):
        return "key"
    raise AttributeError(f"provider {provider_name!r} has no default auth method")
