"""Provider dispatch and model-id inference."""

from __future__ import annotations

import inspect
import sys

from sagent.types.providers import Provider, ProviderOptions


_MODEL_PROVIDER_MAP: list[tuple[str, str]] = [
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

# Maps API-key provider names to their same-vendor account-auth
# (subscription / credentials-file) variant. ``infer_provider`` keeps
# the current account variant only when the inferred model belongs to
# that same vendor. Cross-vendor inference retains the destination's
# API-key default; choosing another vendor's subscription provider must
# be explicit.
_ACCOUNT_OVERRIDES: dict[str, str] = {
    "Anthropic": "AnthropicCLI",
    "OpenAI": "OpenAISubscription",
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
    return getattr(cls, "supported_options", frozenset())


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
