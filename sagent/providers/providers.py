"""Provider dispatch and model-id inference."""

from __future__ import annotations

import inspect
import sys

from sagent.custom_types import Provider


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

_ACCOUNT_OVERRIDES: dict[str, str] = {}
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
    prefer_account = current_provider in _ACCOUNT_PROVIDERS
    for prefix, base_prov in _MODEL_PROVIDER_MAP:
        if model_id.startswith(prefix):
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


def build_provider(
    provider_name: str,
    auth: str = "env",
    *,
    account: str | None = None,
) -> Provider:
    """Dispatch ``<provider>.from_<auth>()`` or treat *auth* as a literal key.

    Args:
      provider_name: Provider class name (e.g. ``"Anthropic"``).
      auth: Either a method suffix (``"env"``, ``"credentials"``) or a
          literal API key string. When no ``from_<auth>`` method exists
          the value is forwarded to ``from_key(auth)``.
      account: Credential slot forwarded to providers that accept it.
          Ignored by providers without an ``account`` parameter.

    Returns:
      provider: Constructed provider instance.

    Raises:
      AttributeError: If the provider class is unknown or has no
          matching auth method and no ``from_key``.

    """
    providers = sys.modules["sagent.providers"]
    cls = getattr(providers, provider_name, None)
    if cls is None:
        raise AttributeError(f"unknown provider {provider_name!r}")
    factory = getattr(cls, f"from_{auth}", None)
    if factory is not None:
        kwargs: dict[str, object] = {}
        try:
            sig = inspect.signature(factory)
        except (TypeError, ValueError):
            sig = None
        if sig is not None and "account" in sig.parameters:
            kwargs["account"] = account
        return factory(**kwargs)
    # No from_{auth} method — treat auth as a literal API key.
    from_key = getattr(cls, "from_key", None)
    if from_key is None:
        raise AttributeError(
            f"provider {provider_name!r} has no ``from_{auth}`` method "
            f"and no ``from_key``",
        )
    return from_key(auth)
