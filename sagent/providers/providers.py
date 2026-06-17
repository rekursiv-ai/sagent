"""Provider dispatch and model-id inference."""

from __future__ import annotations

from collections.abc import Iterable

import argparse
import inspect
import json
import logging
import sys

from sagent.types.providers import Provider


_LOGGER = logging.getLogger(__name__)


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


def _is_local_model_path(model_id: str) -> bool:
    """Return whether ``model_id`` looks like a local HF snapshot path."""
    return model_id.startswith(("/", "./", "../", "~/"))


def build_provider(
    provider_name: str,
    auth: str = "env",
    *,
    account: str | None = None,
    **extra: object,
) -> Provider:
    """Dispatch ``<provider>.from_<auth>()``.

    Args:
      provider_name: Provider class name (e.g. ``"Anthropic"``).
      auth: Auth method suffix (``"env"``, ``"credentials"``, ``"key"``).
      account: Credential slot forwarded to providers that accept it.
          Ignored by providers without an ``account`` parameter.
      **extra: Additional kwargs forwarded to the factory IFF its
          signature accepts them. Lets callers (CLI flags, programmatic
          users) tune provider-specific knobs (e.g.
          ``server_side_context_management`` on ``Anthropic``) without
          this dispatcher needing to know about each one.

    Returns:
      provider: Constructed provider instance.

    Raises:
      AttributeError: If the provider class is unknown or has no
          matching auth method.

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
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        sig = None
    kwargs: dict[str, object] = {}
    if sig is not None:
        if "account" in sig.parameters:
            kwargs["account"] = account
        for k, v in extra.items():
            if k in sig.parameters:
                kwargs[k] = v
            else:
                _LOGGER.warning(
                    "provider %s factory ``%s`` does not accept ``%s``;"
                    " dropping value %r",
                    provider_name,
                    factory.__qualname__,
                    k,
                    v,
                )
    return factory(**kwargs)


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


def parse_provider_arg(spec: str) -> tuple[str, str, object]:
    """Parse a ``Class.key=JSON-value`` triple from a CLI ``--provider-arg``.

    Args:
      spec: A literal of the form ``Class.key=JSON``. ``JSON`` is decoded
          with ``json.loads``; on decode failure the raw string is returned
          as the value (so ``--provider-arg X.path=/tmp/foo`` works without
          shell-quoting the path).

    Returns:
      triple: ``(class_name, key, value)`` ready to be merged into a
          per-class kwargs dict.

    Raises:
      argparse.ArgumentTypeError: If ``spec`` is missing the required
          ``Class.key=value`` shape.

    """
    cls_name, dot, rest = spec.partition(".")
    if not dot or not cls_name:
        raise argparse.ArgumentTypeError(
            f"expected Class.key=value, got {spec!r}",
        )
    key, eq, raw = rest.partition("=")
    if not eq or not key:
        raise argparse.ArgumentTypeError(
            f"expected Class.key=value, got {spec!r}",
        )
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return cls_name, key, value


def collect_provider_args(
    specs: Iterable[str],
    provider_name: str,
) -> dict[str, object]:
    """Resolve ``--provider-arg`` specs against a provider class via MRO.

    A spec keyed on ``OpenAI`` applies to ``OpenAISubscription`` too --
    the subscription inherits the constructor surface from ``OpenAI``.
    Walking the MRO of the chosen class lets the user target a base
    class without re-typing it for every subclass.

    Specs targeting a class outside the chosen provider's MRO are
    silently ignored (the user may pass multi-provider configs in one
    session).

    Args:
      specs: Iterable of raw ``Class.key=JSON`` strings (already
          ``parse_provider_arg``-able).
      provider_name: The leaf provider class chosen for this session.

    Returns:
      kwargs: Merged ``{key: value}`` dict, with leaf-class specs
          winning over base-class specs on key collision.

    """
    providers = sys.modules["sagent.providers"]
    cls = getattr(providers, provider_name, None)
    if cls is None:
        raise AttributeError(f"unknown provider {provider_name!r}")
    mro_names = {c.__name__ for c in cls.__mro__}
    # Bucket by class name so leaf specs can overwrite base specs at apply
    # time. ``cls.__mro__`` orders leaf-first; reverse it so that when we
    # later merge we end with leaf-most values winning.
    by_class: dict[str, dict[str, object]] = {}
    for spec in specs:
        cls_name, key, value = parse_provider_arg(spec)
        if cls_name not in mro_names:
            continue
        by_class.setdefault(cls_name, {})[key] = value
    merged: dict[str, object] = {}
    for ancestor in reversed(cls.__mro__):
        if ancestor.__name__ in by_class:
            merged.update(by_class[ancestor.__name__])
    return merged
