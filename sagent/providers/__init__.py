"""Model providers (Anthropic, Google, OpenAI, Moonshot, DashScope, MiniMax, ...).

Each provider class has a ``DEFAULT_MODEL`` class attr and one or
more zero-arg ``from_*`` classmethods (e.g. ``from_env``,
``from_key``). Host scripts (``cli.py``, ``slack.py``) pick a
provider + method by name and dispatch via ``getattr`` - no shared
registry here.

``Moonshot``, ``DashScope``, ``MiniMax`` - and any future OpenAI chat-
completions compatible endpoint including self-hosted
vLLM/SGLang - subclass ``OpenAICompat``. Override a handful of class
attrs (DEFAULT_MODEL, ENV_VAR, BASE_URL, PRICING) and you're done.
Pass ``base_url=`` to ``from_env`` / ``from_key`` to point any of
them at a localhost inference server.
"""

from typing import Literal, get_args

from sagent.providers.anthropic.api import Anthropic
from sagent.providers.anthropic.cli import AnthropicCLI
from sagent.providers.dashscope.api import DashScope
from sagent.providers.google.api import Google
from sagent.providers.google.cli import GoogleCLI
from sagent.providers.llamacpp.server import LlamaCpp
from sagent.providers.minimax.api import MiniMax
from sagent.providers.moonshot.api import Moonshot
from sagent.providers.openai.api import OpenAI
from sagent.providers.openai.compat import OpenAICompat
from sagent.providers.openai.sub import OpenAISubscription
from sagent.providers.providers import (
    build_provider,
    default_auth_for_provider,
    infer_provider,
    supported_provider_options,
)
from sagent.providers.selfhosted.server import (
    SelfHosted,
    SelfHostedModel,
)


ProviderName = Literal[
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

PROVIDER_NAMES: tuple[ProviderName, ...] = get_args(ProviderName)

__all__ = [
    "PROVIDER_NAMES",
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
    "ProviderName",
    "SelfHosted",
    "SelfHostedModel",
    "build_provider",
    "default_auth_for_provider",
    "infer_provider",
    "supported_provider_options",
]
