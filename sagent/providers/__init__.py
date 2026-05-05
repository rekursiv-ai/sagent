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

from sagent.providers.anthropic import Anthropic
from sagent.providers.dashscope import DashScope
from sagent.providers.google import Google
from sagent.providers.minimax import MiniMax
from sagent.providers.moonshot import Moonshot
from sagent.providers.openai import OpenAI
from sagent.providers.openai_compat import OpenAICompat
from sagent.providers.providers import build_provider, infer_provider
from sagent.providers.selfhosted import SelfHosted, SelfHostedModel


ProviderName = Literal[
    "Anthropic",
    "DashScope",
    "Google",
    "MiniMax",
    "Moonshot",
    "OpenAI",
    "OpenAICompat",
    "SelfHosted",
]

PROVIDER_NAMES: tuple[ProviderName, ...] = get_args(ProviderName)


__all__ = [
    "PROVIDER_NAMES",
    "Anthropic",
    "DashScope",
    "Google",
    "MiniMax",
    "Moonshot",
    "OpenAI",
    "OpenAICompat",
    "ProviderName",
    "SelfHosted",
    "SelfHostedModel",
    "build_provider",
    "infer_provider",
]
