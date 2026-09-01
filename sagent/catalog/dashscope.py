"""DashScope/Qwen model catalog, expressed as ``ModelCapability`` rows.

Source: https://help.aliyun.com/zh/model-studio/models

A row carries only what the MODEL can do; caching, retry, and auth mode
are transport facts declared on ``OpenAICompat.TRANSPORT``. These vendors
publish no image pixel or byte ceiling and preprocess images server-side,
so ``ModelLimits`` carries only the two token windows.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from sagent.catalog.capability import (
    ModelCapability,
    ModelLimits,
)
from sagent.catalog.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)
from sagent.catalog.thinking import ThinkingEffort


__all__ = ["MODELS"]


def _prices(*, request: float, response: float) -> PriceCatalog:
    """USD per million tokens; DashScope quotes one flat tier."""
    return PriceCatalog(
        {PriceCatalogProduct(): TokenPrice(request=request, response=response)}
    )


# DashScope rejects ``reasoning_effort``; it takes ``enable_thinking`` plus an
# optional ``thinking_budget`` reasoning-token cap. Mirrors Google's ladder so
# one effort knob drives both. ``off`` is a zero budget, which the wire maps to
# ``enable_thinking=False``.
_QWEN_BUDGETS: Mapping[ThinkingEffort, str] = MappingProxyType(
    {
        "off": "0",
        "min": "1024",
        "low": "4096",
        "medium": "8192",
        "high": "16384",
        "xhigh": "20480",
        "max": "24576",
    }
)

# The ``-thinking`` ids are thinking-only: Model Studio rejects
# ``enable_thinking=false`` on them with "The value of the enable_thinking
# parameter is restricted to True". ``thinking_budget`` still applies.
# https://www.alibabacloud.com/help/en/model-studio/error-code
_QWEN_THINKING_ONLY_BUDGETS: Mapping[ThinkingEffort, str] = MappingProxyType(
    {k: v for k, v in _QWEN_BUDGETS.items() if k != "off"}
)


MODELS: Mapping[str, ModelCapability] = MappingProxyType(
    {
        "qwen3.6-max-preview": ModelCapability(
            model_id="qwen3.6-max-preview",
            context_limits=ModelLimits(
                max_request_tokens=262_144,
                max_response_tokens=65_536,
            ),
            prices=_prices(request=1.6, response=6.4),
            supported_thinking_efforts=_QWEN_BUDGETS,
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "qwen3.6-plus": ModelCapability(
            model_id="qwen3.6-plus",
            context_limits=ModelLimits(
                max_request_tokens=1_000_000,
                max_response_tokens=65_536,
            ),
            prices=_prices(request=0.5, response=3.0),
            supported_thinking_efforts=_QWEN_BUDGETS,
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "qwen3.6-flash": ModelCapability(
            model_id="qwen3.6-flash",
            context_limits=ModelLimits(
                max_request_tokens=1_000_000,
                max_response_tokens=65_536,
            ),
            prices=_prices(request=0.05, response=0.2),
            supported_thinking_efforts=_QWEN_BUDGETS,
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "qwen3-235b-a22b-instruct-2507": ModelCapability(
            model_id="qwen3-235b-a22b-instruct-2507",
            context_limits=ModelLimits(
                max_request_tokens=262_144,
                max_response_tokens=65_536,
            ),
            prices=_prices(request=0.7, response=2.8),
            # ``-instruct`` / ``-coder`` / ``-turbo`` reject the toggle.
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset(),
            supported_thinking_outputs=frozenset(),
        ),
        "qwen3-235b-a22b-thinking-2507": ModelCapability(
            model_id="qwen3-235b-a22b-thinking-2507",
            context_limits=ModelLimits(
                max_request_tokens=262_144,
                max_response_tokens=32_768,
            ),
            prices=_prices(request=0.7, response=8.4),
            supported_thinking_efforts=_QWEN_THINKING_ONLY_BUDGETS,
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "qwen3-30b-a3b-instruct-2507": ModelCapability(
            model_id="qwen3-30b-a3b-instruct-2507",
            context_limits=ModelLimits(
                max_request_tokens=262_144,
                max_response_tokens=65_536,
            ),
            prices=_prices(request=0.2, response=0.8),
            # ``-instruct`` / ``-coder`` / ``-turbo`` reject the toggle.
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset(),
            supported_thinking_outputs=frozenset(),
        ),
        "qwen3-32b": ModelCapability(
            model_id="qwen3-32b",
            context_limits=ModelLimits(
                max_request_tokens=262_144,
                max_response_tokens=65_536,
            ),
            prices=_prices(request=0.4, response=1.2),
            supported_thinking_efforts=_QWEN_BUDGETS,
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "qwen3-coder-480b-a35b-instruct": ModelCapability(
            model_id="qwen3-coder-480b-a35b-instruct",
            context_limits=ModelLimits(
                max_request_tokens=262_144,
                max_response_tokens=65_536,
            ),
            prices=_prices(request=1.0, response=5.0),
            # ``-instruct`` / ``-coder`` / ``-turbo`` reject the toggle.
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset(),
            supported_thinking_outputs=frozenset(),
        ),
        "qwen-plus": ModelCapability(
            model_id="qwen-plus",
            context_limits=ModelLimits(
                max_request_tokens=1_000_000,
                max_response_tokens=32_768,
            ),
            prices=_prices(request=0.4, response=1.2),
            supported_thinking_efforts=_QWEN_BUDGETS,
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "qwen-max": ModelCapability(
            model_id="qwen-max",
            context_limits=ModelLimits(
                max_request_tokens=262_144,
                max_response_tokens=65_536,
            ),
            prices=_prices(request=1.6, response=6.4),
            supported_thinking_efforts=_QWEN_BUDGETS,
            supported_thinking_budgets=frozenset({"auto", "fixed"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "qwen-turbo": ModelCapability(
            model_id="qwen-turbo",
            context_limits=ModelLimits(
                max_request_tokens=1_000_000,
                max_response_tokens=32_768,
            ),
            prices=_prices(request=0.05, response=0.2),
            # ``-instruct`` / ``-coder`` / ``-turbo`` reject the toggle.
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset(),
            supported_thinking_outputs=frozenset(),
        ),
    }
)
