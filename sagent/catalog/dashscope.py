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

from sagent.types.capability import (
    ModelCapability,
    ModelLimits,
    ThinkingEffort,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)


__all__ = ["models", "thinking_budget"]


def thinking_budget(effort: ThinkingEffort) -> str:
    """Return the ``thinking_budget`` token cap an effort sends.

    DashScope rejects ``reasoning_effort``; it takes ``enable_thinking``
    plus this cap, where ``0`` means ``enable_thinking=False``.

    Args:
      effort: Selected effort level.

    Returns:
      budget: Wire value, as the string the request body carries.

    """
    match effort:
        case "none":
            return "0"
        case "min":
            return "1024"
        case "low":
            return "4096"
        case "medium":
            return "8192"
        case "high":
            return "16384"
        case "xhigh":
            return "20480"
        case "max":
            return "24576"


def models() -> Mapping[str, ModelCapability]:
    """Return every model this vendor serves, keyed by base id.

    Returns:
      models: Capability per base model id.

    """
    return MappingProxyType(
        {
            "qwen3.6-max-preview": ModelCapability(
                model_id="qwen3.6-max-preview",
                context=MappingProxyType(
                    {
                        "": ModelLimits(
                            max_request_tokens=262_144,
                            max_response_tokens=65_536,
                        ),
                    }
                ),
                prices=_prices(request=1.6, response=6.4),
                thinking_effort=_efforts(),
                thinking_budget=frozenset({"none", "auto", "fixed"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "qwen3.6-plus": ModelCapability(
                model_id="qwen3.6-plus",
                context=MappingProxyType(
                    {
                        "": ModelLimits(
                            max_request_tokens=1_000_000,
                            max_response_tokens=65_536,
                        ),
                    }
                ),
                prices=_prices(request=0.5, response=3.0),
                thinking_effort=_efforts(),
                thinking_budget=frozenset({"none", "auto", "fixed"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "qwen3.6-flash": ModelCapability(
                model_id="qwen3.6-flash",
                context=MappingProxyType(
                    {
                        "": ModelLimits(
                            max_request_tokens=1_000_000,
                            max_response_tokens=65_536,
                        ),
                    }
                ),
                prices=_prices(request=0.05, response=0.2),
                thinking_effort=_efforts(),
                thinking_budget=frozenset({"none", "auto", "fixed"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "qwen3-235b-a22b-instruct-2507": ModelCapability(
                model_id="qwen3-235b-a22b-instruct-2507",
                context=MappingProxyType(
                    {
                        "": ModelLimits(
                            max_request_tokens=262_144,
                            max_response_tokens=65_536,
                        ),
                    }
                ),
                prices=_prices(request=0.7, response=2.8),
                # ``-instruct`` / ``-coder`` / ``-turbo`` reject the toggle, so
                # every thinking axis offers only its off value.
            ),
            "qwen3-235b-a22b-thinking-2507": ModelCapability(
                model_id="qwen3-235b-a22b-thinking-2507",
                context=MappingProxyType(
                    {
                        "": ModelLimits(
                            max_request_tokens=262_144,
                            max_response_tokens=32_768,
                        ),
                    }
                ),
                prices=_prices(request=0.7, response=8.4),
                thinking_effort=_thinking_only_efforts(),
                thinking_budget=frozenset({"none", "auto", "fixed"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "qwen3-30b-a3b-instruct-2507": ModelCapability(
                model_id="qwen3-30b-a3b-instruct-2507",
                context=MappingProxyType(
                    {
                        "": ModelLimits(
                            max_request_tokens=262_144,
                            max_response_tokens=65_536,
                        ),
                    }
                ),
                prices=_prices(request=0.2, response=0.8),
                # ``-instruct`` / ``-coder`` / ``-turbo`` reject the toggle, so
                # every thinking axis offers only its off value.
            ),
            "qwen3-32b": ModelCapability(
                model_id="qwen3-32b",
                context=MappingProxyType(
                    {
                        "": ModelLimits(
                            max_request_tokens=262_144,
                            max_response_tokens=65_536,
                        ),
                    }
                ),
                prices=_prices(request=0.4, response=1.2),
                thinking_effort=_efforts(),
                thinking_budget=frozenset({"none", "auto", "fixed"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "qwen3-coder-480b-a35b-instruct": ModelCapability(
                model_id="qwen3-coder-480b-a35b-instruct",
                context=MappingProxyType(
                    {
                        "": ModelLimits(
                            max_request_tokens=262_144,
                            max_response_tokens=65_536,
                        ),
                    }
                ),
                prices=_prices(request=1.0, response=5.0),
                # ``-instruct`` / ``-coder`` / ``-turbo`` reject the toggle, so
                # every thinking axis offers only its off value.
            ),
            "qwen-plus": ModelCapability(
                model_id="qwen-plus",
                context=MappingProxyType(
                    {
                        "": ModelLimits(
                            max_request_tokens=1_000_000,
                            max_response_tokens=32_768,
                        ),
                    }
                ),
                prices=_prices(request=0.4, response=1.2),
                thinking_effort=_efforts(),
                thinking_budget=frozenset({"none", "auto", "fixed"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "qwen-max": ModelCapability(
                model_id="qwen-max",
                context=MappingProxyType(
                    {
                        "": ModelLimits(
                            max_request_tokens=262_144,
                            max_response_tokens=65_536,
                        ),
                    }
                ),
                prices=_prices(request=1.6, response=6.4),
                thinking_effort=_efforts(),
                thinking_budget=frozenset({"none", "auto", "fixed"}),
                thinking_output=frozenset({"none", "text"}),
            ),
            "qwen-turbo": ModelCapability(
                model_id="qwen-turbo",
                context=MappingProxyType(
                    {
                        "": ModelLimits(
                            max_request_tokens=1_000_000,
                            max_response_tokens=32_768,
                        ),
                    }
                ),
                prices=_prices(request=0.05, response=0.2),
                # ``-instruct`` / ``-coder`` / ``-turbo`` reject the toggle, so
                # every thinking axis offers only its off value.
            ),
        }
    )


def _prices(*, request: float, response: float) -> PriceCatalog:
    """USD per million tokens; DashScope quotes one flat tier."""
    return PriceCatalog(
        {PriceCatalogProduct(): TokenPrice(request=request, response=response)}
    )


def _efforts() -> frozenset[ThinkingEffort]:
    """Every effort DashScope accepts."""
    return frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"})


def _thinking_only_efforts() -> frozenset[ThinkingEffort]:
    """Everything but ``none``: those ids reject ``enable_thinking=false``."""
    return _efforts() - {"none"}
