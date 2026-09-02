"""DashScope/Qwen model catalog, expressed as ``ModelCapability`` rows.

Source: https://help.aliyun.com/zh/model-studio/models

A row carries only what the MODEL can do; caching, retry, and auth mode
are transport facts declared on ``OpenAICompat.TRANSPORT``. These vendors
publish no image pixel or byte ceiling and preprocess images server-side,
so ``ModelLimits`` carries only the two token windows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

from sagent.types.capability import (
    ContextTag,
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
    # A reasoning Qwen row: every model below is this one with its window,
    # price, and thinking axes replaced.
    qwen = ModelCapability(
        context=_limits(window=262_144, response=65_536),
        prices=_prices(request=1.6, response=6.4),
        thinking_effort=_efforts(),
        thinking_budget={"none", "auto", "fixed"},
        thinking_output={"none", "text"},
    )
    # ``-instruct`` / ``-coder`` / ``-turbo`` reject the toggle, so every
    # thinking axis offers only its off value.
    instruct = replace(
        qwen,
        thinking_effort={"none"},
        thinking_budget={"none"},
        thinking_output={"none"},
    )
    rows = (
        replace(qwen, model_id="qwen3.6-max-preview"),
        replace(
            qwen,
            model_id="qwen3.6-plus",
            context=_limits(window=1_000_000, response=65_536),
            prices=_prices(request=0.5, response=3.0),
        ),
        replace(
            qwen,
            model_id="qwen3.6-flash",
            context=_limits(window=1_000_000, response=65_536),
            prices=_prices(request=0.05, response=0.2),
        ),
        replace(
            instruct,
            model_id="qwen3-235b-a22b-instruct-2507",
            prices=_prices(request=0.7, response=2.8),
        ),
        replace(
            qwen,
            model_id="qwen3-235b-a22b-thinking-2507",
            context=_limits(window=262_144, response=32_768),
            prices=_prices(request=0.7, response=8.4),
            # Everything but ``none``: these ids reject ``enable_thinking=false``.
            thinking_effort=_efforts() - {"none"},
        ),
        replace(
            instruct,
            model_id="qwen3-30b-a3b-instruct-2507",
            prices=_prices(request=0.2, response=0.8),
        ),
        replace(
            qwen,
            model_id="qwen3-32b",
            prices=_prices(request=0.4, response=1.2),
        ),
        replace(
            instruct,
            model_id="qwen3-coder-480b-a35b-instruct",
            prices=_prices(request=1.0, response=5.0),
        ),
        replace(
            qwen,
            model_id="qwen-plus",
            context=_limits(window=1_000_000, response=32_768),
            prices=_prices(request=0.4, response=1.2),
        ),
        replace(
            qwen,
            model_id="qwen-max",
            prices=_prices(request=1.6, response=6.4),
        ),
        replace(
            instruct,
            model_id="qwen-turbo",
            context=_limits(window=1_000_000, response=32_768),
            prices=_prices(request=0.05, response=0.2),
        ),
    )
    return MappingProxyType({row.model_id: row for row in rows})


def _limits(*, window: int, response: int) -> Mapping[ContextTag, ModelLimits]:
    """One context tag: these vendors ship no window variants."""
    return MappingProxyType(
        {
            "": ModelLimits(
                max_request_tokens=window,
                max_response_tokens=response,
            )
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
