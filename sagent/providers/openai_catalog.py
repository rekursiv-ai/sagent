"""OpenAI model catalog, expressed as ``ModelCapability`` rows.

Sources:
  - Limits: https://developers.openai.com/api/docs/models/<model>
  - Pricing: https://developers.openai.com/api/docs/pricing
  - Images: https://developers.openai.com/api/docs/guides/images-vision

``MODELS`` is the API-key view. Other transports narrow it with ``&``
(see ``CHAT``, ``SUBSCRIPTION``), which can only remove.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Final

from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)
from sagent.types.model import (
    Limits,
    ModelCapability,
    ThinkingEffort,
)


__all__ = ["API", "CHAT_MODELS", "MODELS", "SUBSCRIPTION"]


# Prompts above 272K input tokens bill at 2x input / 1.5x output for the whole
# request on gpt-5.4 and later. The untagged id caps the window here; a ``+1m``
# id opts into the full window and thus into the surcharge.
_TWO_TIER: Final = 272_000


def _limits(*, request: int, response: int, gpt56_images: bool = False) -> Limits:
    """Conservative pre-GPT-5.6 compatibility caps.

    ``detail:high`` images are first scaled to fit a 2048x2048 square
    before tiling, so pre-resizing there caps wire bytes without losing
    fidelity. GPT-5.6 preserves source dimensions and publishes a 512 MB
    total request cap with no per-image cap.
    """
    if gpt56_images:
        return Limits(
            max_request_tokens=request,
            max_response_tokens=response,
            max_request_bytes=512 * 1024 * 1024,
        )
    return Limits(
        max_request_tokens=request,
        max_response_tokens=response,
        max_request_bytes=20 * 1024 * 1024,
        max_image_edge_px=2048,
        max_image_bytes=20 * 1024 * 1024,
    )


def _windowed(
    *, request: int, response: int, long: int, gpt56_images: bool = False
) -> Mapping[str, Limits]:
    """Both context tags; ``+1m`` opts into the full window and its surcharge."""
    return MappingProxyType(
        {
            "": _limits(request=request, response=response, gpt56_images=gpt56_images),
            "+1m": _limits(request=long, response=response, gpt56_images=gpt56_images),
        }
    )


def _prices(
    *,
    request: float,
    response: float,
    cache_write: float = 0.0,
    cache_read: float = 0.0,
    two_tier: bool = False,
) -> PriceCatalog:
    rows = {
        PriceCatalogProduct(): TokenPrice(
            request=request,
            response=response,
            cache_write=cache_write,
            cache_read=cache_read,
        )
    }
    if two_tier:
        # The >272K surcharge applies the input multiplier to all three input
        # pools and the output multiplier to the whole response.
        rows[PriceCatalogProduct(min_request_tokens=_TWO_TIER)] = TokenPrice(
            request=request * 2.0,
            response=response * 1.5,
            cache_write=cache_write * 2.0,
            cache_read=cache_read * 2.0,
        )
    return PriceCatalog(rows)


def _efforts(pairs: Mapping[ThinkingEffort, str]) -> Mapping[ThinkingEffort, str]:
    """Sagent effort -> wire ``reasoning_effort``, per model."""
    return MappingProxyType(dict(pairs))


# The pre-5.6 reasoning wire: ``minimal`` absorbs ``off``, and ``high``
# absorbs both ``xhigh`` and ``max`` -- a silent downgrade the old code hid
# inside a mapping function, now visible as data.
_LEGACY_REASONING = _efforts(
    {
        "off": "minimal",
        "min": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
        "max": "high",
    }
)

# GPT-5.6 accepts ``none`` and ``xhigh`` natively. Only ``max`` differs
# between transports, so the Chat transport narrows it (see ``CHAT``).
_GPT56_REASONING = _efforts(
    {
        "off": "none",
        "min": "none",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "max",
    }
)


# A row carries only what the MODEL can do; caching, retry, and auth mode are
# transport facts declared on ``API`` / ``SUBSCRIPTION``, since ``&`` can only
# remove. Every reasoning model takes an auto budget and returns readable text.
MODELS: Mapping[str, ModelCapability] = MappingProxyType(
    {
        "gpt-5.6-sol": ModelCapability(
            model_id="gpt-5.6-sol",
            context_limits=_windowed(
                request=272_000, response=128_000, gpt56_images=True, long=1_050_000
            ),
            prices=_prices(
                request=5.0,
                response=30.0,
                cache_write=6.25,
                cache_read=0.5,
                two_tier=True,
            ),
            supported_thinking_efforts=_GPT56_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gpt-5.6": ModelCapability(
            model_id="gpt-5.6",
            context_limits=_windowed(
                request=272_000, response=128_000, gpt56_images=True, long=1_050_000
            ),
            prices=_prices(
                request=5.0,
                response=30.0,
                cache_write=6.25,
                cache_read=0.5,
                two_tier=True,
            ),
            supported_thinking_efforts=_GPT56_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gpt-5.6-luna": ModelCapability(
            model_id="gpt-5.6-luna",
            context_limits=_windowed(
                request=272_000, response=128_000, gpt56_images=True, long=1_050_000
            ),
            prices=_prices(
                request=1.0,
                response=6.0,
                cache_write=1.25,
                cache_read=0.1,
                two_tier=True,
            ),
            supported_thinking_efforts=_GPT56_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gpt-5.6-terra": ModelCapability(
            model_id="gpt-5.6-terra",
            context_limits=_windowed(
                request=272_000, response=128_000, gpt56_images=True, long=1_050_000
            ),
            prices=_prices(
                request=2.5,
                response=15.0,
                cache_write=3.125,
                cache_read=0.25,
                two_tier=True,
            ),
            supported_thinking_efforts=_GPT56_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gpt-5.5": ModelCapability(
            model_id="gpt-5.5",
            context_limits=_windowed(request=272_000, response=128_000, long=1_000_000),
            prices=_prices(request=5.0, response=30.0, cache_read=0.5, two_tier=True),
            supported_thinking_efforts=_LEGACY_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gpt-5.5-pro": ModelCapability(
            model_id="gpt-5.5-pro",
            context_limits=_windowed(request=272_000, response=128_000, long=1_050_000),
            prices=_prices(request=30.0, response=180.0, two_tier=True),
            supported_thinking_efforts=_LEGACY_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gpt-5.4": ModelCapability(
            model_id="gpt-5.4",
            context_limits=_windowed(request=272_000, response=128_000, long=1_050_000),
            prices=_prices(request=2.5, response=15.0, cache_read=0.25, two_tier=True),
            supported_thinking_efforts=_LEGACY_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gpt-5.4-pro": ModelCapability(
            model_id="gpt-5.4-pro",
            context_limits=_windowed(request=272_000, response=128_000, long=1_050_000),
            prices=_prices(request=30.0, response=180.0, two_tier=True),
            supported_thinking_efforts=_LEGACY_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gpt-5.4-mini": ModelCapability(
            model_id="gpt-5.4-mini",
            context_limits=_limits(request=400_000, response=128_000),
            prices=_prices(request=0.75, response=4.5, cache_read=0.075),
            supported_thinking_efforts=_LEGACY_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gpt-5.4-nano": ModelCapability(
            model_id="gpt-5.4-nano",
            context_limits=_limits(request=400_000, response=128_000),
            prices=_prices(request=0.2, response=1.25, cache_read=0.02),
            supported_thinking_efforts=_LEGACY_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gpt-5.3-codex": ModelCapability(
            model_id="gpt-5.3-codex",
            context_limits=_limits(request=400_000, response=128_000),
            prices=_prices(request=1.75, response=14.0, cache_read=0.175),
            supported_thinking_efforts=_LEGACY_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        # Research preview: 128k context / 32k output, NOT the 400k/128k the
        # sibling gpt-5.3-codex carries.
        # https://openai.com/index/introducing-gpt-5-3-codex-spark/
        "gpt-5.3-codex-spark": ModelCapability(
            model_id="gpt-5.3-codex-spark",
            context_limits=_limits(request=128_000, response=32_768),
            prices=_prices(request=1.75, response=14.0, cache_read=0.175),
            supported_thinking_efforts=_LEGACY_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gpt-5.3-chat-latest": ModelCapability(
            model_id="gpt-5.3-chat-latest",
            context_limits=_limits(request=128_000, response=16_384),
            prices=_prices(request=1.75, response=14.0, cache_read=0.175),
            supported_thinking_efforts=_LEGACY_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gpt-5.2": ModelCapability(
            model_id="gpt-5.2",
            context_limits=_limits(request=400_000, response=128_000),
            prices=_prices(request=1.75, response=14.0, cache_read=0.175),
            supported_thinking_efforts=_LEGACY_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "o1": ModelCapability(
            model_id="o1",
            context_limits=_limits(request=200_000, response=100_000),
            prices=_prices(request=15.0, response=60.0, cache_read=7.5),
            supported_thinking_efforts=_LEGACY_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "o1-mini": ModelCapability(
            model_id="o1-mini",
            context_limits=_limits(request=128_000, response=65_536),
            prices=_prices(request=3.0, response=12.0, cache_read=1.5),
            supported_thinking_efforts=_LEGACY_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "o3-mini": ModelCapability(
            model_id="o3-mini",
            context_limits=_limits(request=200_000, response=100_000),
            prices=_prices(request=1.1, response=4.4, cache_read=0.55),
            supported_thinking_efforts=_LEGACY_REASONING,
            supported_thinking_budgets=frozenset({"auto"}),
            supported_thinking_outputs=frozenset({"text"}),
        ),
        "gpt-4.1": ModelCapability(
            model_id="gpt-4.1",
            context_limits=_windowed(
                request=1_047_576, response=32_768, long=1_047_576
            ),
            prices=_prices(request=2.0, response=8.0, cache_read=0.5),
            # No reasoning knob on this generation.
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset(),
            supported_thinking_outputs=frozenset(),
        ),
        "gpt-4.1-mini": ModelCapability(
            model_id="gpt-4.1-mini",
            context_limits=_windowed(
                request=1_047_576, response=32_768, long=1_047_576
            ),
            prices=_prices(request=0.4, response=1.6, cache_read=0.1),
            # No reasoning knob on this generation.
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset(),
            supported_thinking_outputs=frozenset(),
        ),
        "gpt-4.1-nano": ModelCapability(
            model_id="gpt-4.1-nano",
            context_limits=_windowed(
                request=1_047_576, response=32_768, long=1_047_576
            ),
            prices=_prices(request=0.1, response=0.4, cache_read=0.025),
            # No reasoning knob on this generation.
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset(),
            supported_thinking_outputs=frozenset(),
        ),
        "gpt-4o": ModelCapability(
            model_id="gpt-4o",
            context_limits=_limits(request=128_000, response=16_384),
            prices=_prices(request=2.5, response=10.0, cache_read=1.25),
            # No reasoning knob on this generation.
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset(),
            supported_thinking_outputs=frozenset(),
        ),
        "gpt-4o-mini": ModelCapability(
            model_id="gpt-4o-mini",
            context_limits=_limits(request=128_000, response=16_384),
            prices=_prices(request=0.15, response=0.6, cache_read=0.075),
            # No reasoning knob on this generation.
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset(),
            supported_thinking_outputs=frozenset(),
        ),
        "gpt-4-turbo": ModelCapability(
            model_id="gpt-4-turbo",
            context_limits=_limits(request=128_000, response=4_096),
            prices=_prices(request=10.0, response=30.0),
            # No reasoning knob on this generation.
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset(),
            supported_thinking_outputs=frozenset(),
        ),
        "gpt-4": ModelCapability(
            model_id="gpt-4",
            context_limits=_limits(request=8_192, response=8_192),
            prices=_prices(request=30.0, response=60.0),
            # No reasoning knob on this generation.
            supported_thinking_efforts=MappingProxyType({}),
            supported_thinking_budgets=frozenset(),
            supported_thinking_outputs=frozenset(),
        ),
    }
)


# Chat Completions cannot express GPT-5.6's ``max``; it downgrades to
# ``xhigh``. ``&`` filters KEYS, so a value remap is a separate derivation.
CHAT_MODELS: Mapping[str, ModelCapability] = MappingProxyType(
    {
        name: replace(
            cap,
            supported_thinking_efforts=MappingProxyType(
                {
                    effort: ("xhigh" if wire == "max" else wire)
                    for effort, wire in cap.supported_thinking_efforts.items()
                }
            ),
        )
        for name, cap in MODELS.items()
    }
)

# The REST API exposes no prompt-cache breakpoint and no server-side context
# management, and does not retry internally. Key auth, not account auth.
API: Final = ModelCapability(
    manages_context=False,
    service_tiers=frozenset({"auto", "default", "flex", "priority"}),
    prompt_cache_breakpoints=False,
    retries_internally=False,
    account_auth=False,
)

# Codex subscription: account auth, and ``/fast`` maps to the priority tier.
SUBSCRIPTION: Final = ModelCapability(
    manages_context=False,
    service_tiers=frozenset({"priority"}),
    prompt_cache_breakpoints=False,
    retries_internally=False,
)
"""Codex ``/fast`` selects the priority tier, so the fast path stays open."""
