"""Typed YAML document schema for provider catalogs.

The reference row supplies defaults; an optional profile overrides it, then
model fields override the result. Nested objects replace whole objects. Context
presets are max/min; default_context selects the unqualified model, normally max.
Prices are explicit per model and require all five meters, including zeros.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from sagent.types.capability import ModelLimits, ThinkingCapability
from sagent.types.cost import ServiceTier, TokenPrice


type ProviderName = Literal[
    "anthropic",
    "openai",
    "google",
    "dashscope",
    "minimax",
    "moonshot",
    "llamacpp",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class CapabilityProfile:
    """Reusable model facts; no identity, prices, or profile inheritance."""

    knowledge_cutoff: str | None = None
    approx_chars_per_token: float = 4.0
    default_context: str = "max"
    context: dict[str, ModelLimits] = field(default_factory=dict)
    thinking: ThinkingCapability = field(default_factory=ThinkingCapability)
    effort_as_level: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class PriceRow(TokenPrice):
    """One explicit rate card and the conditions selecting it."""

    service_tier: ServiceTier = "auto"
    effective_prompt_tokens: int = 0
    effective_date: str | date = "0001-01-01"


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRow(CapabilityProfile):
    """Resolved model document after its profile has been applied."""

    model_id: str
    prices: tuple[PriceRow, ...]
    wire_model_id: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogDocument:
    """Provider document envelope; model overrides are validated after resolution."""

    models: tuple[dict[str, object], ...]
    reference: dict[str, object] = field(default_factory=dict)
    profiles: dict[str, dict[str, object]] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
