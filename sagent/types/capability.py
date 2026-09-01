"""What one model accepts: its ceilings, its prices, and its thinking modes.

The catalog rows every provider declares, and the ceilings a transport
narrows them to. Separate from the ``Model`` protocol that calls a model:
these are data a catalog states, so a reader that only wants to know what a
model costs need not import the machinery that runs one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Final, Self

from sagent.types.cost import PriceCatalog
from sagent.types.thinking import (
    ALL_THINKING_EFFORTS,
    ThinkingBudget,
    ThinkingEffort,
    ThinkingOutput,
)


__all__ = [
    "ModelCapability",
    "ModelLimits",
]


_THINKING_STATES: Final = (
    "adaptive-show",
    "adaptive-hide",
    "on-show",
    "on-hide",
    "off-hide",
    "redact-hide",
)
"""Canonical thinking states, in the order the UI presents them."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelLimits:
    """Hard ceilings for one context configuration of one model."""

    max_request_tokens: int = 0
    """Input tokens the model accepts; ``0`` means unknown."""

    max_response_tokens: int = 0
    """Output tokens the model can generate in one response."""

    max_request_bytes: int = 0
    """HTTP wire ceiling, distinct from the token window. ``0`` = none."""

    max_image_edge_px: int = 0
    """Long edge above which the server downscales. ``0`` = no resize."""

    max_image_bytes: int = 0
    """Per-image byte cap after resize. ``0`` = no cap."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCapability:
    """One catalog row, or one transport's restrictions.

    ``context_limits`` is a mapping when the model offers several
    context configurations (``""``, ``"+1m"``), and a bare ``ModelLimits``
    once narrowed to one.
    """

    model_id: str = ""
    """The id the vendor accepts on the wire, without option tags."""

    context_limits: Mapping[str, ModelLimits] | ModelLimits = field(
        default_factory=ModelLimits
    )
    """ModelLimits per context tag, or one ``ModelLimits`` once narrowed."""

    prices: PriceCatalog = field(default_factory=PriceCatalog)
    """USD rates, keyed by latency tier and request-size threshold."""

    chars_per_token: float = 4.0
    """Divisor for the local token estimate; measured per tokenizer."""

    supported_thinking_efforts: Mapping[ThinkingEffort, str] = ALL_THINKING_EFFORTS
    """Offered effort levels, each mapped to its wire value."""

    supported_thinking_budgets: frozenset[ThinkingBudget] = frozenset(
        {"auto", "fixed"},
    )
    """Whether the caller may leave the reasoning budget open, fix it, or both."""

    supported_thinking_outputs: frozenset[ThinkingOutput] = frozenset(
        {"text", "redacted"}
    )
    """Whether reasoning comes back readable, redacted, or either."""

    fast: bool = True
    """Whether this transport exposes the fast tier at all."""

    manages_context: bool = True
    """Whether the server rolls history under quota pressure."""

    prompt_cache_breakpoints: bool = True
    """Whether the caller may place explicit prompt-cache breakpoints."""

    retries_internally: bool = True
    """Whether the transport retries transient failures on its own."""

    account_auth: bool = True
    """Whether this transport bills an account rather than an API key."""

    latency_modes: frozenset[str] = frozenset({"fast"})
    """Latency hints this transport accepts; ``fast`` is the only one defined."""

    service_tiers: frozenset[str] = frozenset(
        {"auto", "default", "flex", "priority", "standard_only"}
    )
    """Vendor service-tier values; the transport narrows to what it accepts."""

    @property
    def serves_fast(self) -> bool:
        """Whether a fast-tier price row is reachable."""
        return self.fast and any(k.fast for k in self.prices)

    @property
    def valid_thinking_states(self) -> tuple[str, ...]:
        """Canonical thinking states reachable on this model.

        The three thinking axes already determine the state set:
        ``auto`` is the ``adaptive-*`` states, ``fixed`` the ``on-*``
        ones, ``text`` gates every ``-show``, and ``redacted`` gates
        ``redact-hide`` (which rides ``adaptive``). ``off-hide`` is
        always reachable.
        """
        budgets = self.supported_thinking_budgets
        outputs = self.supported_thinking_outputs
        if not budgets:
            return ("off-hide",)
        states: list[str] = []
        for state in _THINKING_STATES:
            if state == "off-hide":
                states.append(state)
            elif state == "redact-hide":
                if "redacted" in outputs and "auto" in budgets:
                    states.append(state)
            elif (
                (state.startswith("adaptive-") and "auto" not in budgets)
                or (state.startswith("on-") and "fixed" not in budgets)
                or (state.endswith("-show") and "text" not in outputs)
            ):
                continue
            else:
                states.append(state)
        return tuple(states)

    def __and__(self, other: ModelCapability) -> Self:
        # ``replace`` keeps the concrete class: meeting a narrowed
        # ``ModelSpec`` must not silently downgrade it and drop the
        # ``context`` / ``serve_fast`` tags it carries.
        return replace(
            self,
            context_limits=self.context_limits,
            chars_per_token=self.chars_per_token,
            prices=(
                self.prices
                if other.fast
                else PriceCatalog({k: v for k, v in self.prices.items() if not k.fast})
            ),
            supported_thinking_efforts=MappingProxyType(
                {
                    k: v
                    for k, v in self.supported_thinking_efforts.items()
                    if k in other.supported_thinking_efforts
                }
            ),
            supported_thinking_budgets=(
                self.supported_thinking_budgets & other.supported_thinking_budgets
            ),
            supported_thinking_outputs=(
                self.supported_thinking_outputs & other.supported_thinking_outputs
            ),
            fast=self.fast and other.fast,
            manages_context=self.manages_context and other.manages_context,
            prompt_cache_breakpoints=(
                self.prompt_cache_breakpoints and other.prompt_cache_breakpoints
            ),
            retries_internally=self.retries_internally and other.retries_internally,
            account_auth=self.account_auth and other.account_auth,
            latency_modes=self.latency_modes & other.latency_modes,
            service_tiers=self.service_tiers & other.service_tiers,
        )
