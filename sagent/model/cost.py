"""Token counts, prices and the context sizes they name, and what a request cost.

tokens = TokenCount(request=1_000, response=200, cache_read=5_000)
prices = model.capability.prices
prices["auto", "smallcontext"]              # the size ``smallcontext`` names
cost = prices["auto", tokens.prompt, today] * tokens
cost.total                              # USD
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Literal, Self, overload, override

import math

from sagent.model.context import ContextSize, ContextSizeName


__all__ = [
    "PriceCatalog",
    "PriceCondition",
    "ServiceTier",
    "TokenCost",
    "TokenCount",
    "TokenPrice",
    "TokenStats",
]


type ServiceTier = Literal[
    "auto",
    "default",
    "flex",
    "priority",
    "ultrafast",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenStats[T: (int, float)]:
    """Five token meters, shared by counts, prices, and costs.

    The four input meters are disjoint: a prompt token is exactly one of
    uncached, cache-written (either lifetime), or cache-read.
    """

    request: T
    """Uncached prompt tokens."""

    response: T
    """Generated output tokens, reasoning included."""

    cache_write: T
    """Prompt tokens written to cache at the vendor's default lifetime."""

    cache_write_1h: T
    """Prompt tokens written to cache at a one-hour lifetime."""

    cache_read: T
    """Prompt tokens served from cache."""

    @property
    def total(self) -> T:
        """Sum across all five meters."""
        return (
            self.request
            + self.response
            + self.cache_write
            + self.cache_write_1h
            + self.cache_read
        )

    def __add__(self, other: Self) -> Self:
        """Add two token stats, handling non-matching types gracefully."""
        # An operand of another kind defers, so Python raises the usual
        # ``TypeError`` rather than an ``AttributeError`` from inside.
        if not isinstance(other, type(self)):
            return NotImplemented
        return replace(
            self,
            request=self.request + other.request,
            response=self.response + other.response,
            cache_write=self.cache_write + other.cache_write,
            cache_write_1h=self.cache_write_1h + other.cache_write_1h,
            cache_read=self.cache_read + other.cache_read,
        )

    def __sub__(self, other: Self) -> Self:
        """Subtract two token stats, handling non-matching types gracefully."""
        # See ``__add__`` for the deferral.
        if not isinstance(other, type(self)):
            return NotImplemented
        return replace(
            self,
            request=self._floor(self.request - other.request),
            response=self._floor(self.response - other.response),
            cache_write=self._floor(self.cache_write - other.cache_write),
            cache_write_1h=self._floor(self.cache_write_1h - other.cache_write_1h),
            cache_read=self._floor(self.cache_read - other.cache_read),
        )

    @classmethod
    def _floor(cls, value: T) -> T:
        """Lower bound on a subtraction result; unbounded by default."""
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenCount(TokenStats[int]):
    """Tokens billed, by meter.

    ``request`` excludes both cache pools. A provider whose API reports a
    cache-inclusive total (OpenAI, Google) subtracts the cached portion at
    construction.
    """

    request: int = 0
    response: int = 0
    cache_write: int = 0
    cache_write_1h: int = 0
    cache_read: int = 0

    @property
    def prompt(self) -> int:
        """Every prompt token the server counted, cached or not."""
        return self.request + self.cache_write + self.cache_write_1h + self.cache_read

    @override
    @classmethod
    def _floor(cls, value: int) -> int:
        # A count is never negative, so a difference that would be, such as a
        # total read before a resume restored a smaller one, clamps to zero. A
        # cost delta may be negative, so only counts clamp.
        return max(0, value)


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenCost(TokenStats[float]):
    """USD spent, by meter."""

    request: float = 0.0
    response: float = 0.0
    cache_write: float = 0.0
    cache_write_1h: float = 0.0
    cache_read: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenPrice(TokenStats[float]):
    """USD per ``tokens_per_unit`` tokens, by meter, as the vendor publishes it."""

    request: float = 0.0
    response: float = 0.0
    cache_write: float = 0.0
    cache_write_1h: float = 0.0
    cache_read: float = 0.0
    tokens_per_unit: int = 1_000_000
    """Tokens each rate is quoted per; vendors publish per million."""

    def __mul__(self, tokens: TokenCount) -> TokenCost:
        """Multiply price by token count to get cost."""
        unit = self.tokens_per_unit
        return TokenCost(
            request=self.request * tokens.request / unit,
            response=self.response * tokens.response / unit,
            cache_write=self.cache_write * tokens.cache_write / unit,
            cache_write_1h=self.cache_write_1h * tokens.cache_write_1h / unit,
            cache_read=self.cache_read * tokens.cache_read / unit,
        )

    __rmul__ = __mul__


@dataclass(frozen=True, slots=True, kw_only=True)
class PriceCondition:
    """When one price applies."""

    service_tier: ServiceTier = "auto"
    token_size_range: tuple[int, int | float] = (0, math.inf)
    """``[floor, ceil)``: prompts of at least ``floor`` tokens and fewer than ``ceil``."""

    effective_date: date = date.min


@dataclass(frozen=True, slots=True, kw_only=True)
class PriceCatalog:
    """A model's prices, and the context sizes they name.

    ``catalog[tier, size]`` resolves ``smallcontext``, ``midcontext``, ``fullcontext``, or a count
    to a :class:`ContextSize`; ``catalog[tier, tokens, at]`` is the price of a
    prompt of ``tokens`` on ``at``.
    """

    prices: Mapping[PriceCondition, TokenPrice] = field(
        default_factory=dict[PriceCondition, TokenPrice],
    )
    window: int = 0
    """The most the model takes in, which ``fullcontext`` names; ``0`` is unknown."""

    @override
    def __hash__(self) -> int:
        return hash((frozenset(self.prices.items()), self.window))

    @property
    def service_tiers(self) -> frozenset[ServiceTier]:
        """The tiers priced, and ``default`` wherever ``auto`` is: it bills as ``auto``."""
        tiers: set[ServiceTier] = {c.service_tier for c in self.prices}
        if "auto" in tiers:
            tiers.add("default")
        return frozenset(tiers)

    @overload
    def __getitem__(
        self,
        key: tuple[ServiceTier, ContextSizeName | int],
    ) -> ContextSize: ...

    @overload
    def __getitem__(
        self,
        key: tuple[ServiceTier, ContextSizeName | int, date],
    ) -> TokenPrice: ...

    def __getitem__(
        self,
        key: tuple[ServiceTier, ContextSizeName | int]
        | tuple[ServiceTier, ContextSizeName | int, date],
    ) -> ContextSize | TokenPrice:
        """Return the size ``key`` names at its tier, or a prompt's price on a date.

        Raises:
          KeyError: A size below 1 token, or no price in effect.

        """
        if len(key) == 2:
            size = _get_context_size(self.prices, key[0], key[1], window=self.window)
            if size is None:
                raise KeyError(key)
            return size
        tier, spec, at = key
        tokens = spec if isinstance(spec, int) else self[tier, spec].tokens
        price = _get_price(self.prices, tier, tokens, at=at)
        if price is None:
            raise KeyError(key)
        return price

    def sizes(self, tier: ServiceTier) -> tuple[ContextSize, ...]:
        """Return the sizes ``tier`` names, ascending: ``smallcontext``, ``midcontext``, ``fullcontext``.

        Args:
          tier: The service tier whose bands name them.

        Returns:
          sizes: Each band's largest prompt, then the window.

        """
        return tuple(_get_context_sizes(self.prices, tier, window=self.window).values())


def _get_context_sizes(
    prices: Mapping[PriceCondition, TokenPrice],
    tier: ServiceTier,
    *,
    window: int,
) -> dict[ContextSizeName, ContextSize]:
    """Return ``tier``'s named sizes: each band's largest prompt, then ``window``."""
    priced: ServiceTier = "auto" if tier == "default" else tier
    tops = sorted(
        {
            int(ceil) - 1
            for c in prices
            if c.service_tier == priced
            and not math.isinf(ceil := c.token_size_range[1])
            and (window <= 0 or ceil <= window)
        },
    )
    named: tuple[ContextSizeName, ...] = ("smallcontext", "midcontext")[: len(tops)]
    return {
        **{
            name: ContextSize(
                context_name=name,
                tokens=top,
                display_name=_tokens_to_display_name(top),
            )
            for name, top in zip(named, tops, strict=False)
        },
        "fullcontext": ContextSize(
            context_name="fullcontext",
            tokens=window,
            display_name=_tokens_to_display_name(window),
        ),
    }


def _get_context_size(
    prices: Mapping[PriceCondition, TokenPrice],
    tier: ServiceTier,
    spec: ContextSizeName | int,
    *,
    window: int,
) -> ContextSize | None:
    """Return the size ``spec`` names at ``tier``; ``None`` for a count below 1."""
    sizes = _get_context_sizes(prices, tier, window=window)
    if isinstance(spec, str):
        return sizes.get(spec, sizes["fullcontext"])
    if spec < 1:
        return None
    tokens = min(spec, window) if window > 0 else spec
    ascending: Sequence[ContextSize] = tuple(sizes.values())
    band = next((s for s in ascending if tokens <= s.tokens), sizes["fullcontext"])
    return ContextSize(
        context_name=band.context_name,
        tokens=tokens,
        display_name=_tokens_to_display_name(tokens),
    )


def _get_price(
    prices: Mapping[PriceCondition, TokenPrice],
    tier: ServiceTier,
    prompt_tokens: int,
    *,
    at: date,
) -> TokenPrice | None:
    """Return the price of ``prompt_tokens`` at ``tier`` on ``at``; ``None`` if none holds."""
    priced: ServiceTier = "auto" if tier == "default" else tier
    held = [
        c
        for c in prices
        if c.service_tier == priced
        and c.token_size_range[0] <= prompt_tokens < c.token_size_range[1]
        and c.effective_date <= at
    ]
    return prices[max(held, key=lambda c: c.effective_date)] if held else None


def _tokens_to_display_name(tokens: int) -> str:
    """Return ``tokens`` as a person reads it: ``1m``, ``258k``, ``258123``."""
    for unit, scale in (("m", 1_000_000), ("k", 1_000)):
        if tokens and tokens % scale == 0:
            return f"{tokens // scale}{unit}"
    return str(tokens)
