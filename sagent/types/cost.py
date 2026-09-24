"""Token count, price, and cost calculus.

Three parallel shapes over the same five token meters: ``TokenCount``
(integers), ``TokenPrice`` (USD per ``tokens_per_unit``), and
``TokenCost`` (USD). ``TokenPrice * TokenCount -> TokenCost``; the
illegal products are absent methods rather than runtime checks.

``PriceCatalog`` maps a ``PriceKey`` -- service tier, prompt-size band, and
effective date -- to the ``TokenPrice`` a vendor publishes for it, and
resolves one request to exactly one of them. There is no fallback: a tier
or date nothing prices raises rather than billing some other rate.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import date
from typing import Literal, NamedTuple, Self, override


__all__ = [
    "PriceCatalog",
    "PriceKey",
    "ServiceTier",
    "TokenCost",
    "TokenCount",
    "TokenPrice",
    "TokenStats",
]


type ServiceTier = Literal["auto", "default", "flex", "priority"]


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
        # Non-``TokenStats`` operands defer rather than raising
        # ``AttributeError`` mid-expression: callers reaching through
        # ``object``-typed plumbing (status pane, persisted-metadata
        # round-trip) can still pass an int or str.
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
        # ``CostTracker.restore_totals`` can move the cumulative total BELOW
        # a pre-restore snapshot, and the status pane reads
        # ``current - snapshot``. A negative token COUNT is not a thing; a
        # negative cost delta is, so only counts clamp.
        return max(0, value)


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenCost(TokenStats[float]):
    """USD spent, by meter."""

    request: float = 0.0
    response: float = 0.0
    cache_write: float = 0.0
    cache_write_1h: float = 0.0
    cache_read: float = 0.0


# No defaults on the rates: a meter left unstated would bill $0, which is how
# an unpriced cache pool went free. A vendor that charges nothing says ``0.0``.
@dataclass(frozen=True, slots=True, kw_only=True)
class TokenPrice(TokenStats[float]):
    """USD per ``tokens_per_unit`` tokens, by meter, as the vendor publishes it."""

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


class PriceKey(NamedTuple):
    """The conditions under which one ``TokenPrice`` applies."""

    service_tier: ServiceTier
    """Tier that was served; ``auto`` is the vendor's standard rate."""

    effective_prompt_tokens: int = 0
    """Applies to prompts LARGER than this; the lowest band covers every size."""

    effective_date: date = date.min
    """Applies to requests on or after this date."""


class PriceCatalog(Mapping[PriceKey, TokenPrice]):
    """A model's published price table.

    Indexing is exact, like any mapping; :meth:`rate` answers "which price
    did this request pay", and raises rather than guess when nothing does.
    """

    def __init__(self, prices: Mapping[PriceKey, TokenPrice] | None = None) -> None:
        """Build the catalog.

        Args:
          prices: Every published (tier, band, date) and its rates.

        """
        self._prices: dict[PriceKey, TokenPrice] = dict(sorted((prices or {}).items()))

    @property
    def service_tiers(self) -> frozenset[ServiceTier]:
        """Tiers this table prices, plus ``default`` wherever ``auto`` is priced.

        ``default`` is the explicit request for the standard tier, so it
        bills whatever ``auto`` bills.
        """
        tiers: set[ServiceTier] = {key.service_tier for key in self._prices}
        if "auto" in tiers:
            tiers.add("default")
        return frozenset(tiers)

    def rate(
        self,
        *,
        service_tier: ServiceTier,
        prompt_tokens: int,
        at: date,
    ) -> TokenPrice:
        """Return the price one request paid.

        Args:
          service_tier: Tier the server reports it served.
          prompt_tokens: The request's whole prompt, every cache pool included.
          at: Day the request was served.

        Returns:
          price: The latest-effective rate of the prompt's size band.

        Raises:
          KeyError: Nothing prices this tier on this date.

        """
        tier: ServiceTier = "auto" if service_tier == "default" else service_tier
        # A key sorts by (tier, band, date), so the last match is the highest
        # band the prompt exceeds, at its latest effective date. The base
        # band (0) matches every prompt, an empty one included.
        matches = [
            key
            for key in self._prices
            if key.service_tier == tier
            and key.effective_date <= at
            and (
                key.effective_prompt_tokens < prompt_tokens
                or key.effective_prompt_tokens == 0
            )
        ]
        if not matches:
            raise KeyError(PriceKey(tier, prompt_tokens, at))
        return self._prices[matches[-1]]

    def cost(
        self,
        tokens: TokenCount,
        *,
        service_tier: ServiceTier,
        at: date,
    ) -> TokenCost:
        """Price ONE request's ``tokens`` at the band its whole prompt selects.

        Args:
          tokens: One request's usage. Summing requests first would size the
              band from the session rather than the request.
          service_tier: Tier the server reports it served.
          at: Day the request was served.

        Returns:
          cost: USD, by meter.

        """
        return (
            self.rate(service_tier=service_tier, prompt_tokens=tokens.prompt, at=at)
            * tokens
        )

    @override
    def __getitem__(self, key: PriceKey) -> TokenPrice:
        return self._prices[key]

    @override
    def __iter__(self) -> Iterator[PriceKey]:
        return iter(self._prices)

    @override
    def __len__(self) -> int:
        return len(self._prices)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._prices!r})"
