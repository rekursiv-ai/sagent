"""Token count, price, and cost calculus.

Three parallel shapes over the same four token buckets: ``TokenCount``
(integers), ``TokenPrice`` (USD per ``tokens_per_unit``), and
``TokenCost`` (USD). ``TokenPrice * TokenCount -> TokenCost``; the
illegal products are absent methods rather than runtime checks.

``PriceCatalog`` maps a ``PriceCatalogProduct`` -- the (fast, request
size) pair a vendor prices on -- to a ``TokenPrice``, resolving a
request size to the highest tier at or below it.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import NamedTuple, Self, override


__all__ = [
    "PriceCatalog",
    "PriceCatalogProduct",
    "TokenCost",
    "TokenCount",
    "TokenPrice",
    "TokenStats",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenStats[T: (int, float)]:
    """Four token buckets, shared by counts, prices, and costs.

    Defaults live on each concrete subclass: a dataclass default cannot
    be typed ``T`` here, since pyright rejects ``Literal[0]`` against an
    unbound TypeVar (microsoft/pyright#11226).
    """

    request: T
    """Non-cached prompt tokens; disjoint from the two cache pools."""

    response: T
    """Generated output tokens."""

    cache_write: T
    """Tokens spent creating prompt-cache entries."""

    cache_read: T
    """Tokens served from prompt cache."""

    @property
    def total(self) -> T:
        """Sum across all four buckets."""
        return self.request + self.response + self.cache_write + self.cache_read

    def __add__(self, other: Self) -> Self:
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
            cache_read=self.cache_read + other.cache_read,
        )

    def __sub__(self, other: Self) -> Self:
        # See ``__add__`` for the deferral.
        if not isinstance(other, type(self)):
            return NotImplemented
        return replace(
            self,
            request=self._floor(self.request - other.request),
            response=self._floor(self.response - other.response),
            cache_write=self._floor(self.cache_write - other.cache_write),
            cache_read=self._floor(self.cache_read - other.cache_read),
        )

    @classmethod
    def _floor(cls, value: T) -> T:
        """Lower bound on a subtraction result; unbounded by default."""
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenCount(TokenStats[int]):
    """Tokens billed, by bucket.

    ``request`` excludes ``cache_read`` and ``cache_write``: the three
    pools are disjoint, so the full prompt the server counted is their
    sum. A provider whose API reports a cache-inclusive total (OpenAI,
    Google) subtracts the cached portion at construction.
    """

    request: int = 0
    """Non-cached prompt tokens."""

    response: int = 0
    """Generated output tokens."""

    cache_write: int = 0
    """Tokens written to prompt cache."""

    cache_read: int = 0
    """Tokens served from prompt cache."""

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
    """USD spent, by bucket."""

    request: float = 0.0
    """USD spent on non-cached prompt tokens."""

    response: float = 0.0
    """USD spent on generated output tokens."""

    cache_write: float = 0.0
    """USD spent writing prompt-cache entries."""

    cache_read: float = 0.0
    """USD spent reading from prompt cache."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenPrice(TokenStats[float]):
    """USD per ``tokens_per_unit`` tokens, by bucket."""

    request: float = 0.0
    """USD per unit of non-cached prompt tokens."""

    response: float = 0.0
    """USD per unit of generated output tokens."""

    cache_write: float = 0.0
    """USD per unit of tokens written to prompt cache."""

    cache_read: float = 0.0
    """USD per unit of tokens served from prompt cache."""

    tokens_per_unit: int = 1_000_000
    """Tokens each rate is quoted per; vendors publish per million."""

    def __mul__(self, tokens: TokenCount) -> TokenCost:
        return TokenCost(
            request=self.request * tokens.request / self.tokens_per_unit,
            response=self.response * tokens.response / self.tokens_per_unit,
            cache_write=self.cache_write * tokens.cache_write / self.tokens_per_unit,
            cache_read=self.cache_read * tokens.cache_read / self.tokens_per_unit,
        )

    __rmul__ = __mul__


class PriceCatalogProduct(NamedTuple):
    """The billable product a vendor quotes a ``TokenPrice`` for."""

    fast: bool = False
    """Whether this row prices the fast tier."""

    min_request_tokens: int = 0
    """Prompt size at which this row takes over from the one below."""


class PriceCatalog(Mapping[PriceCatalogProduct, TokenPrice]):
    """Ordered price table with floor lookup on ``min_request_tokens``.

    A vendor's long-context surcharge is a step function: the price for
    a 300k-token request is the tier declared at or below 300k. A plain
    dict cannot answer that predecessor query, so keys are kept sorted
    and resolved with ``bisect_right``.

    A lookup that finds no tier raises ``KeyError`` rather than
    returning a zero price, so a missing row cannot silently bill $0.
    """

    def __init__(
        self,
        prices: Mapping[PriceCatalogProduct, TokenPrice] | None = None,
    ) -> None:
        """Build the catalog.

        Args:
          prices: Declared tiers. Order is irrelevant; keys are sorted.

        """
        self._prices: dict[PriceCatalogProduct, TokenPrice] = dict(
            sorted((prices or {}).items())
        )
        self._by_mode: dict[bool, list[PriceCatalogProduct]] = {}
        for key in self._prices:
            self._by_mode.setdefault(key.fast, []).append(key)

    @override
    def __getitem__(self, key: PriceCatalogProduct) -> TokenPrice:
        keys = self._by_mode.get(key.fast, [])
        i = bisect_right(keys, key)
        if i:
            return self._prices[keys[i - 1]]
        if key.fast:
            return self[PriceCatalogProduct(False, key.min_request_tokens)]
        raise KeyError(key)

    @override
    def __contains__(self, key: object) -> bool:
        # Defined as "the lookup succeeds" rather than re-deriving the floor
        # rule: a second copy drifted, reporting membership for a key whose
        # tier starts above it and then raising on access.
        if not isinstance(key, PriceCatalogProduct):
            return False
        try:
            _ = self[key]
        except KeyError:
            return False
        return True

    @override
    def __iter__(self) -> Iterator[PriceCatalogProduct]:
        return iter(self._prices)

    @override
    def __len__(self) -> int:
        return len(self._prices)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._prices!r})"
