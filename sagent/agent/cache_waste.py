"""Per-turn prompt-cache miss detection.

Every provider's prompt cache reduces to the same primitive regardless of
vendor: an exact-prefix match against what the server has already seen.
:func:`detect_cache_miss` compares one turn's token counts against the
prior turn's to flag when a prefix that *should* have been servable from
cache (the prior turn's own request) instead reprocessed as fresh input --
and distinguishes an expected cause (the provider's cache TTL lapsed) from
an avoidable one (something -- e.g. a subagent spawn mutating the system
prompt, see #361 -- perturbed the prefix bytes).

This is detection only, mirroring Pi Agent's ``cache-stats.ts``: it reports
waste after the fact rather than preventing it. Prevention lives in
``tools.agent_spawn``'s hot/cold spawn flag.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from sagent.types.cost import TokenCost, TokenCount


__all__ = [
    "CacheMiss",
    "CacheWasteTotals",
    "detect_cache_miss",
    "summarize_cache_waste",
]

# Provider cache breakpoints operate at coarse granularity (Anthropic's own
# minimum cacheable block is on this order); a delta at or below this is
# breakpoint-placement noise, not a real miss worth reporting.
_NOISE_FLOOR_TOKENS = 1024  # config-globals: ignore -- cache-miss noise-floor dial


@dataclass(frozen=True, slots=True, kw_only=True)
class CacheMiss:
    """One turn's avoidable prompt-cache miss."""

    missed_tokens: int
    """Prior-turn prefix tokens that reprocessed as fresh input this turn."""

    wasted_cost_usd: float
    """Extra USD spent billing ``missed_tokens`` at the request rate instead
    of the cache-read rate."""

    idle_sec: float
    """Wall-clock gap since the prior response."""

    cause: Literal["ttl_expired", "prefix_mutated"]
    """``ttl_expired`` when the idle gap exceeds the provider's cache TTL --
    an expected miss. ``prefix_mutated`` otherwise: the TTL had not lapsed,
    so something changed the request bytes themselves."""


def detect_cache_miss(
    previous: TokenCount,
    current: TokenCount,
    *,
    idle_sec: float,
    cache_ttl_sec: float,
    model_changed: bool,
    spend: Callable[[TokenCount], TokenCost],
) -> CacheMiss | None:
    """Detect an avoidable cache miss between two consecutive responses.

    Args:
      previous: Token counts from the prior response on this agent.
      current: Token counts from the response just recorded.
      idle_sec: Wall-clock seconds between the two responses.
      cache_ttl_sec: The active model's selected prompt-cache lifetime.
      model_changed: True when ``current`` was served by a different model
          or provider than ``previous``. Token counts aren't comparable
          across tokenizers, so this always returns ``None``.
      spend: The active model's pricing function (``Model.spend``), used to
          price ``missed_tokens`` at both the request and cache-read rates.

    Returns:
      miss: The detected miss, or ``None`` when there is nothing to
          report (first turn, model swap, no prompt cache on this model,
          a prompt that shrank, or a delta at or below the noise floor).

    """
    if model_changed or cache_ttl_sec <= 0.0:
        return None
    prior_prefix = previous.request + previous.cache_write + previous.cache_read
    if prior_prefix <= 0:
        return None
    # A shorter prompt (compaction, ``clear``) dropped the old prefix on
    # purpose; only a prompt at least as long could have re-read it.
    if current.request + current.cache_write + current.cache_read < prior_prefix:
        return None
    missed = prior_prefix - current.cache_read
    if missed <= _NOISE_FLOOR_TOKENS:
        return None
    cause: Literal["ttl_expired", "prefix_mutated"] = (
        "ttl_expired" if idle_sec > cache_ttl_sec else "prefix_mutated"
    )
    as_fresh = spend(TokenCount(request=missed)).total
    as_cached = spend(TokenCount(cache_read=missed)).total
    return CacheMiss(
        missed_tokens=missed,
        wasted_cost_usd=max(0.0, as_fresh - as_cached),
        idle_sec=idle_sec,
        cause=cause,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CacheWasteTotals:
    """Session-wide rollup of detected cache misses."""

    missed_tokens: int = 0
    wasted_cost_usd: float = 0.0
    miss_count: int = 0
    ttl_expired_count: int = 0
    prefix_mutated_count: int = 0


def summarize_cache_waste(misses: Sequence[CacheMiss]) -> CacheWasteTotals:
    """Roll up a sequence of detected misses into session-wide totals.

    Args:
      misses: Detected misses, e.g. ``CostTracker.cache_misses``.

    Returns:
      totals: Aggregate counts and cost across ``misses``.

    """
    return CacheWasteTotals(
        missed_tokens=sum(m.missed_tokens for m in misses),
        wasted_cost_usd=sum(m.wasted_cost_usd for m in misses),
        miss_count=len(misses),
        ttl_expired_count=sum(1 for m in misses if m.cause == "ttl_expired"),
        prefix_mutated_count=sum(1 for m in misses if m.cause == "prefix_mutated"),
    )
