"""What a model offers, where it starts, and what one model instance chose.

model = catalog["anthropic"]["opus"]
model.capability.thinking_effort                    # every effort level offered
model.capability.default                            # what a new model starts with
model.settings.context_size.tokens
model.capability.prices["auto", "smallcontext"]         # the size ``smallcontext`` names
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAliasType, cast, get_args

from sagent.model.context import ContextSize
from sagent.model.cost import PriceCatalog, ServiceTier


if TYPE_CHECKING:
    from collections.abc import Collection


__all__ = [
    "ModelCapability",
    "ModelLimits",
    "ModelSettings",
    "Permission",
    "PruningPolicy",
    "SummaryKind",
    "ThinkingBudget",
    "ThinkingEffort",
    "ThinkingOutput",
    "literal_members",
]


type ThinkingEffort = Literal[
    "none",
    "min",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]

type ThinkingBudget = Literal[
    "none",
    "auto",
    "fixed",
]

type ThinkingOutput = Literal[
    "none",
    "text",
    "redacted",
]

type SummaryKind = Literal[
    "none",
    "auto",
    "concise",
    "detailed",
]

type Permission = Literal[
    "ask",
    "accept_edits",
    "bypass",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelLimits:
    """Hard ceilings of one model's window."""

    max_request_tokens: int = 0
    """Input tokens the model accepts; ``0`` means unknown."""

    max_response_tokens: int = 0
    """Output tokens the model can generate in one response, thinking included."""

    max_context_tokens: int = 0
    """Input and output together; ``0`` means ``max_request_tokens`` bounds both."""

    truncates_at_context: bool | None = None
    """Whether input plus ``max_tokens`` past the context is accepted and the reply
    stopped there (``True``) or the request rejected (``False``); ``None`` is unknown."""

    max_request_bytes: int = 0
    """HTTP wire ceiling, distinct from the token window; ``0`` = none."""

    max_image_edge_px: int = 0
    """Long edge above which the server downscales; ``0`` = no resize."""

    max_image_bytes: int = 0
    """Per-image byte cap after resize; ``0`` = no cap."""

    request_betas: frozenset[str] = frozenset()
    """Vendor beta features every request needs."""

    @property
    def context_tokens(self) -> int:
        """Return what input and output may hold together."""
        return self.max_context_tokens or self.max_request_tokens


@dataclass(frozen=True, slots=True, kw_only=True)
class PruningPolicy:
    """Which old tool output may be replaced by a stub, to keep the context small.

    The client prunes when the cost says so (``agent.pruner.Pruner``); a server
    that prunes, with ``manage_context_server_side``, goes by the two token marks.
    """

    trigger_tokens: int = 100_000
    """A server that prunes starts once a request's context passes this."""

    target_tokens: int = 40_000
    """A server that prunes stops once the context is back near this."""

    keep_recent: int = 3
    """The newest tool results, which are never pruned."""

    expiry: float = 1.0
    """``λ``: an unused result expires once its rent reaches ``λ`` times the cost of fetching it again."""

    keep_tools: frozenset[str] = frozenset({"Skill"})
    """Tools whose results are never pruned; a skill's steps stay while it runs."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelSettings:
    """What one model instance chose, each choice resolved."""

    thinking_effort: ThinkingEffort = "none"
    thinking_budget: ThinkingBudget = "none"
    thinking_output: ThinkingOutput = "none"
    service_tier: ServiceTier = "auto"
    cache_ttl_sec: float = 0.0
    manage_context_server_side: bool = False
    context_size: ContextSize = field(default_factory=ContextSize)
    pruning: PruningPolicy | None = field(default_factory=PruningPolicy)


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCapability:
    """What one model offers, where it starts, and what it costs."""

    thinking_effort: frozenset[ThinkingEffort] = frozenset({"none"})
    thinking_budget: frozenset[ThinkingBudget] = frozenset({"none"})
    thinking_output: frozenset[ThinkingOutput] = frozenset({"none"})
    service_tier: frozenset[ServiceTier] = frozenset({"auto", "default"})
    manage_context_server_side: frozenset[bool] = frozenset({False, True})
    cache_ttl_sec: frozenset[float] = frozenset({0.0})
    default: ModelSettings = field(default_factory=ModelSettings)

    model_id: str = ""
    wire_model_id: str = ""
    knowledge_cutoff: str | None = None
    approx_chars_per_token: float = 4.0
    limits: ModelLimits = field(default_factory=ModelLimits)
    prices: PriceCatalog = field(default_factory=PriceCatalog)
    effort_as_level: bool = False
    retries_internally: bool = False
    account_auth: bool = False

    def check(self, settings: ModelSettings) -> None:
        """Raise unless this model offers every choice in ``settings``.

        Args:
          settings: The choices to check.

        Raises:
          ValueError: A choice this model does not offer, named with what it does.

        """
        for name, value, offered in (
            ("thinking_effort", settings.thinking_effort, self.thinking_effort),
            ("thinking_budget", settings.thinking_budget, self.thinking_budget),
            ("thinking_output", settings.thinking_output, self.thinking_output),
            ("service_tier", settings.service_tier, self.service_tier),
            ("cache_ttl_sec", settings.cache_ttl_sec, self.cache_ttl_sec),
            (
                "manage_context_server_side",
                settings.manage_context_server_side,
                self.manage_context_server_side,
            ),
        ):
            _check_offered(self.model_id, name, value, offered)

    def fit(
        self,
        settings: ModelSettings,
        *,
        was: ModelCapability,
        service_tier: ServiceTier | None = None,
    ) -> ModelSettings:
        """Return ``settings`` as this model takes them, at ``service_tier``.

        Each choice this model offers is kept, the rest are its defaults. A size
        ``was`` named stays named; a size counted stays that count, within the window.

        Args:
          settings: Choices made on ``was``.
          was: The model they were made on; this one, to change only the tier.
          service_tier: The tier to move to; ``None`` keeps the one chosen.

        Returns:
          settings: Each choice offered here.

        """
        d = self.default
        tier: ServiceTier = _kept(
            service_tier or settings.service_tier,
            self.service_tier,
            d.service_tier,
        )
        size = settings.context_size
        named = was.prices[settings.service_tier, size.context_name] == size
        return ModelSettings(
            thinking_effort=_kept(
                settings.thinking_effort,
                self.thinking_effort,
                d.thinking_effort,
            ),
            thinking_budget=_kept(
                settings.thinking_budget,
                self.thinking_budget,
                d.thinking_budget,
            ),
            thinking_output=_kept(
                settings.thinking_output,
                self.thinking_output,
                d.thinking_output,
            ),
            service_tier=tier,
            cache_ttl_sec=_kept(
                settings.cache_ttl_sec,
                self.cache_ttl_sec,
                d.cache_ttl_sec,
            ),
            manage_context_server_side=_kept(
                settings.manage_context_server_side,
                self.manage_context_server_side,
                d.manage_context_server_side,
            ),
            context_size=self.prices[tier, size.context_name if named else size.tokens],
            pruning=settings.pruning,
        )


def literal_members[T](alias: object) -> tuple[T, ...]:
    """Return a ``type X = Literal[...]`` alias's members, in declaration order.

    ``get_args`` of the alias itself is ``()``: the members live on its
    ``__value__``, which typeshed types as ``Any``.

    Args:
      alias: The alias, such as ``ThinkingEffort``.

    Returns:
      members: Each member, the least committing first where a ladder is meant.

    """
    return cast(
        tuple[T, ...],
        get_args(cast(object, cast(TypeAliasType, alias).__value__)),
    )


def _kept[T](value: T, offered: Collection[T], default: T) -> T:
    """Return ``value`` if ``offered`` holds it, else ``default``."""
    return value if value in offered else default


def _check_offered(
    model_id: str,
    name: str,
    value: object,
    offered: Collection[object],
) -> None:
    """Raise unless ``offered`` holds ``value``."""
    if value in offered:
        return
    # ``repr`` before ``sorted``: these sets mix strings, floats, and bools,
    # which do not order against each other.
    allowed = ", ".join(sorted(repr(v) for v in offered))
    raise ValueError(
        f"{name}={value!r} is not offered by {model_id or 'this model'};"
        f" allowed: {allowed}",
    )
