"""What one model accepts, and what one instance chose from it.

:class:`ModelCapability` and :class:`ModelSettings` share field NAMES:
``capability.x`` is the allowed set, ``settings.x`` is the chosen value, so
validation is ``settings.x in capability.x`` for every field. An axis added
to one side without the other is a field nothing can select or nothing
validates.

Each axis is TOTAL: its unset value is spelled ``none``, so no field is
``| None``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Literal, Self, TypeAliasType, cast, get_args, override

from sagent.types.cost import PriceCatalog


__all__ = [
    "ContextTag",
    "ModelCapability",
    "ModelLimits",
    "ModelSettings",
    "Permission",
    "ServiceTier",
    "SummaryKind",
    "ThinkingBudget",
    "ThinkingEffort",
    "ThinkingOutput",
]


type ThinkingEffort = Literal["none", "min", "low", "medium", "high", "xhigh", "max"]

type ThinkingBudget = Literal["none", "auto", "fixed"]

type ThinkingOutput = Literal["none", "text", "redacted"]

type ContextTag = Literal["", "+200k", "+1m"]

type ServiceTier = Literal["auto", "default", "flex", "priority"]

type SummaryKind = Literal["none", "auto", "concise", "detailed"]

type Permission = Literal["ask", "accept_edits", "bypass"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelLimits:
    """Hard ceilings for one context configuration of one model."""

    max_request_tokens: int = 0
    """Input tokens the model accepts; ``0`` means unknown."""

    max_response_tokens: int = 0
    """Output tokens the model can generate in one response."""

    max_request_bytes: int = 0
    """HTTP wire ceiling, distinct from the token window; ``0`` = none."""

    max_image_edge_px: int = 0
    """Long edge above which the server downscales; ``0`` = no resize."""

    max_image_bytes: int = 0
    """Per-image byte cap after resize; ``0`` = no cap."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCapability:
    """What one model offers: every value a caller may select.

    One catalog row, or one transport's restrictions. ``prices`` and the two
    trailing transport facts have no :class:`ModelSettings` counterpart --
    nothing selects a price, a retry policy, or an auth mode.

    ``temperature`` is absent: its domain is a continuous range, so it cannot
    join the membership check. It rides ``ModelRequest``.
    """

    model_id: str = ""
    """The id the vendor accepts on the wire, without option tags."""

    context: Mapping[ContextTag, ModelLimits] = field(
        default_factory=lambda: MappingProxyType({"": ModelLimits()})
    )
    """Limits per selectable context tag; its KEYS are the allowed set."""

    prices: PriceCatalog = field(default_factory=PriceCatalog)
    """USD rates, keyed by service tier and request-size threshold."""

    thinking_effort: frozenset[ThinkingEffort] = frozenset({"none"})
    """Effort levels this transport accepts."""

    thinking_budget: frozenset[ThinkingBudget] = frozenset({"none"})
    """Budget modes this transport accepts."""

    thinking_output: frozenset[ThinkingOutput] = frozenset({"none"})
    """Reasoning-visibility modes this transport accepts."""

    service_tier: frozenset[ServiceTier] = frozenset({"auto", "default"})
    """Speed/price tiers this transport accepts.

    ``auto`` and ``default`` are universal -- every vendor serves a request
    that names no tier. ``flex`` and ``priority`` are opt-in per row.
    """

    cache_ttl_sec: float = 0.0
    """Longest prompt-cache lifetime accepted, in seconds; ``0`` = no cache."""

    # Defaults to BOTH values, not to its unset one: a model row does not know
    # whether the server rolls history, and ``&`` can only remove -- so a row
    # asserting ``{False}`` would pin every transport and empty the meet.
    manage_context_server_side: frozenset[bool] = frozenset({False, True})
    """Whether the server may roll history under quota pressure."""

    # The two below are DECLARED by the transport, not narrowed from the row:
    # a catalog row cannot know who bills it or who retries it, so ``&`` takes
    # the transport's answer rather than intersecting with a meaningless one.
    retries_internally: bool = False
    """Whether the transport retries transient failures on its own."""

    account_auth: bool = False
    """Whether this transport bills an account rather than an API key."""

    def __and__(self, other: ModelCapability) -> Self:
        """Narrow to what BOTH offer -- a catalog row met with a transport.

        Args:
          other: The transport's restrictions.

        Returns:
          narrowed: What survives on both sides.

        """
        # ``model_id``, ``context``, and ``prices`` pass through: a transport
        # restricts which knobs it may send, never which windows a model has
        # or what it costs.
        return replace(
            self,
            model_id=self.model_id,
            context=self.context,
            prices=self.prices,
            thinking_effort=self.thinking_effort & other.thinking_effort,
            thinking_budget=self.thinking_budget & other.thinking_budget,
            thinking_output=self.thinking_output & other.thinking_output,
            service_tier=self.service_tier & other.service_tier,
            cache_ttl_sec=min(self.cache_ttl_sec, other.cache_ttl_sec),
            manage_context_server_side=(
                self.manage_context_server_side & other.manage_context_server_side
            ),
            retries_internally=other.retries_internally,
            account_auth=other.account_auth,
        )


@dataclass(slots=True, kw_only=True)
class ModelSettings:
    """What one model instance chose, within what its capability offers.

    MUTABLE, and the sole owner of these axes. A caller that keeps its own
    copy of a knob has to re-derive "is this still offered?" on every model
    swap, and each such copy drifted: an ``Agent`` field shadowing
    ``thinking_effort`` was set by ``/effort`` and never read by the wire
    builders, so the selected level silently never shipped.

    Every axis validates ON ASSIGNMENT against :attr:`capability`, so
    ``settings.thinking_budget = "auto"`` raises on a model that cannot
    think. That is why the capability is a field rather than an argument:
    a setter has no other way to reach it, and a caller who had to pass it
    could pass a different one than the model was built from.

    Choices only. A derived value does not belong here: the selected
    context's ceilings are :meth:`limits`, not a field.
    """

    # DECLARED FIRST so ``__init__`` binds it before any axis: the setter
    # below reads it to validate, and a later position leaves the first axis
    # assignment reaching an unset attribute.
    capability: ModelCapability = field(default_factory=ModelCapability)
    """What the transport offers. Not an axis -- nothing selects it."""

    context: ContextTag = ""
    """Selected context window."""

    thinking_effort: ThinkingEffort = "none"
    """Selected effort level."""

    thinking_budget: ThinkingBudget = "none"
    """Selected budget mode."""

    thinking_output: ThinkingOutput = "none"
    """Selected reasoning visibility."""

    service_tier: ServiceTier = "auto"
    """Selected speed/price tier."""

    cache_ttl_sec: float = 0.0
    """Selected prompt-cache lifetime, in seconds."""

    manage_context_server_side: bool = False
    """Whether the server was asked to roll history."""

    @override
    def __setattr__(self, name: str, value: object) -> None:
        """Reject an axis value :attr:`capability` does not offer.

        Assignment is the API -- ``settings.thinking_budget = "auto"`` --
        so the check lives here rather than in a method a caller can route
        around. ``capability`` itself is exempt: it is the authority, not a
        choice, and swapping it re-derives every axis via :meth:`adopt`.

        Args:
          name: Attribute being set.
          value: Its new value.

        Raises:
          ValueError: ``value`` is outside what :attr:`capability` allows.

        """
        if name != "capability":
            _reject_unoffered(self.capability, name, value)
        object.__setattr__(self, name, value)

    def adopt(self, other: Self) -> None:
        """Take every choice of ``other`` this capability still offers.

        The model-swap path: a swap keeps what the user selected. Excludes
        ``context``, because the incoming model id already named its own
        window -- overwriting it would undo the selection the swap was
        performed to make.

        Args:
          other: The outgoing model's settings.

        """
        self.take(
            **{
                f.name: getattr(other, f.name)
                for f in fields(self)
                if f.name not in ("capability", "context")
            }
        )

    def take(self, **choices: object) -> None:
        """Apply each of ``choices`` this capability offers; drop the rest.

        The lenient counterpart to assignment, and the ONLY way to cross
        capabilities: a knob the incoming model rejects is not merely stale,
        it ships on the next request and earns a 400 -- but a swap or a
        resume must not FAIL on it either. Each unoffered axis falls back to
        the lowest rung this capability does offer.

        Args:
          choices: Field name to desired value.

        Raises:
          ValueError: A name is not an axis of this class.

        """
        names = {f.name for f in fields(self)}
        unknown = sorted(set(choices) - names)
        if unknown:
            raise ValueError(f"not settings axes: {', '.join(unknown)}")
        narrow = type(self).narrowest(self.capability, context=self.context)
        for name, wanted in choices.items():
            offered = _offers(self.capability, name, wanted)
            setattr(self, name, wanted if offered else getattr(narrow, name))

    @classmethod
    def narrowest(
        cls, capability: ModelCapability, *, context: ContextTag = ""
    ) -> Self:
        """Return the least-committing selection ``capability`` offers.

        The field defaults are not it: a row can withhold the unset value of
        an axis (Model Studio's ``-thinking`` ids reject "do not think", a CLI
        transport requires server-side history), and a default-constructed
        settings object is then unbuildable against that row.

        Args:
          capability: What the transport offers.
          context: The context tag the model id selected.

        Returns:
          settings: The lowest rung of every ladder that survives.

        Raises:
          ValueError: An axis offers nothing at all.

        """
        return cls(
            capability=capability,
            context=context,
            thinking_effort=_lowest(
                "thinking_effort", capability.thinking_effort, _ladder(ThinkingEffort)
            ),
            thinking_budget=_lowest(
                "thinking_budget", capability.thinking_budget, _ladder(ThinkingBudget)
            ),
            thinking_output=_lowest(
                "thinking_output", capability.thinking_output, _ladder(ThinkingOutput)
            ),
            service_tier=_lowest(
                "service_tier", capability.service_tier, _ladder(ServiceTier)
            ),
            # Not the field default: a bound's lowest value is the EXPENSIVE
            # one, and the wire caches whether or not this says so. 300s is
            # the tier every price row encodes; a transport offering less
            # (or none) clamps.
            cache_ttl_sec=min(300.0, capability.cache_ttl_sec),
            manage_context_server_side=_lowest(
                "manage_context_server_side",
                capability.manage_context_server_side,
                (False, True),
            ),
        )

    @property
    def limits(self) -> ModelLimits:
        """Ceilings the selected context tag carries."""
        return self.capability.context[self.context]


def _offers(capability: ModelCapability, name: str, value: object) -> bool:
    """Whether ``capability`` allows ``value`` on the axis ``name``."""
    allowed = getattr(capability, name)
    # A ``Collection`` is the enumerated shape; anything else on a capability
    # is a numeric upper bound, checked by comparison.
    if isinstance(allowed, Collection):
        return value in allowed
    return cast(float, value) <= allowed


def _reject_unoffered(capability: ModelCapability, name: str, value: object) -> None:
    """Raise unless ``capability`` offers ``value`` on the axis ``name``."""
    if _offers(capability, name, value):
        return
    model = capability.model_id or "this model"
    allowed = getattr(capability, name)
    if isinstance(allowed, Collection):
        # ``repr`` before ``sorted``: these sets mix strings and bools, which
        # do not order against each other.
        offered = ", ".join(sorted(repr(v) for v in allowed))
        detail = f"is not offered by {model}; allowed: {offered}"
    else:
        detail = f"exceeds the most {model} offers: {allowed!r}"
    raise ValueError(f"{name}={value!r} {detail}")


def _ladder[T](alias: object) -> tuple[T, ...]:
    """The Literal's members, in declaration order -- least committing first."""
    return cast(tuple[T, ...], get_args(cast(TypeAliasType, alias).__value__))


def _lowest[T](name: str, offered: Collection[T], ladder: Sequence[T]) -> T:
    """The first rung of ``ladder`` that ``offered`` contains."""
    for rung in ladder:
        if rung in offered:
            return rung
    raise ValueError(f"{name} offers nothing selectable: {offered!r}")
