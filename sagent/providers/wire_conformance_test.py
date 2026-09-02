"""Capability-to-wire conformance: the catalog is the only vocabulary.

``providers/conformance_test.py`` checks the SHAPE of the ``Model`` and
``Provider`` protocols. This module checks their CONTENT: for every
provider and every catalog row, each capability the row advertises must
survive to the request body, and nothing the row withholds may.

The bug class it exists to catch is a parallel mechanism for one fact --
a private per-provider table keyed off a different vocabulary than the
catalog. Symptoms come in both directions:

- Advertised but unreachable: the agent offers ``effort=off``, the wire
  builder has never heard of it, and the send raises.
- Withheld but reachable: a row declares no efforts because the model
  rejects ``thinkingConfig``, and the builder sends one anyway.

Every check drives a REAL body builder. A mapping table asserted against
another mapping table would restate the bug.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, Protocol, get_args

import pytest

from sagent.providers import PROVIDER_NAMES
from sagent.providers.anthropic.api import Anthropic, _AnthropicModel
from sagent.providers.dashscope.api import DashScope
from sagent.providers.google.api import Google, _build_request
from sagent.providers.minimax.api import MiniMax
from sagent.providers.moonshot.api import Moonshot
from sagent.providers.openai.api import OpenAI
from sagent.providers.openai.compat import OpenAICompatModel
from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelSettings,
    ThinkingBudget,
    ThinkingEffort,
    ThinkingOutput,
)
from sagent.types.model import ModelRequest
from sagent.types.providers import (
    UnsupportedTagError,
    resolve,
)
from sagent.types.runtime import UserMessage


if TYPE_CHECKING:
    from sagent.lib.custom_json import MutableJSON
    from sagent.types.model import Model
    from sagent.types.providers import ModelRole


class _CatalogProvider(Protocol):
    """The catalog surface every provider in the roster exposes.

    ``CAPABILITIES`` and ``TRANSPORT`` are ``ClassVar`` on the providers,
    so they must be ``ClassVar`` here for the structural match to hold.
    """

    CAPABILITIES: ClassVar[Mapping[str, ModelCapability]]
    TRANSPORT: ClassVar[ModelCapability]

    @property
    def ROLES(self) -> Mapping[ModelRole, str]:  # noqa: N802 -- matches the provider class attribute.
        ...

    def model(self, model_id: str | None = None) -> Model: ...


def _request() -> ModelRequest:
    """The smallest request that carries a thinking knob to the wire."""
    return ModelRequest(messages=[UserMessage(text="x")])


def _anthropic_thinking(model: Model, settings: ModelSettings) -> object:
    """Anthropic's ``output_config``, or ``None`` when it sends none."""
    assert isinstance(model, _AnthropicModel)
    model._settings = settings
    return model._build_kwargs(_request(), []).get("output_config")


def _google_thinking(model: Model, settings: ModelSettings) -> object:
    """Gemini's ``thinkingConfig``, or ``None`` when it sends none."""
    body: MutableJSON = _build_request(
        _request(), model.capability, settings, settings.limits
    )
    gen_config = body["generationConfig"]
    assert isinstance(gen_config, dict)
    return gen_config.get("thinkingConfig")


def _chat_thinking(model: Model, settings: ModelSettings) -> object:
    """Chat-completions reasoning knobs, or ``None`` when it sends none."""
    assert isinstance(model, OpenAICompatModel)
    model._settings = settings
    body = model._build_body(_request(), stream=False)
    knobs = {
        k: v
        for k, v in body.items()
        if k in ("reasoning_effort", "enable_thinking", "thinking_budget")
    }
    return knobs or None


# Each provider paired with the builder that turns a request into its
# wire body. Adding a provider without adding it here leaves its catalog
# unverified, so the roster is asserted complete below.
_WireBuilder = tuple[
    Callable[[], "_CatalogProvider"], Callable[["Model", ModelSettings], object]
]
_WIRE_BUILDERS: Mapping[str, _WireBuilder] = {
    "Anthropic": (lambda: Anthropic.from_key("k"), _anthropic_thinking),
    "Google": (lambda: Google.from_key("k"), _google_thinking),
    "OpenAI": (lambda: OpenAI.from_key("k"), _chat_thinking),
    "DashScope": (lambda: DashScope.from_key("k"), _chat_thinking),
    "MiniMax": (lambda: MiniMax.from_key("k"), _chat_thinking),
    "Moonshot": (lambda: Moonshot.from_key("k"), _chat_thinking),
}


def _rows() -> list[tuple[str, str]]:
    """Every ``(provider_name, model_id)`` the wire builders cover."""
    out: list[tuple[str, str]] = []
    for name, (make, _) in _WIRE_BUILDERS.items():
        provider = make()
        out.extend((name, model_id) for model_id in provider.CAPABILITIES)
    return out


_ROWS = _rows()


def _settings_for(capability: ModelCapability, effort: ThinkingEffort) -> ModelSettings:
    """The settings that ask for ``effort`` with the widest budget offered."""
    budgets: tuple[ThinkingBudget, ...] = ("fixed", "auto", "none")
    budget: ThinkingBudget = next(b for b in budgets if b in capability.thinking_budget)
    output: ThinkingOutput = "text" if "text" in capability.thinking_output else "none"
    # ``replace`` re-runs ``__init__``, which validates every axis against
    # the carried capability -- so an unofferable combination raises here
    # rather than reaching a wire builder.
    return replace(
        ModelSettings.narrowest(capability),
        thinking_effort=effort,
        thinking_budget=budget if effort != "none" else "none",
        thinking_output=output,
    )


@pytest.mark.parametrize(("provider_name", "model_id"), _ROWS, ids=str)
def test_every_advertised_effort_reaches_the_wire(
    provider_name: str, model_id: str
) -> None:
    """Each effort the row advertises builds a body carrying it.

    ``capability.thinking_effort`` is what ``ModelSettings.validate``
    accepts. An effort that passes that gate and then finds no wire
    mapping is a crash at send time.
    """
    make, thinking_of = _WIRE_BUILDERS[provider_name]
    model = make().model(model_id)
    for effort in model.capability.thinking_effort - {"none"}:
        sent = thinking_of(model, _settings_for(model.capability, effort))
        assert sent is not None, (
            f"{provider_name}/{model_id} advertises effort {effort!r} but the"
            " wire body carries no thinking knob"
        )


@pytest.mark.parametrize(("provider_name", "model_id"), _ROWS, ids=str)
def test_a_row_with_no_efforts_sends_no_thinking_knob(
    provider_name: str, model_id: str
) -> None:
    """A row that advertises no effort must not send one.

    ``thinking_effort == {"none"}`` is a positive claim: the model
    REJECTS the knob (gemini-1.5 rejects ``thinkingConfig`` outright, the
    qwen ``-instruct`` ids reject ``enable_thinking``). A builder reading
    a private table instead of the row sends it anyway and earns a 400.
    """
    make, thinking_of = _WIRE_BUILDERS[provider_name]
    model = make().model(model_id)
    if model.capability.thinking_effort - {"none"}:
        pytest.skip("row advertises efforts")
    assert thinking_of(model, _settings_for(model.capability, "none")) is None, (
        f"{provider_name}/{model_id} advertises no effort yet the wire body"
        " carries a thinking knob"
    )


@pytest.mark.parametrize(("provider_name", "model_id"), _ROWS, ids=str)
def test_distinct_efforts_stay_distinct_on_the_wire(
    provider_name: str, model_id: str
) -> None:
    """Efforts the row distinguishes must not collapse into one body.

    A ladder that flattens bills one level while the user selected
    another, and the collapse is invisible from either side alone.
    """
    make, thinking_of = _WIRE_BUILDERS[provider_name]
    model = make().model(model_id)
    efforts = sorted(model.capability.thinking_effort - {"none"})
    if not efforts:
        pytest.skip("row advertises no efforts")
    by_wire: dict[str, set[str]] = {}
    for effort in efforts:
        sent = str(thinking_of(model, _settings_for(model.capability, effort)))
        by_wire.setdefault(sent, set()).add(effort)
    # Vendors legitimately fold the ends of the ladder (OpenAI pre-5.6 maps
    # both ``xhigh`` and ``max`` to ``high``), so a collapse is a defect only
    # when every effort lands on one body.
    assert len(by_wire) > 1 or len(efforts) == 1, (
        f"{provider_name}/{model_id} sends one body for all {len(efforts)}"
        " efforts it advertises"
    )


@pytest.mark.parametrize(("provider_name", "model_id"), _ROWS, ids=str)
def test_every_advertised_context_resolves(provider_name: str, model_id: str) -> None:
    """``resolve`` is the only path from a tagged id to a capability.

    A context the catalog offers but ``resolve`` rejects is a window the
    user can see and never select.
    """
    make, _ = _WIRE_BUILDERS[provider_name]
    provider = make()
    caps = provider.CAPABILITIES
    for context in caps[model_id].context:
        _, settings = resolve(
            model_id + context,
            models=caps,
            roles=provider.ROLES,
            transport=provider.TRANSPORT,
        )
        assert settings.context == context
        assert settings.limits.max_request_tokens > 0


@pytest.mark.parametrize(("provider_name", "model_id"), _ROWS, ids=str)
def test_an_unoffered_context_is_rejected(provider_name: str, model_id: str) -> None:
    """Serving the base window under a ``+1m`` id understates the budget 4x."""
    make, _ = _WIRE_BUILDERS[provider_name]
    provider = make()
    caps = provider.CAPABILITIES
    for tag in get_args(ContextTag.__value__):
        if not tag or tag in caps[model_id].context:
            continue
        with pytest.raises(UnsupportedTagError):
            resolve(
                model_id + tag,
                models=caps,
                roles=provider.ROLES,
                transport=provider.TRANSPORT,
            )


@pytest.mark.parametrize(("provider_name", "model_id"), _ROWS, ids=str)
def test_every_priced_tier_survives_the_transport(
    provider_name: str, model_id: str
) -> None:
    """A tier the row prices but the transport withholds bills unreachably."""
    make, _ = _WIRE_BUILDERS[provider_name]
    provider = make()
    capability, settings = resolve(
        model_id,
        models=provider.CAPABILITIES,
        roles=provider.ROLES,
        transport=provider.TRANSPORT,
    )
    for product in capability.prices:
        # Construction validates, so an unreachable tier raises here.
        _ = replace(settings, service_tier=product.service_tier)


def test_every_catalog_backed_provider_has_a_wire_builder() -> None:
    """No provider ships a catalog this suite never drives to the wire.

    Without this the suite passes by omission: a new provider's rows are
    simply never parametrized.
    """
    skipped = {
        # Subprocess transports whose catalogs are the parent's, narrowed
        # by a CLI transport that strips the effort knob entirely; the
        # parent row is what this suite verifies.
        "AnthropicCLI",
        "GoogleCLI",
        # Subscription transports share the parent's catalog and body
        # builder; only auth and service tier differ.
        "OpenAISubscription",
        # No catalog of its own (empty base) or requires a live server.
        "OpenAICompat",
        "LlamaCpp",
        "SelfHosted",
    }
    missing = [
        name
        for name in PROVIDER_NAMES
        if name not in _WIRE_BUILDERS and name not in skipped
    ]
    assert not missing, (
        f"providers with no wire builder in this suite: {missing};"
        " add one so their catalog rows are driven to the wire"
    )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
