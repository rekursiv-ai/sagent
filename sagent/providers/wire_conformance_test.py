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
from typing import TYPE_CHECKING, ClassVar, Protocol

import pytest

from sagent.providers import PROVIDER_NAMES
from sagent.providers.anthropic.api import Anthropic, _AnthropicModel
from sagent.providers.dashscope.api import DashScope
from sagent.providers.google.api import Google, _build_request
from sagent.providers.minimax.api import MiniMax
from sagent.providers.moonshot.api import Moonshot
from sagent.providers.openai.api import OpenAI
from sagent.providers.openai.compat import OpenAICompatModel
from sagent.types.model import (
    ALL_THINKING_EFFORTS,
    Limits,
    ModelRequest,
)
from sagent.types.providers import (
    UnsupportedTagError,
    resolve,
)
from sagent.types.runtime import UserMessage


if TYPE_CHECKING:
    from sagent.lib.custom_json import MutableJSON
    from sagent.types.model import Model, ModelCapability
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


def _request(effort: str) -> ModelRequest:
    """The smallest request that carries an effort to the wire."""
    return ModelRequest(messages=[UserMessage(text="x")], effort=effort)


def _anthropic_thinking(model: Model, effort: str) -> object:
    """Anthropic's ``output_config``, or ``None`` when it sends none."""
    assert isinstance(model, _AnthropicModel)
    return model._build_kwargs(_request(effort), []).get("output_config")


def _google_thinking(model: Model, effort: str) -> object:
    """Gemini's ``thinkingConfig``, or ``None`` when it sends none."""
    body: MutableJSON = _build_request(_request(effort), model.spec)
    gen_config = body["generationConfig"]
    assert isinstance(gen_config, dict)
    return gen_config.get("thinkingConfig")


def _chat_thinking(model: Model, effort: str) -> object:
    """Chat-completions reasoning knobs, or ``None`` when it sends none."""
    assert isinstance(model, OpenAICompatModel)
    body = model._build_body(_request(effort), stream=False)
    knobs = {
        k: v
        for k, v in body.items()
        if k in ("reasoning_effort", "enable_thinking", "thinking_budget")
    }
    return knobs or None


# Each provider paired with the builder that turns a request into its
# wire body. Adding a provider without adding it here leaves its catalog
# unverified, so the roster is asserted complete below.
_WireBuilder = tuple[Callable[[], "_CatalogProvider"], Callable[["Model", str], object]]
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


@pytest.mark.parametrize(("provider_name", "model_id"), _ROWS, ids=str)
def test_every_advertised_effort_reaches_the_wire(
    provider_name: str, model_id: str
) -> None:
    """Each effort the row advertises builds a body carrying it.

    ``supported_thinking_efforts`` is what the agent offers the user and
    what ``Agent.effort`` validates against. An effort that passes that
    gate and then finds no wire mapping is a crash at send time.
    """
    make, thinking_of = _WIRE_BUILDERS[provider_name]
    model = make().model(model_id)
    for effort in model.spec.supported_thinking_efforts:
        sent = thinking_of(model, effort)
        assert sent is not None, (
            f"{provider_name}/{model_id} advertises effort {effort!r} but the"
            " wire body carries no thinking knob"
        )


@pytest.mark.parametrize(("provider_name", "model_id"), _ROWS, ids=str)
def test_a_row_with_no_efforts_sends_no_thinking_knob(
    provider_name: str, model_id: str
) -> None:
    """A row that advertises no effort must not send one.

    An empty ``supported_thinking_efforts`` is a positive claim: the
    model REJECTS the knob (gemini-1.5 rejects ``thinkingConfig``
    outright, the qwen ``-instruct`` ids reject ``enable_thinking``).
    A builder reading a private table instead of the row sends it
    anyway and earns a 400.
    """
    make, thinking_of = _WIRE_BUILDERS[provider_name]
    model = make().model(model_id)
    if model.spec.supported_thinking_efforts:
        pytest.skip("row advertises efforts")
    for effort in ALL_THINKING_EFFORTS:
        assert thinking_of(model, effort) is None, (
            f"{provider_name}/{model_id} advertises no effort yet the wire"
            f" body carries a thinking knob for {effort!r}"
        )


@pytest.mark.parametrize(("provider_name", "model_id"), _ROWS, ids=str)
def test_distinct_catalog_values_stay_distinct_on_the_wire(
    provider_name: str, model_id: str
) -> None:
    """Efforts the catalog distinguishes must not collapse in the body.

    Two tables encoding one ladder drift silently: the reader sees the
    catalog's value and the server sees the other one. Comparing the
    partition the catalog induces against the partition the wire induces
    catches drift without assuming a vendor's encoding (a number for
    Gemini, a bare toggle for a zero Qwen budget).
    """
    make, thinking_of = _WIRE_BUILDERS[provider_name]
    model = make().model(model_id)
    by_wire: dict[str, set[str]] = {}
    for effort, wire in model.spec.supported_thinking_efforts.items():
        by_wire.setdefault(str(thinking_of(model, effort)), set()).add(wire)
    collapsed = {sent: wires for sent, wires in by_wire.items() if len(wires) > 1}
    assert not collapsed, (
        f"{provider_name}/{model_id} sends one body for catalog values the"
        f" row declares distinct: {collapsed}"
    )


@pytest.mark.parametrize(("provider_name", "model_id"), _ROWS, ids=str)
def test_every_advertised_context_and_tier_resolves(
    provider_name: str, model_id: str
) -> None:
    """Every context tag resolves and every priced tier answers ``spend``.

    ``resolve`` is the single path from a tagged id to a ``ModelSpec``;
    a context the catalog offers but ``resolve`` rejects is a window the
    user can see and never select.
    """
    make, _ = _WIRE_BUILDERS[provider_name]
    provider = make()
    caps = provider.CAPABILITIES
    roles = provider.ROLES
    transport = provider.TRANSPORT
    limits = caps[model_id].context_limits
    contexts = [""] if isinstance(limits, Limits) else list(limits)
    for context in contexts:
        spec = resolve(
            model_id + context, models=caps, roles=roles, transport=transport
        )
        assert spec.context_limits.max_request_tokens > 0
    if caps[model_id].serves_fast:
        fast = resolve(
            model_id + "+fast", models=caps, roles=roles, transport=transport
        )
        assert fast.serve_fast


@pytest.mark.parametrize(("provider_name", "model_id"), _ROWS, ids=str)
def test_fast_is_rejected_exactly_when_it_is_unpriced(
    provider_name: str, model_id: str
) -> None:
    """``+fast`` resolves iff the row can bill it.

    Accepting the tag without a fast price row bills standard while
    reporting fast; rejecting it on a row that has one hides a tier the
    user pays for.
    """
    make, _ = _WIRE_BUILDERS[provider_name]
    provider = make()
    caps = provider.CAPABILITIES
    roles = provider.ROLES
    transport = provider.TRANSPORT
    if (caps[model_id] & transport).serves_fast:
        return
    with pytest.raises(UnsupportedTagError):
        resolve(model_id + "+fast", models=caps, roles=roles, transport=transport)


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
