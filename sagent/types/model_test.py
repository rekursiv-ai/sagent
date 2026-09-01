"""Tests for ``types.model``: validation and arithmetic on data classes."""

from __future__ import annotations

from types import MappingProxyType
from typing import cast

import pytest

from sagent.catalog import anthropic as anthropic_catalog
from sagent.providers.anthropic import api as anthropic
from sagent.types.cost import (
    PriceCatalog,
    PriceCatalogProduct,
    TokenPrice,
)
from sagent.types.model import (
    ContextBudget,
    Model,
    ModelCapability,
    ModelLimits,
    ModelRecipe,
    ModelResponse,
    ModelSpec,
    StreamInterruptedError,
    ThinkingBudget,
    ThinkingOutput,
    TokenCount,
    base_model_id,
    latency_from_model_id,
    split_model_id,
)
from sagent.types.providers import (
    UnsupportedTagError,
    resolve,
)
from sagent.types.runtime import AssistantMessage


class _TinyModel:
    """Smallest-possible ``Model``-shaped stub for ``ContextBudget.from_model``.

    Only the ``spec`` ``from_model`` reads is declared; callers
    ``cast`` it to ``Model`` because constructing the full Protocol
    surface for a one-method test is more noise than signal.
    """

    def __init__(
        self, *, max_request_tokens: int = 4_096, max_response_tokens: int = 1_024
    ) -> None:
        self.spec = ModelSpec(
            context_limits=ModelLimits(
                max_request_tokens=max_request_tokens,
                max_response_tokens=max_response_tokens,
            )
        )


def _tiny(*, max_request_tokens: int = 4_096) -> Model:
    return cast(Model, _TinyModel(max_request_tokens=max_request_tokens))


# ---- ContextBudget ---------------------------------------------------------


def test_context_budget_from_model_handles_small_window() -> None:
    """B12: small windows must not trip the ``buffer < max_request`` invariant.

    Pre-fix, ``buffer_tokens = max(inp // 15, 8_000)`` floored at
    8_000; any model with ``max_request_tokens < 8_000`` then failed
    ``__post_init__``'s ``buffer_tokens < max_request_tokens`` check.
    Test fakes and small local models broke loud.
    """
    _ = ContextBudget.from_model(_tiny(max_request_tokens=4_096))


def test_context_budget_from_model_caps_buffer_at_half_window() -> None:
    """The buffer never exceeds half the window even for tiny models."""
    budget = ContextBudget.from_model(_tiny(max_request_tokens=4_096))
    assert budget.buffer_tokens <= 4_096 // 2


def test_context_budget_from_model_keep_recent_defers_to_compactor() -> None:
    """B13: ``from_model`` leaves ``keep_recent_on_compact`` as ``None``.

    ``None`` is the "let the compactor pick" contract; baking a number
    here would silently override the strategy-specific default.
    """
    budget = ContextBudget.from_model(_tiny())
    assert budget.keep_recent_on_compact is None


def test_context_budget_rejects_negative_reattach_count() -> None:
    """S7: ``reattach_count < 0`` underflows downstream slice math."""
    with pytest.raises(ValueError, match="reattach_count"):
        _ = ContextBudget(
            max_request_tokens=1_000,
            max_response_tokens=100,
            reattach_count=-1,
        )


def test_context_budget_rejects_negative_reattach_max_chars() -> None:
    with pytest.raises(ValueError, match="reattach_max_chars"):
        _ = ContextBudget(
            max_request_tokens=1_000,
            max_response_tokens=100,
            reattach_max_chars=-1,
        )


def test_context_budget_rejects_negative_reattach_budget() -> None:
    """S7: caught for downstream char-budget math."""
    with pytest.raises(ValueError, match="reattach_budget"):
        _ = ContextBudget(
            max_request_tokens=1_000,
            max_response_tokens=100,
            reattach_budget=-1,
        )


def test_context_budget_rejects_negative_persist_tokens() -> None:
    with pytest.raises(ValueError, match="persist_tokens"):
        _ = ContextBudget(
            max_request_tokens=1_000,
            max_response_tokens=100,
            persist_tokens=-1,
        )


def test_context_budget_rejects_negative_message_budget_tokens() -> None:
    with pytest.raises(ValueError, match="message_budget_tokens"):
        _ = ContextBudget(
            max_request_tokens=1_000,
            max_response_tokens=100,
            message_budget_tokens=-1,
        )


def test_context_budget_rejects_negative_keep_recent_on_compact() -> None:
    """``keep_recent_on_compact`` is ``None | int``; ``-1`` is neither."""
    with pytest.raises(ValueError, match="keep_recent_on_compact"):
        _ = ContextBudget(
            max_request_tokens=1_000,
            max_response_tokens=100,
            keep_recent_on_compact=-1,
        )


def test_context_budget_accepts_none_keep_recent_on_compact() -> None:
    budget = ContextBudget(
        max_request_tokens=1_000,
        max_response_tokens=100,
        keep_recent_on_compact=None,
    )
    assert budget.keep_recent_on_compact is None


def test_context_budget_accepts_zero_buffer_tokens() -> None:
    """``buffer_tokens = 0`` is a legal "no headroom" setting."""
    budget = ContextBudget(
        max_request_tokens=1_000,
        max_response_tokens=100,
        buffer_tokens=0,
    )
    assert budget.buffer_tokens == 0


def test_context_budget_rejects_negative_buffer_tokens() -> None:
    """Negative ``buffer_tokens`` would inflate the usable window past the cap."""
    with pytest.raises(ValueError, match="buffer_tokens"):
        _ = ContextBudget(
            max_request_tokens=1_000,
            max_response_tokens=100,
            buffer_tokens=-1,
        )


# ---- TokenCount ------------------------------------------------------------


def test_token_count_sub_clamps_at_zero() -> None:
    """B10: ``__sub__`` never produces a negative field.

    ``CostTracker.restore_totals`` may move the cumulative total *below*
    a pre-restore snapshot; the status pane reads ``current - snapshot``
    and would render "-12 tokens" without the clamp.
    """
    small = TokenCount(request=10, response=5)
    big = TokenCount(request=100, response=50)
    diff = small - big
    assert diff.request == 0
    assert diff.response == 0
    assert diff.cache_write == 0
    assert diff.cache_read == 0


def test_token_count_sub_returns_positive_for_monotonic_case() -> None:
    """The clamp doesn't break the common monotonic path."""
    big = TokenCount(request=100, response=50)
    small = TokenCount(request=10, response=5)
    diff = big - small
    assert diff.request == 90
    assert diff.response == 45


def test_token_count_add_returns_not_implemented_for_non_token_count() -> None:
    """B15: arithmetic with a non-``TokenCount`` defers to the other operand.

    Pre-fix, ``TokenCount() + 1`` raised ``AttributeError`` mid-expression
    when Python's ``__add__`` dispatch tried to read ``other.request``.
    Returning ``NotImplemented`` lets the runtime try ``other.__radd__``
    and (if that also fails) raise a clear ``TypeError`` naming both
    operand types.
    """
    other = cast(TokenCount, 1)
    assert TokenCount().__add__(other) is NotImplemented


def test_token_count_sub_returns_not_implemented_for_non_token_count() -> None:
    """B15: ``__sub__`` similarly defers instead of raising ``AttributeError``."""
    other = cast(TokenCount, "nope")
    assert TokenCount().__sub__(other) is NotImplemented


# ---- ModelRecipe -------------------------------------------------------------


def test_model_recipe_rejects_empty_provider() -> None:
    """B14: empty ``provider`` would resolve to "no such provider" downstream."""
    with pytest.raises(ValueError, match="provider"):
        _ = ModelRecipe(provider="", auth="env", model_id="m")


def test_model_recipe_rejects_empty_auth() -> None:
    with pytest.raises(ValueError, match="auth"):
        _ = ModelRecipe(provider="P", auth="", model_id="m")


def test_model_recipe_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        _ = ModelRecipe(provider="P", auth="env", model_id="")


def test_model_recipe_allows_empty_account() -> None:
    """``account=""`` means "default backend"; not a degenerate spec."""
    spec = ModelRecipe(provider="P", auth="env", model_id="m", account="")
    assert spec.account == ""


def test_model_recipe_allows_none_account() -> None:
    spec = ModelRecipe(provider="P", auth="env", model_id="m", account=None)
    assert spec.account is None


# ---- StreamInterruptedError ------------------------------------------------


def test_stream_interrupted_message_embeds_response_counts() -> None:
    """B8: the hardcoded message now surfaces token counts + stop_reason.

    The provider's stop signal is the operator's first hint at what
    went wrong; embedding it in the exception message saves a separate
    log dive.
    """
    response = ModelResponse(
        message=AssistantMessage(text="partial"),
        tokens=TokenCount(request=42, response=7),
        stop_reason="tool_use",
    )
    err = StreamInterruptedError(response)
    text = str(err)
    assert "input_tokens=42" in text
    assert "output_tokens=7" in text
    assert "tool_use" in text


@pytest.mark.parametrize(
    ("model_id", "base"),
    [
        ("claude-opus-4-7+1m", "claude-opus-4-7"),
        ("claude-opus-4-7+200k", "claude-opus-4-7"),
        ("claude-opus-4-7", "claude-opus-4-7"),
        ("Claude-Opus-4-7+1M", "Claude-Opus-4-7"),
    ],
)
def test_base_model_id_strips_window_tag(model_id: str, base: str) -> None:
    assert base_model_id(model_id) == base


@pytest.mark.parametrize(
    ("model_id", "base", "tags"),
    [
        ("claude-opus-4-8", "claude-opus-4-8", frozenset[str]()),
        ("claude-opus-4-8+fast", "claude-opus-4-8", frozenset({"+fast"})),
        ("claude-opus-4-8+1m+fast", "claude-opus-4-8", frozenset({"+1m", "+fast"})),
        ("claude-opus-4-8+fast+1m", "claude-opus-4-8", frozenset({"+1m", "+fast"})),
        ("Claude-Opus-4-8+FAST", "Claude-Opus-4-8", frozenset({"+fast"})),
        ("model+unknown", "model+unknown", frozenset[str]()),
    ],
)
def test_split_model_id(model_id: str, base: str, tags: frozenset[str]) -> None:
    assert split_model_id(model_id) == (base, tags)


def test_latency_from_model_id() -> None:
    assert latency_from_model_id("claude-opus-4-8+fast") == "fast"
    assert latency_from_model_id("claude-opus-4-8+1m+fast") == "fast"
    assert latency_from_model_id("claude-opus-4-8+1m") is None
    assert latency_from_model_id("claude-opus-4-8") is None


# ---- ModelCapability / ModelSpec -------------------------------------------


def _opus() -> ModelCapability:
    return ModelCapability(
        model_id="claude-opus-4-8",
        context_limits=MappingProxyType(
            {
                "": ModelLimits(max_request_tokens=200_000, max_image_bytes=5_000_000),
                "+1m": ModelLimits(
                    max_request_tokens=1_000_000, max_image_bytes=5_000_000
                ),
            }
        ),
        prices=PriceCatalog(
            {
                PriceCatalogProduct(False, 0): TokenPrice(request=5.0),
                PriceCatalogProduct(True, 0): TokenPrice(request=15.0),
            }
        ),
        supported_thinking_efforts=MappingProxyType({"off": "off", "max": "max"}),
    )


def _cli() -> ModelCapability:
    return ModelCapability(
        fast=False,
        prompt_cache_breakpoints=False,
        retries_internally=False,
        supported_thinking_efforts=MappingProxyType({}),
        supported_thinking_outputs=frozenset({"text"}),
    )


def test_every_field_is_default_constructible() -> None:
    assert ModelCapability().model_id == ""
    assert ModelSpec().context_limits == ModelLimits()


def test_bare_capability_restricts_nothing() -> None:
    assert (_opus() & ModelCapability()) == _opus()


def test_meet_only_removes() -> None:
    met = _opus() & _cli()
    assert dict(met.supported_thinking_efforts) == {}
    assert met.prompt_cache_breakpoints is False
    assert met.retries_internally is False
    assert met.serves_fast is False
    assert met.supported_thinking_outputs == frozenset({"text"})


def test_meet_cannot_grant() -> None:
    restricted = ModelCapability(model_id="m", prompt_cache_breakpoints=False)
    assert (restricted & ModelCapability()).prompt_cache_breakpoints is False


def test_meet_drops_fast_price_rows() -> None:
    met = _opus() & _cli()
    assert not any(k.fast for k in met.prices)


def test_narrow_selects_one_context() -> None:
    spec = ModelSpec.narrow(_opus(), context="+1m", fast=True)
    assert spec.context_limits == ModelLimits(
        max_request_tokens=1_000_000,
        max_image_bytes=5_000_000,
    )


def test_narrow_of_single_context_capability() -> None:
    cap = ModelCapability(
        model_id="haiku", context_limits=ModelLimits(max_request_tokens=7)
    )
    assert ModelSpec.narrow(cap).context_limits.max_request_tokens == 7


def test_narrow_rejects_an_unknown_context() -> None:
    with pytest.raises(KeyError):
        ModelSpec.narrow(_opus(), context="+2m")


def test_wire_id_excludes_tags_and_display_id_includes_them() -> None:
    spec = ModelSpec.narrow(_opus(), context="+1m", fast=True)
    assert spec.model_id == "claude-opus-4-8"
    assert spec.tagged_model_id == "claude-opus-4-8+1m+fast"


def test_narrow_carries_every_capability_field() -> None:
    spec = ModelSpec.narrow(_opus() & _cli())
    assert spec.prompt_cache_breakpoints is False
    assert dict(spec.supported_thinking_efforts) == {}


def test_spend_uses_the_tier_the_request_falls_into() -> None:
    spec = ModelSpec.narrow(
        ModelCapability(
            prices=PriceCatalog(
                {
                    PriceCatalogProduct(False, 0): TokenPrice(request=2.0),
                    PriceCatalogProduct(False, 200_000): TokenPrice(request=4.0),
                }
            )
        )
    )
    assert spec.spend(TokenCount(request=1_000_000)).request == 4.0
    assert spec.spend(TokenCount(request=100_000)).request == 0.2


def test_serves_fast_requires_both_a_row_and_transport_support() -> None:
    assert _opus().serves_fast is True
    assert (_opus() & _cli()).serves_fast is False


def test_spend_honors_the_server_reported_speed() -> None:
    """A request that asked for fast but fell back bills at standard."""
    cap = ModelCapability(
        prices=PriceCatalog(
            {
                PriceCatalogProduct(): TokenPrice(request=5.0),
                PriceCatalogProduct(fast=True): TokenPrice(request=10.0),
            }
        )
    )
    spec = ModelSpec.narrow(cap, fast=True)
    tokens = TokenCount(request=1_000_000)
    assert spec.spend(tokens).request == 10.0
    assert spec.spend(tokens, served_fast=False).request == 5.0


def test_chars_per_token_survives_meet_and_narrow() -> None:
    cap = ModelCapability(chars_per_token=2.83) & _cli()
    assert ModelSpec.narrow(cap).chars_per_token == 2.83


def _spec(*, budgets: frozenset[str], outputs: frozenset[str]) -> ModelSpec:
    return ModelSpec(
        supported_thinking_budgets=cast(frozenset[ThinkingBudget], budgets),
        supported_thinking_outputs=cast(frozenset[ThinkingOutput], outputs),
    )


@pytest.mark.parametrize(
    ("budgets", "outputs", "expected"),
    [
        # No thinking (OpenAI-chat, SelfHosted, LlamaCpp): off only.
        (frozenset[str](), frozenset[str](), ("off-hide",)),
        # Readable, no redaction (Google, DashScope): all but redact.
        (
            frozenset({"auto", "fixed"}),
            frozenset({"text"}),
            ("adaptive-show", "adaptive-hide", "on-show", "on-hide", "off-hide"),
        ),
        # Readable + redaction (plain Anthropic 4-6): all six.
        (
            frozenset({"auto", "fixed"}),
            frozenset({"text", "redacted"}),
            (
                "adaptive-show",
                "adaptive-hide",
                "on-show",
                "on-hide",
                "off-hide",
                "redact-hide",
            ),
        ),
        # Signed-but-empty blocks: every ``-show`` drops out.
        (
            frozenset({"auto", "fixed"}),
            frozenset({"redacted"}),
            ("adaptive-hide", "on-hide", "off-hide", "redact-hide"),
        ),
        # Adaptive-only (rejects ``enabled``): the ``on-*`` states drop out.
        (
            frozenset({"auto"}),
            frozenset({"text", "redacted"}),
            ("adaptive-show", "adaptive-hide", "off-hide", "redact-hide"),
        ),
        # opus-4-8: adaptive-only AND no readable text.
        (
            frozenset({"auto"}),
            frozenset({"redacted"}),
            ("adaptive-hide", "off-hide", "redact-hide"),
        ),
        # 4-5 generation: enabled-only, readable, no redaction.
        (
            frozenset({"fixed"}),
            frozenset({"text"}),
            ("on-show", "on-hide", "off-hide"),
        ),
    ],
)
def test_valid_thinking_states_derive_from_the_three_axes(
    budgets: frozenset[str],
    outputs: frozenset[str],
    expected: tuple[str, ...],
) -> None:
    assert _spec(budgets=budgets, outputs=outputs).valid_thinking_states == expected


def test_valid_thinking_states_always_includes_off() -> None:
    """``off-hide`` is reachable on every model."""
    cases = cast(
        tuple[tuple[frozenset[str], frozenset[str]], ...],
        (
            (frozenset(), frozenset()),
            (frozenset({"auto", "fixed"}), frozenset({"text"})),
            (frozenset({"auto"}), frozenset({"redacted"})),
            (frozenset({"fixed"}), frozenset({"text", "redacted"})),
        ),
    )
    for budgets, outputs in cases:
        spec = _spec(budgets=budgets, outputs=outputs)
        assert "off-hide" in spec.valid_thinking_states


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)


def test_resolve_rejects_fast_the_transport_removed() -> None:
    """The fast check must run AFTER the transport meet, not before.

    A row with fast rows plus a transport that strips fast (the
    ``AnthropicCLI`` shape) otherwise yields ``serve_fast=True`` on a
    spec whose fast price row is gone: sagent believes it is serving
    fast, sends no fast flag, and bills standard.
    """
    models = {
        "m": ModelCapability(
            model_id="m",
            prices=PriceCatalog(
                {
                    PriceCatalogProduct(): TokenPrice(request=2.0),
                    PriceCatalogProduct(fast=True): TokenPrice(request=9.0),
                }
            ),
        )
    }
    with pytest.raises(UnsupportedTagError, match="does not support fast"):
        _ = resolve(
            "m+fast",
            models=models,
            roles={},
            transport=ModelCapability(fast=False),
        )


def test_narrow_is_idempotent_on_an_already_narrowed_spec() -> None:
    """Re-narrowing must not silently drop ``context`` / ``serve_fast``.

    ``ModelSpec`` IS a ``ModelCapability``, so it type-checks as input to
    ``narrow``; dropping the tags there yields a spec that claims the
    default context while carrying the ``+1m`` limits.
    """
    cap = ModelCapability(
        model_id="m",
        context_limits=MappingProxyType(
            {
                "": ModelLimits(max_request_tokens=1),
                "+1m": ModelLimits(max_request_tokens=2),
            }
        ),
    )
    once = ModelSpec.narrow(cap, context="+1m", fast=True)
    twice = ModelSpec.narrow(once, context="+1m", fast=True)
    assert twice.context == "+1m"
    assert twice.serve_fast is True
    assert twice.tagged_model_id == once.tagged_model_id


def test_meet_preserves_the_narrowed_type() -> None:
    """``ModelSpec & cap`` must stay a ``ModelSpec``.

    Returning the base class silently discards ``context`` and
    ``serve_fast``, so a later ``tagged_model_id`` read loses the tags.
    """
    cap = ModelCapability(
        model_id="m",
        context_limits=MappingProxyType({"": ModelLimits(), "+1m": ModelLimits()}),
    )
    spec = ModelSpec.narrow(cap, context="+1m", fast=True)
    met = spec & ModelCapability()
    assert isinstance(met, ModelSpec)
    assert met.tagged_model_id == "m+1m+fast"


@pytest.mark.parametrize(
    ("model_id", "serves_fast"),
    [
        ("claude-opus-5", True),
        ("claude-opus-4-8", True),
        # Fast mode was removed from 4-7 on 2026-07-24 and never shipped on
        # 4-6: a fast request there is served at standard speed and billed at
        # standard rates, so a fast price row overstates the cost.
        ("claude-opus-4-7", False),
        ("claude-opus-4-6", False),
    ],
)
def test_only_documented_models_carry_a_fast_tier(
    model_id: str, serves_fast: bool
) -> None:
    """Fast mode is Opus 5 and Opus 4.8 only.

    https://code.claude.com/docs/en/fast-mode
    """
    assert anthropic_catalog.MODELS[model_id].serves_fast is serves_fast


def test_resolve_accepts_tags_in_any_order() -> None:
    """``split_model_id`` documents tags "in any order"; resolve must agree.

    A hand-rolled parser that stripped ``+fast`` only as a trailing
    suffix made ``+fast+1m`` read as context ``+fast+1m`` and raise,
    while ``+1m+fast`` resolved.
    """
    models = {
        "m": ModelCapability(
            model_id="m",
            context_limits=MappingProxyType(
                {
                    "": ModelLimits(max_request_tokens=1),
                    "+1m": ModelLimits(max_request_tokens=2),
                }
            ),
            prices=PriceCatalog(
                {
                    PriceCatalogProduct(): TokenPrice(),
                    PriceCatalogProduct(fast=True): TokenPrice(),
                }
            ),
        )
    }
    first = resolve("m+1m+fast", models=models, roles={})
    second = resolve("m+fast+1m", models=models, roles={})
    assert first.context == second.context == "+1m"
    assert first.serve_fast is second.serve_fast is True
    assert first.tagged_model_id == second.tagged_model_id


def test_from_model_uses_the_measured_chars_per_token() -> None:
    """The budget must plan against the model's real tokenizer density.

    The two re-attach caps are the only fields still denominated in
    characters, so they must plan against the model's real tokenizer
    density -- hardcoding 4 against a measured 2.38 makes them ~68% too
    generous against the token window they exist to protect.
    """
    model = anthropic.Anthropic.from_key("k").model("claude-opus-5")
    assert model.spec.chars_per_token == 2.38
    budget = ContextBudget.from_model(model)
    assert budget.chars_per_token == 2
