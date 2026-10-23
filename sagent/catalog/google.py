"""Gemini thinking-wire mappings and transport restrictions.

Load model facts with ``sagent.catalog.models("google")``.
Transport capabilities narrow those rows with ``&``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.types.capability import (
    ModelCapability,
    ThinkingCapability,
    ThinkingEffort,
)


if TYPE_CHECKING:
    from treekle import MutablePlainTree

    from sagent.types.capability import ModelSettings


__all__ = [
    "api",
    "cli",
    "subscription",
    "thinking_budget",
    "thinking_config",
]


def thinking_budget(effort: ThinkingEffort) -> str:
    """Return the ``thinkingConfig.thinkingBudget`` an effort sends.

    Args:
      effort: Selected effort level.

    Returns:
      budget: Wire value, as the string the request body carries.

    Raises:
      ValueError: ``effort`` is ``none``; a disabled request omits
          ``thinkingConfig`` rather than sending a budget of zero.

    """
    match effort:
        case "none":
            raise ValueError(f"effort {effort!r} sends no budget; omit thinkingConfig")
        case "min":
            return "1024"
        case "low":
            return "4096"
        case "medium":
            return "8192"
        case "high":
            return "16384"
        case "xhigh":
            return "20480"
        case "max":
            return "24576"


def thinking_config(
    capability: ModelCapability,
    settings: ModelSettings,
) -> dict[str, MutablePlainTree] | None:
    """Return the ``generationConfig.thinkingConfig`` a selection sends.

    Budget ``none`` omits the config and effort ``none`` omits the level or
    cap, so the model's own default applies in both cases. Gemini 3 rows take
    a ``thinkingLevel`` under either budget mode, its levels being dynamic
    already; 2.5 rows take a ``thinkingBudget`` token cap, ``-1`` under
    ``auto``.

    Args:
      capability: The row the request is built against.
      settings: The selection in effect.

    Returns:
      config: The wire object, or ``None`` when the request sends none.

    """
    if settings.thinking_budget == "none":
        return None
    config: dict[str, MutablePlainTree] = {
        "includeThoughts": settings.thinking_output == "text",
    }
    effort = settings.thinking_effort
    if capability.effort_as_level:
        # Gemini 3 accepts ``thinkingBudget`` only for backwards compatibility.
        if effort != "none":
            config["thinkingLevel"] = _thinking_level(effort)
    elif settings.thinking_budget == "auto":
        config["thinkingBudget"] = -1
    elif effort != "none":
        config["thinkingBudget"] = int(thinking_budget(effort))
    return config


# Gemini surfaces readable thought parts and offers no server-side redaction.
# A row carries only what the MODEL can do; caching, retry, auth mode, and the
# fast path are transport facts, since ``&`` can only remove and a row saying
# ``False`` would pin every transport.
def api() -> ModelCapability:
    """Return what the REST API adds on top of a model row.

    Nothing; named so the three transports read alike at their call sites.

    One model, three ways in::

        from sagent.catalog import models

        models("google")["gemini-pro-3.1"] & api()          # REST, key auth
        models("google")["gemini-pro-3.1"] & subscription() # same wire, OAuth
        models("google")["gemini-pro-3.1"] & cli()          # ACP: no effort knob

    Returns:
      capability: The API transport's restrictions.

    """
    # Every axis stated: ``&`` can only remove, and a defaulted axis is the
    # narrow value, so an omitted one would strip the model's real capability.
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
    )


def cli() -> ModelCapability:
    """Return what the ``gemini`` subprocess narrows a model row to.

    One model, three ways in::

        from sagent.catalog import models

        models("google")["gemini-pro-3.1"] & api()          # REST, key auth
        models("google")["gemini-pro-3.1"] & subscription() # same wire, OAuth
        models("google")["gemini-pro-3.1"] & cli()          # ACP: no effort knob

    Returns:
      capability: The CLI transport's restrictions.

    """
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
        manage_context_server_side=frozenset({True}),
        account_auth=True,
    )


def subscription() -> ModelCapability:
    """Return what OAuth billing narrows a model row to.

    One model, three ways in::

        from sagent.catalog import models

        models("google")["gemini-pro-3.1"] & api()          # REST, key auth
        models("google")["gemini-pro-3.1"] & subscription() # same wire, OAuth
        models("google")["gemini-pro-3.1"] & cli()          # ACP: no effort knob

    Returns:
      capability: The subscription transport's restrictions.

    """
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
        account_auth=True,
    )


def _thinking_level(effort: ThinkingEffort) -> str:
    """Return the Gemini 3 ``thinkingLevel`` an effort sends."""
    match effort:
        case "min":
            return "minimal"
        case "low" | "medium" | "high":
            return effort
        case "none" | "xhigh" | "max":
            raise ValueError(f"effort {effort!r} has no Gemini thinkingLevel")
