"""AgentSelf: change the calling agent: its title, model, or settings.

A patch is planned whole, then applied: an invalid setting rejects all of it.
What the agent's session holds is the Session tool's to change.

Typical usage example:

  me = AgentSelf(team=team)
  result = await me.call(context, {"status": "Fixing the parser", "effort": "high"})
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Literal

from sagent.agent.team import Team
from sagent.model.capability import ThinkingBudget, ThinkingEffort, ThinkingOutput
from sagent.model.cost import ServiceTier
from sagent.providers.catalog import successor
from sagent.providers.table import ModelTags
from sagent.runtime import types
from sagent.session.types import UncategorizedToolResult
from sagent.tools.display import ToolDisplay
from sagent.tools.types import parse, schema_of


if TYPE_CHECKING:
    from collections.abc import Mapping

    from treekle import PlainTree

    from sagent.agent.agent import Agent
    from sagent.model.capability import ModelCapability, ModelSettings
    from sagent.providers.types import Model
    from sagent.tools.types import ToolContext


__all__ = [
    "AgentSelf",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSelf:
    """Change the calling agent."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        status: Annotated[str | None, "3-7 words naming what the session is about."] = (
            None
        )
        provider: Annotated[str | None, "Catalog provider to switch to."] = None
        model: Annotated[
            str | None,
            "Model name or role to switch to: [PROVIDER/]MODEL[+CONTEXT][+EFFORT].",
        ] = None
        effort: Annotated[ThinkingEffort | None, "Reasoning effort."] = None
        budget: Annotated[
            ThinkingBudget | None,
            "Reasoning budget: none (no reasoning), auto, or fixed.",
        ] = None
        output: Annotated[
            ThinkingOutput | None,
            "What of the reasoning comes back: none, text, or redacted.",
        ] = None
        service_tier: Annotated[
            ServiceTier | None,
            "Serving tier, such as priority.",
        ] = None
        diagnostics: Annotated[bool, "Report your model, settings, and limits."] = False
        catalog: Annotated[
            Literal["providers", "models"] | None,
            "List the providers, or one provider's models.",
        ] = None
        catalog_provider: Annotated[
            str | None,
            "Whose models 'catalog: models' lists; yours by default.",
        ] = None

    name = "AgentSelf"
    description = (files(__package__) / "assets" / "agentself.md").read_text(
        encoding="utf-8",
    )
    schema = schema_of(Arguments)

    team: Team = field(repr=False)
    """Where the caller is found, and the catalog it may switch within."""

    display: ToolDisplay = field(default=ToolDisplay())

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        """Return ``AgentSelf`` and the keys set, such as ``AgentSelf status model``."""
        keys = [k for k, v in arguments.items() if v not in (None, False, "")]
        return " ".join(["AgentSelf", *keys])

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        """Return ``"*"``: a patch must not interleave with another."""
        del context, arguments
        return "*"

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        """Plan the patch, apply it whole, and say what changed.

        Args:
          context: The calling agent's state; its label finds the agent.
          arguments: What the model passed, as ``AgentSelf.Arguments``.

        Returns:
          result: One line per change, then any report asked for.

        Raises:
          ValueError: The caller is in no team, or a setting or model is not
            offered; nothing was changed.

        """
        a = parse(arguments, AgentSelf.Arguments)
        if context.label not in self.team:
            raise ValueError("AgentSelf needs a calling agent in the team.")
        agent = self.team[context.label]
        model = await self._model(agent, a)
        target = model if model is not None else agent.model
        settings = target.settings
        try:
            planned = _planned(target.capability, settings, a)
        except ValueError:
            # The agent owns the models assigned to it; one never assigned is ours.
            if model is not None:
                await model.aclose()
            raise
        lines: list[str] = []
        if model is not None:
            before = agent.model
            agent.model = model
            agent.account = self.team.catalog.account
            was, now = (
                ModelTags.from_settings(
                    m.capability,
                    m.settings,
                    provider=m.provider.name,
                ).name
                for m in (before, model)
            )
            lines.append(f"model: {was} -> {now}")
        changes = (
            ("thinking_budget", settings.thinking_budget, planned.thinking_budget),
            ("thinking_output", settings.thinking_output, planned.thinking_output),
            ("thinking_effort", settings.thinking_effort, planned.thinking_effort),
            ("service_tier", settings.service_tier, planned.service_tier),
        )
        lines.extend(
            f"{axis}: {old} -> {new}" for axis, old, new in changes if old != new
        )
        target.settings = planned
        if a.status is not None:
            agent.runtime.publish(types.StatusChanged(text=a.status))
            lines.append(f"status: {a.status}")
        if a.diagnostics:
            lines.extend(_diagnostics(agent, label=context.label))
        if a.catalog == "providers":
            lines.append("providers: " + ", ".join(self.team.catalog.factories))
        elif a.catalog == "models":
            provider = a.catalog_provider or agent.model.provider.name
            lines.append(
                f"{provider} models: " + ", ".join(self.team.catalog[provider]),
            )
        return UncategorizedToolResult(
            call_id="",
            content="\n".join(lines) or "No change.",
        )

    async def _model(self, agent: Agent, a: AgentSelf.Arguments) -> Model | None:
        """Return the model to switch to, the current choices kept; ``None`` for none."""
        if a.provider is None and a.model is None:
            return None
        current = agent.model
        catalog = self.team.catalog
        return await successor(
            partial(catalog.model, current=current.provider.name)
            if a.provider is None
            else catalog[a.provider].__getitem__,
            a.model if a.model is not None else current.capability.model_id,
            current=current,
            model_cache=agent.model_cache,
        )


def _planned(
    capability: ModelCapability,
    settings: ModelSettings,
    a: AgentSelf.Arguments,
) -> ModelSettings:
    """Return ``settings`` with the reasoning and serving arguments; raise if not offered."""
    planned = settings
    if a.effort is not None:
        planned = replace(planned, thinking_effort=a.effort)
    if a.budget is not None:
        planned = replace(planned, thinking_budget=a.budget)
    if a.output is not None:
        planned = replace(planned, thinking_output=a.output)
    if a.service_tier is not None:
        capability.check(replace(planned, service_tier=a.service_tier))
        planned = capability.fit(planned, was=capability, service_tier=a.service_tier)
    capability.check(planned)
    return planned


def _diagnostics(agent: Agent, *, label: str) -> list[str]:
    """Return what the agent runs on: label, model, settings, limits, directory."""
    model = agent.model
    s = model.settings
    limits = model.capability.limits
    return [
        f"label: {label}",
        f"model: {ModelTags.from_settings(model.capability, s, provider=model.provider.name).name}",
        f"context: {s.context_size.display_name}",
        (
            f"thinking: budget={s.thinking_budget} output={s.thinking_output}"
            f" effort={s.thinking_effort}"
        ),
        f"service_tier: {s.service_tier}; cache_ttl_sec: {s.cache_ttl_sec:g}",
        f"limits: request={limits.max_request_tokens} response={limits.max_response_tokens}",
        f"cwd: {agent.runtime.tool_context.cwd}",
        f"tools: {', '.join(agent.tools)}",
    ]
