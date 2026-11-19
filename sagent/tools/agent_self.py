"""AgentSelf: change yourself or a descendant: title, model, or settings.

A patch is planned whole, then applied: an invalid setting rejects all of it.
What the agent's session holds is the AgentSession tool's to change.

Typical usage example:

  me = AgentSelf(team=team)
  result = await settled(me.call(context, {"status": "Fixing the parser", "effort": "high"}))
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Literal, NotRequired, TypedDict, cast

from treekle import PlainTree, from_plain

from sagent.agent.team import Team
from sagent.model.capability import ThinkingBudget, ThinkingEffort, ThinkingOutput
from sagent.model.cost import ServiceTier
from sagent.providers.catalog import counterpart, successor
from sagent.providers.table import ModelTags
from sagent.runtime import types
from sagent.runtime.types import DetachSchedule, WakeSchedule
from sagent.session.types import AgentSelfResult
from sagent.tools.agent_tool import target_agent
from sagent.tools.display import ToolDisplay
from sagent.tools.types import (
    ToolOutput,
    ToolStreamingResult,
    json_schema,
    parse,
    whole_output,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from sagent.agent.agent import Agent
    from sagent.model.capability import (
        ModelCapability,
        ModelSettingName,
        ModelSettings,
    )
    from sagent.providers.types import Model
    from sagent.tools.types import ToolContext


__all__ = [
    "AgentSelf",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSelf:
    """Change yourself or a descendant with your own permissions."""

    @dataclass(frozen=True, slots=True, kw_only=True)
    class Arguments:
        """What the model passes."""

        agent: Annotated[str, "Agent label or relation; omit for yourself."] = ""
        status: Annotated[str | None, "3-7 words naming what the session is about."] = (
            None
        )
        peer_messaging: Annotated[
            bool | None,
            "Participate in cross-session discovery and messaging; on by default.",
        ] = None
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
        detach: Annotated[
            Mapping[str, PlainTree] | None,
            "When long calls wake you: {default: KNOBS, tools: {TOOL: KNOBS}};"
            " KNOBS is {first_sec, growth, max_gap_sec} or {off: true}.",
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
    schema = json_schema(Arguments)

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

    def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolStreamingResult[AgentSelfResult, str, str]:
        """Plan the patch, apply it whole, and say what changed.

        Args:
          context: The calling agent's state; its label finds the agent.
          arguments: What the model passed, as ``AgentSelf.Arguments``.

        Returns:
          streaming_result: One line per change, then any report asked for.

        Raises:
          ValueError: The caller is in no team, or a setting or model is not
            offered; nothing was changed.

        """
        return whole_output(self._run(context, arguments))

    async def _run(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> ToolOutput[AgentSelfResult, str, str]:
        a = parse(arguments, AgentSelf.Arguments)
        agent = target_agent(
            self.team,
            context=context,
            name=self.name,
            agent=a.agent,
        )
        label = agent.runtime.tool_context.label
        model = await self._model(agent, a)
        target = model if model is not None else agent.model
        settings = target.settings
        try:
            planned = _planned(target.capability, settings, a)
            schedule = (
                None
                if a.detach is None
                else _scheduled(agent.runtime.detach_schedule, a.detach)
            )
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
        if schedule is not None:
            agent.runtime.detach_schedule = schedule
            lines.append(f"detach: {_schedule_text(schedule)}")
        if a.peer_messaging is not None:
            self.team.set_peer_messaging(label, a.peer_messaging)
            lines.append(f"peer_messaging: {'on' if a.peer_messaging else 'off'}")
        if a.status is not None:
            agent.runtime.publish(types.StatusChanged(text=a.status))
            lines.append(f"status: {a.status}")
        if a.diagnostics:
            lines.extend(_diagnostics(agent, label=label))
        if a.catalog == "providers":
            lines.append("providers: " + ", ".join(self.team.catalog.factories))
        elif a.catalog == "models":
            provider = a.catalog_provider or agent.model.provider.name
            lines.append(
                f"{provider} models: " + ", ".join(self.team.catalog[provider]),
            )
        return ToolOutput(
            stdout=("\n".join(lines) or "No change.",),
            result=AgentSelfResult(),
        )

    async def _model(self, agent: Agent, a: AgentSelf.Arguments) -> Model | None:
        """Return the model to switch to, the current choices kept; ``None`` for none."""
        if a.provider is None and a.model is None:
            return None
        current = agent.model
        catalog = self.team.catalog
        if a.provider is None:
            build: Callable[[str], Model] = partial(
                catalog.model,
                current=current.provider.name,
            )
            name = a.model
        else:
            provider = catalog[a.provider]
            build = provider.__getitem__
            name = a.model or await counterpart(provider, current)
        return await successor(
            build,
            name or current.capability.model_id,
            current=current,
            model_cache=agent.model_cache,
        )


def _planned(
    capability: ModelCapability,
    settings: ModelSettings,
    a: AgentSelf.Arguments,
) -> ModelSettings:
    """Return ``settings`` with the reasoning and serving arguments; raise if not offered."""
    given: dict[ModelSettingName, object | None] = {
        "thinking_effort": a.effort,
        "thinking_budget": a.budget,
        "thinking_output": a.output,
        "service_tier": a.service_tier,
    }
    planned = capability.with_choices(
        settings,
        {name: value for name, value in given.items() if value is not None},
    )
    capability.check(planned)
    return planned


class _Knobs(TypedDict):
    """One wake schedule as the model patches it: knobs left out keep their value."""

    first_sec: NotRequired[float]
    growth: NotRequired[float]
    max_gap_sec: NotRequired[float]
    off: NotRequired[bool]


class _Detach(TypedDict):
    """A ``detach`` patch: the default schedule, and tools' own."""

    default: NotRequired[_Knobs]
    tools: NotRequired[dict[str, _Knobs]]


def _scheduled(
    schedule: DetachSchedule,
    patch: Mapping[str, PlainTree],
) -> DetachSchedule:
    """Return ``schedule`` with ``patch``; raise, changing nothing, if it does not fit."""
    detach = from_plain(dict(patch), _Detach)
    _refuse_unknown(patch, _Detach, where="detach")
    default_knobs, tools_knobs = patch.get("default"), patch.get("tools")
    if isinstance(default_knobs, Mapping):
        _refuse_unknown(default_knobs, _Knobs, where="detach.default")
    if isinstance(tools_knobs, Mapping):
        for tool, knobs in tools_knobs.items():
            if isinstance(knobs, Mapping):
                _refuse_unknown(knobs, _Knobs, where=f"detach.tools.{tool}")
    default = (
        _knobbed(schedule.default, detach["default"])
        if "default" in detach
        else schedule.default
    )
    tools = dict(schedule.tools)
    for tool, knobs in detach.get("tools", {}).items():
        tools[tool] = _knobbed(tools.get(tool, default), knobs)
    return DetachSchedule(default=default, tools=tools)


# Treekle reads a TypedDict's own keys and leaves others be; a misspelled knob would
# then change nothing, unseen.
def _refuse_unknown(value: Mapping[str, PlainTree], kind: type, *, where: str) -> None:
    """Raise if ``value`` holds a key the TypedDict ``kind`` does not name."""
    known = cast(frozenset[str], vars(kind)["__required_keys__"]) | cast(
        frozenset[str],
        vars(kind)["__optional_keys__"],
    )
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(
            f"{where} takes {', '.join(sorted(known))}; not {', '.join(unknown)}.",
        )


# A schedule that is off is patched from the default one: knobs alone turn it on.
def _knobbed(schedule: WakeSchedule | None, knobs: _Knobs) -> WakeSchedule | None:
    """Return ``schedule`` with ``knobs`` set; ``None`` when they turn it off."""
    if knobs.get("off", False):
        return None
    base = WakeSchedule() if schedule is None else schedule
    return WakeSchedule(
        first_sec=float(knobs.get("first_sec", base.first_sec)),
        growth=float(knobs.get("growth", base.growth)),
        max_gap_sec=float(knobs.get("max_gap_sec", base.max_gap_sec)),
    )


def _schedule_text(schedule: DetachSchedule) -> str:
    """Return ``schedule`` as ``15s 1 300s; Bash off``: the default, then each tool's."""
    return "; ".join(
        f"{name}{' ' if name else ''}{'off' if s is None else s.text()}"
        for name, s in (("", schedule.default), *sorted(schedule.tools.items()))
    )


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
        "tools: "
        + ", ".join(
            name
            for name in agent.tools
            if agent.runtime.allowed is None or name in agent.runtime.allowed
        ),
        f"offered_tools: {', '.join(agent.tools)}",
    ]
