"""Long-lived headless Sagent daemon and RPC protocol process.

This is a transport adapter, not another agent implementation. Sagent owns the
model loop, tools, team, journal, and runtime; this module only turns requests
into calls on those objects and turns runtime events into frontend notifications.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast

import argparse
import asyncio
import contextlib
import sys

from treekle import json_dumps, json_loads
from wesearch.fetch import FetchSession

from sagent.agent.agent import Agent, default_system_prompt
from sagent.agent.agents_md import AgentsMdConfig
from sagent.agent.compactor import SummaryCompactor
from sagent.agent.default_tools import build_tools, preload_builtin_tools
from sagent.agent.pruner import Pruner
from sagent.agent.team import Team
from sagent.providers.catalog import ModelCatalog, default_provider
from sagent.providers.types import HasLogin
from sagent.runtime import types
from sagent.session.jsonl import JsonlJournal
from sagent.session.projects import list_sessions, new_session_path, project_dir
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    FileEditResult,
    FileWriteResult,
    Thinking,
    ToolCall,
    ToolResult,
    TurnContext,
    UserMessage,
)
from sagent.tools.agent_self import AgentSelf
from sagent.tools.skill import Skill
from sagent.server.commands import SagentCommandService

if TYPE_CHECKING:
    from treekle import PlainTree

    from sagent.model.capability import ModelCapability, ModelSettingName
    from sagent.tui.types import CommandOutput


__all__ = ["SagentServer", "main", "serve"]


def _resolve_prompt_text(prompt: str | None, cwd: Path) -> str | None:
    """Resolve prompt text, reading from a file if the path exists."""
    if prompt is None:
        return None
    stripped = prompt.strip()
    if not stripped:
        return ""
    candidate_path = Path(stripped)
    if not candidate_path.is_absolute():
        candidate_path = cwd / candidate_path
    if candidate_path.is_file():
        try:
            return candidate_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return stripped


_resolve_extra_prompt = _resolve_prompt_text


class SagentServer:
    """Own one Sagent session and expose it through request/notification messages."""

    def __init__(
        self,
        *,
        cwd: Path,
        model_name: str,
        allowed_providers: str = "",
        tool_names: Sequence[str] | None = None,
        system_prompt: str | None = None,
        append_system_prompt: str = "",
        extra_prompt: str = "",
        skill_directories: Sequence[Path] = (),
        output: asyncio.Queue[dict[str, object]],
    ) -> None:
        # These tool modules have existing Agent/Team import dependencies. Load
        # them before the session's async resources so the runtime sees the same
        # initialization order as the terminal surface.
        preload_builtin_tools()
        self.cwd = cwd
        self.model_name = model_name
        self.allowed_providers = allowed_providers
        self.tool_names = tuple(tool_names) if tool_names is not None else None
        self.system_prompt = _resolve_prompt_text(system_prompt, cwd)
        prompt_to_append = append_system_prompt or extra_prompt
        self.append_system_prompt = _resolve_prompt_text(prompt_to_append, cwd) or ""
        self.skill_directories = tuple(
            path if path.is_absolute() else cwd / path
            for path in skill_directories
        )
        self.output = output
        self._resources: AsyncExitStack | None = None
        self._agent: Agent | None = None
        self._team: Team | None = None
        self._command_service: SagentCommandService | None = None
        self._session_path: Path | None = None
        self._observer: Any = None
        self._turn: asyncio.Task[str] | None = None
        self._turn_sequence = 0
        self._active_turn_id: str | None = None
        self._active_agent_label: str | None = None
        self._notification_agent_label: str | None = None

    async def handle(self, request: Mapping[str, object]) -> None:
        """Dispatch one decoded request and enqueue its response, if it has an id."""
        request_id = request.get("id")
        try:
            method = _required_string(request, "method")
            params = _required_object(request.get("params", {}), "params")
            result = await self._dispatch(method, params)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 -- protocol errors are returned to the client.
            if request_id is not None:
                self._respond_error(request_id, str(error))
            return
        if request_id is not None:
            self._respond(request_id, result)

    async def _dispatch(
        self,
        method: str,
        params: Mapping[str, object],
    ) -> dict[str, object]:
        if method == "initialize":
            return {
                "protocol": "sagent/v1",
                "capabilities": {
                    "streaming": True,
                    "sessions": True,
                    "team": True,
                    "agentSessions": True,
                    "modelSettings": True,
                    "commands": True,
                    "nativeWorkspace": True,
                    "skills": True,
                },
            }
        if method == "getSkills":
            return self._get_skills()
        if method == "openSession":
            return await self._open_session(params)
        if method == "listSessions":
            return self._list_sessions()
        if method == "getSession":
            return self._session_state()
        if method == "getTeam":
            return self._team_state()
        if method == "getAgentSession":
            return self._agent_session_state(params)
        if method == "getModelSettings":
            return self._model_settings(params)
        if method == "setModelSettings":
            return await self._set_model_settings(params)
        if method == "getCommands":
            return await self._get_commands(params)
        if method == "executeCommand":
            return await self._execute_command(params)
        if method == "sendMessage":
            return await self._send_message(params)
        if method == "cancelTurn":
            return self._cancel_turn()
        if method == "closeSession":
            await self.close()
            return {"closed": True}
        if method == "shutdown":
            await self.close()
            return {"stopped": True}
        raise ValueError(f"unknown method {method!r}")

    def _get_skills(self) -> dict[str, object]:
        """Return all discovered skills across project, user, and additional skill directories."""
        skill = Skill(additional_dirs=self.skill_directories)
        skills = skill.discover(self.cwd)
        return {
            "skills": [
                {
                    "name": info.name,
                    "description": info.description,
                    "source": info.source,
                }
                for info in skills
            ]
        }

    def _list_sessions(self) -> dict[str, object]:
        """Project Sagent's native session index for the VS Code navigator.

        ``list_sessions`` already knows how to identify valid Sagent journals,
        read their titles, and order them. The integration must not scan or
        parse journal files a second time just to populate a tree view.
        """
        sessions = list_sessions(project_dir(self.cwd))
        return {
            "activePath": (
                str(self._session_path) if self._session_path is not None else None
            ),
            "sessions": [
                {
                    "id": session.id,
                    "path": str(session.path),
                    "mtime": session.mtime,
                    "title": session.title,
                    "records": session.records,
                    "model": session.model,
                }
                for session in sessions
            ],
        }

    async def _open_session(self, params: Mapping[str, object]) -> dict[str, object]:
        requested = params.get("sessionPath")
        if requested is None:
            path = new_session_path(project_dir(self.cwd))
        elif isinstance(requested, str) and requested:
            path = Path(requested)
            if not path.is_absolute():
                path = self.cwd / path
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        else:
            raise ValueError("sessionPath must be a non-empty string")

        # The journal is exclusively locked. Reopening the active path cannot
        # construct a second session over the current one, and reopening it is
        # already an idempotent request from the client's perspective.
        if self._session_path == path and self._agent is not None:
            return self._session_state()

        model_name = params.get("model", self.model_name)
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model must be a non-empty string")
        tool_names = params.get("tools", self.tool_names)
        if tool_names is not None and (
            isinstance(tool_names, (str, bytes))
            or not isinstance(tool_names, Sequence)
            or not all(isinstance(name, str) for name in tool_names)
        ):
            raise ValueError("tools must be an array of names")

        resources = AsyncExitStack()
        await resources.__aenter__()
        try:
            catalog = await resources.enter_async_context(ModelCatalog())
            if self.allowed_providers:
                catalog.allow(_comma_list(self.allowed_providers))
            team = await resources.enter_async_context(Team(catalog=catalog))
            fetch = await resources.enter_async_context(FetchSession())
            journal = await resources.enter_async_context(JsonlJournal(path))
            model = catalog.model(model_name, current=default_provider())
            system: str | None = self.system_prompt
            if self.append_system_prompt:
                base = system if system is not None else default_system_prompt()
                system = f"{base}\n\n{self.append_system_prompt}"

            agent = Agent(
                model=model,
                tools=build_tools(
                    cwd=self.cwd,
                    team=team,
                    session=fetch,
                    names=cast("Sequence[str] | None", tool_names),
                    skill_directories=self.skill_directories,
                ),
                journal=journal,
                compactor=SummaryCompactor,
                pruner=Pruner,
                cwd=self.cwd,
                agents_md=AgentsMdConfig(),
                system_prompt=system,
                persistent_retry=True,
            )
            await agent.recall(catalog)
            await resources.enter_async_context(agent)
            team.join(agent, label="Agent")

            def observe_root(event: types.RuntimeEvent) -> None:
                self._observe(event, "Agent")

            observer = observe_root
            agent.runtime.observers.append(observer)
        except BaseException:
            await resources.aclose()
            raise

        # Build the replacement completely before closing the current session.
        # A bad model, unavailable provider, journal lock, or tool construction
        # failure must leave an already-open session usable.
        try:
            await self.close()
        except BaseException:
            await resources.aclose()
            raise

        self._resources = resources
        self._agent = agent
        self._team = team
        self._session_path = path
        self._observer = observer
        self._command_service = SagentCommandService(
            agent=agent,
            team=team,
            catalog=catalog,
            output=self._emit_command_output,
            focus=self._focus_agent,
        )
        snapshot = self._session_snapshot()
        self._turn_sequence = _snapshot_turn_count(snapshot)
        return self._session_state(snapshot=snapshot)

    async def _send_message(self, params: Mapping[str, object]) -> dict[str, object]:
        team = self._team
        if team is None:
            raise RuntimeError("no Sagent team is open")
        if self._turn is not None and not self._turn.done():
            raise RuntimeError("a turn is already running")
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        agent_label = params.get("agentLabel", "Agent")
        if not isinstance(agent_label, str) or not agent_label:
            raise ValueError("agentLabel must be a non-empty string")
        if agent_label not in team:
            raise ValueError(f"unknown agent {agent_label!r}")
        agent = team[agent_label]
        self._turn_sequence += 1
        turn_id = f"turn-{self._turn_sequence}"
        self._active_turn_id = turn_id
        self._active_agent_label = agent_label
        turn_observer = None
        if agent_label != "Agent":

            def observe_turn(event: types.RuntimeEvent) -> None:
                self._observe(event, agent_label)

            turn_observer = observe_turn
        if turn_observer is not None:
            agent.runtime.observers.append(turn_observer)
        self._notify(
            "turn.started",
            {
                "turnId": turn_id,
                "text": text,
                "agentLabel": agent_label,
                "timestamp": datetime.now(UTC).isoformat(),
                **_agent_model_identity(agent),
            },
        )
        self._turn = asyncio.create_task(agent.send(text))
        try:
            reply = await self._turn
        except asyncio.CancelledError:
            self._notify(
                "turn.cancelled",
                {"turnId": turn_id, "agentLabel": agent_label},
            )
            return {"cancelled": True}
        except Exception as error:
            self._notify(
                "turn.failed",
                {
                    "turnId": turn_id,
                    "message": str(error),
                    "agentLabel": agent_label,
                },
            )
            raise
        finally:
            self._turn = None
            self._active_turn_id = None
            self._active_agent_label = None
            if turn_observer is not None and turn_observer in agent.runtime.observers:
                agent.runtime.observers.remove(turn_observer)
        self._notify(
            "turn.completed",
            {"turnId": turn_id, "text": reply, "agentLabel": agent_label},
        )
        return {"text": reply}

    def _cancel_turn(self) -> dict[str, object]:
        active_label = self._active_agent_label
        agent = (
            self._team[active_label]
            if self._team is not None
            and active_label is not None
            and active_label in self._team
            else self._agent
        )
        if self._turn is None or self._turn.done() or agent is None:
            return {"cancelled": False}
        agent.runtime.halt()
        return {"cancelled": True}

    async def close(self) -> None:
        """Close the live agent and all resources in their dependency order."""
        turn = self._turn
        self._turn = None
        if turn is not None and not turn.done():
            active_label = self._active_agent_label
            active_agent = (
                self._team[active_label]
                if self._team is not None
                and active_label is not None
                and active_label in self._team
                else self._agent
            )
            if active_agent is not None:
                active_agent.runtime.halt()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await turn
        agent = self._agent
        observer = self._observer
        if agent is not None and observer in agent.runtime.observers:
            agent.runtime.observers.remove(observer)
        resources = self._resources
        if resources is not None:
            try:
                await resources.aclose()
            except BaseException:
                # Keep the references intact if cleanup itself fails. The
                # caller may still need to report or retry closing this
                # session, and replacement cleanup must not erase its handle.
                if (
                    agent is not None
                    and observer is not None
                    and observer not in agent.runtime.observers
                ):
                    agent.runtime.observers.append(observer)
                raise
        self._agent = None
        self._resources = None
        self._team = None
        self._command_service = None
        self._session_path = None
        self._observer = None
        self._active_turn_id = None
        self._active_agent_label = None
        self._notification_agent_label = None

    async def _get_commands(self, params: Mapping[str, object]) -> dict[str, object]:
        service = self._command_service
        if service is None:
            raise RuntimeError("no Sagent session is open")
        label = _agent_label(params)
        return {
            "agentLabel": label,
            "commands": await service.descriptors(label),
        }

    async def _execute_command(
        self,
        params: Mapping[str, object],
    ) -> dict[str, object]:
        service = self._command_service
        if service is None:
            raise RuntimeError("no Sagent session is open")
        line = params.get("line")
        if not isinstance(line, str) or not line.strip():
            raise ValueError("line must be a non-empty string")
        label = _agent_label(params)
        done = await service.execute_line(label=label, line=line)
        return {"done": done}

    def _emit_command_output(self, label: str, output: CommandOutput) -> None:
        self._notify(
            "command.output",
            {
                "agentLabel": label,
                "text": output.text,
                "tone": output.tone,
                "command": output.command,
            },
        )

    def _focus_agent(self, label: str) -> None:
        self._notify("agent.focused", {"agentLabel": label})

    def _observe(self, event: types.RuntimeEvent, agent_label: str) -> None:
        self._notification_agent_label = agent_label
        try:
            self._emit_observed_event(event, agent_label)
        finally:
            self._notification_agent_label = None

    def _emit_observed_event(
        self,
        event: types.RuntimeEvent,
        agent_label: str,
    ) -> None:
        """Project Sagent events without changing the runtime's source of truth.

        The TUI receives both live runtime events and resumed session records. The
        VS Code client follows the same rule: event payloads identify stable Sagent
        records (especially tool ``callId`` values), while the client owns only
        presentation state.
        """
        turn_id = (
            self._active_turn_id if agent_label == self._active_agent_label else None
        )
        if isinstance(event, UserMessage):
            self._notify(
                "message.created",
                {
                    "id": _event_id(event),
                    "kind": "user",
                    "turnId": turn_id,
                    "text": event.content or "",
                    **_record_metadata(event),
                },
            )
        elif isinstance(event, AgentToAgentMessage):
            self._notify(
                "message.created",
                {
                    "id": _event_id(event),
                    "kind": "agent",
                    "turnId": turn_id,
                    "text": event.content or "",
                    "sender": event.sender,
                    "recipient": event.recipient,
                    **_record_metadata(event),
                },
            )
        elif isinstance(event, types.ModelCallStarted):
            self._notify(
                "status.changed",
                {"status": "working", "text": "working", "phase": "thinking"},
            )
        elif isinstance(event, types.ModelResponsePartial):
            self._notify(
                "assistant.delta",
                {
                    "turnId": turn_id,
                    "text": event.text,
                    "record": event.record,
                },
            )
        elif isinstance(event, types.ModelResponseThinking):
            self._notify(
                "assistant.thinking",
                {
                    "turnId": turn_id,
                    "text": event.text,
                    "record": event.record,
                },
            )
        elif isinstance(event, types.ModelResponseComplete):
            for record in event.response.records:
                if isinstance(record, AssistantMessage):
                    metadata = _record_metadata(
                        record,
                        fallback_timestamp=datetime.now(UTC).isoformat(),
                    )
                    self._notify(
                        "assistant.completed",
                        {
                            "turnId": turn_id,
                            "record": _event_id(record),
                            "text": record.content or "",
                            **metadata,
                        },
                    )
                elif isinstance(record, Thinking):
                    self._notify(
                        "assistant.thinking.completed",
                        {
                            "turnId": turn_id,
                            "record": _event_id(record),
                            "text": record.content or "",
                        },
                    )
                elif isinstance(record, ToolCall):
                    if record.call_id.endswith(types.Markers().arrival):
                        continue
                    source_agent = self._team[agent_label] if self._team else None
                    tool = source_agent.tools.get(record.name) if source_agent else None
                    self._notify(
                        "tool.started",
                        {
                            "turnId": turn_id,
                            "callId": record.call_id,
                            "name": record.name,
                            "arguments": dict(record.arguments),
                            "summary": tool.summary(record.arguments)
                            if tool
                            else record.name,
                            "status": "running",
                        },
                    )
        elif hasattr(types, "ToolCallStarted") and isinstance(event, types.ToolCallStarted):
            source_agent = self._team[agent_label] if self._team else None
            tool = source_agent.tools.get(event.call.name) if source_agent else None
            summary = (
                event.title
                or (tool.summary(event.call.arguments) if tool else None)
                or str(event.call.arguments.get("toolSummary") or "")
                or str(event.call.arguments.get("CommandLine") or "")
                or event.call.name
            )
            self._notify(
                "tool.started",
                {
                    "turnId": turn_id,
                    "callId": event.call.call_id,
                    "name": event.call.name,
                    "arguments": dict(event.call.arguments),
                    "summary": summary,
                    "status": "running",
                    **(
                        {"agentLabel": self._notification_agent_label}
                        if self._notification_agent_label
                        else {}
                    ),
                },
            )
        elif isinstance(event, types.DetachedResult):
            self._notify("tool.completed", self._tool_result_params(event.result))
        elif isinstance(event, ToolResult):
            self._notify("tool.completed", self._tool_result_params(event))
        elif isinstance(event, types.ChildEvent):
            self._notify(
                "agent.event",
                {
                    "label": event.label,
                    "event": _event_name(event.inner),
                    **self._child_event_params(event.inner),
                },
            )
        elif isinstance(event, types.ChildDoneEvent):
            self._notify(
                "agent.completed",
                {
                    "label": event.label,
                    "elapsed": event.elapsed,
                    "tokens": event.tokens,
                    "cost": event.cost,
                },
            )
        elif isinstance(event, types.StatusChanged):
            self._notify("status.changed", {"status": "notice", "text": event.text})
        elif isinstance(event, types.AgentIdle):
            self._notify("status.changed", {"status": "ready", "text": "ready"})
        elif isinstance(event, types.CohortStarted):
            self._notify(
                "status.changed",
                {"status": "tool", "text": "running tools", "phase": "tools"},
            )
        elif isinstance(event, types.ModelCallSuspended):
            self._notify(
                "retry.waiting",
                {
                    "attempt": event.attempt,
                    "until": event.until,
                    "message": str(event.exception),
                },
            )
        elif isinstance(event, types.CompactStarted):
            self._notify("compaction.started", {})
        elif isinstance(event, types.CompactComplete):
            self._notify(
                "compaction.completed",
                {
                    "tokenBefore": event.token_before,
                    "tokenAfter": event.token_after,
                    "payloadEntries": event.payload_entries,
                    "fallbackReason": event.fallback_reason,
                    "preservedTailCount": event.preserved_tail_count,
                },
            )
        elif isinstance(event, types.CompactFailed):
            self._notify(
                "compaction.failed",
                {"message": str(event.exception), "tapeLength": event.tape_len},
            )
        elif isinstance(event, types.ModelResponseCancelled):
            self._notify(
                "status.changed",
                {"status": "cancelled", "text": "cancelled"},
            )
        elif isinstance(event, types.ModelResponseError):
            self._notify(
                "turn.failed",
                {"turnId": turn_id, "message": str(event.exception)},
            )

    def _tool_result_params(self, result: ToolResult) -> dict[str, object]:
        """Return one stable lifecycle update for a Sagent tool result."""
        call_id = result.call_id.removesuffix(types.Markers().arrival)
        path = (
            result.path
            if isinstance(result, (FileEditResult, FileWriteResult))
            else None
        )
        status = (
            "detached"
            if result.stub == "pending"
            else "cancelled"
            if result.stub == "cancelled"
            else "failed"
            if result.is_error is True
            else "completed"
        )
        return {
            "callId": call_id,
            "resultCallId": result.call_id,
            "status": status,
            "text": result.text(),
            "isError": result.is_error,
            **({} if result.stub is None else {"stub": result.stub}),
            **({} if path is None else {"path": path}),
            **_file_change_payload(result),
        }

    def _child_event_params(self, event: types.RuntimeEvent) -> dict[str, object]:
        """Return useful child details without flattening child state into the UI."""
        if isinstance(event, types.ModelResponsePartial | types.ModelResponseThinking):
            return {"text": event.text, "record": event.record}
        if hasattr(types, "ToolCallStarted") and isinstance(event, types.ToolCallStarted):
            return {"kind": "tool", "callId": event.call.call_id, "name": event.call.name}
        if isinstance(event, types.ModelResponseComplete):
            return {
                "records": [
                    {
                        "kind": "tool" if isinstance(record, ToolCall) else "assistant",
                        **(
                            {"callId": record.call_id, "name": record.name}
                            if isinstance(record, ToolCall)
                            else {"text": record.content or ""}
                        ),
                    }
                    for record in event.response.records
                    if isinstance(record, (ToolCall, AssistantMessage))
                ],
            }
        if isinstance(event, ToolResult):
            return self._tool_result_params(event)
        if isinstance(event, types.ModelResponseError):
            return {"message": str(event.exception)}
        if isinstance(event, types.ModelCallSuspended):
            return {
                "attempt": event.attempt,
                "until": event.until,
                "message": str(event.exception),
            }
        return {}

    def _session_state(
        self,
        *,
        snapshot: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if snapshot is None and self._agent is not None:
            snapshot = self._session_snapshot()
        return {
            "open": self._agent is not None,
            "path": str(self._session_path) if self._session_path is not None else None,
            "sessionId": (
                self._session_path.parent.name
                if self._session_path is not None
                else None
            ),
            "tools": list(self._agent.tools) if self._agent is not None else [],
            "running": self._agent.running if self._agent is not None else False,
            "snapshot": snapshot
            if snapshot is not None
            else {"version": 1, "items": []},
        }

    def _team_state(self) -> dict[str, object]:
        """Project the live native Team without creating a second team model."""
        team = self._team
        if team is None:
            return {"open": False, "agents": []}
        return {
            "open": True,
            "agents": [self._agent_state(team, label) for label in team],
        }

    def _agent_state(self, team: Team, label: str) -> dict[str, object]:
        """Return stable display data from one native Team member."""
        agent = team[label]
        journal_path = agent.session.journal.path
        parent = team.parent(label)
        return {
            "label": label,
            "parent": parent,
            "above": team.above(label),
            "children": team.children(label),
            "watching": team.watched(label),
            "persistent": label in team.persistent(),
            "running": agent.running,
            "conversationId": agent.runtime.conversation,
            "sessionPath": str(journal_path) if journal_path is not None else None,
            "model": agent.model.capability.model_id,
            "provider": agent.model.provider.name,
            "account": agent.account,
            "tools": list(agent.tools),
            "allowedTools": sorted(agent.allowed)
            if agent.allowed is not None
            else None,
            "spendUsd": agent.spend_usd,
            "maxSpendUsd": agent.max_spend_usd,
            "owedReplies": team.owes(label),
        }

    def _agent_session_state(self, params: Mapping[str, object]) -> dict[str, object]:
        """Return one native agent's journal-backed session snapshot."""
        team = self._team
        if team is None:
            raise RuntimeError("no Sagent team is open")
        label = params.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string")
        if label not in team:
            raise ValueError(f"unknown agent {label!r}")
        agent = team[label]
        journal_path = agent.session.journal.path
        return {
            "label": label,
            "sessionId": agent.runtime.conversation,
            "path": str(journal_path) if journal_path is not None else None,
            "snapshot": self._session_snapshot(agent),
        }

    def _model_settings(self, params: Mapping[str, object]) -> dict[str, object]:
        """Project the selected native Agent's catalog and capability choices.

        The response is deliberately shaped from the live Sagent objects. The
        extension never maintains a provider/model registry or guesses which
        thinking settings a model supports.
        """
        agent = self._agent_for_label(params)
        team = self._require_team()
        model = agent.model
        capability = model.capability
        settings = model.settings
        provider = model.provider
        provider_name = provider.name
        accounts = ["default"]
        if isinstance(provider, HasLogin):
            accounts.extend(account for account in provider.logins.accounts() if account != "default")
        acp_servers = ["agy"] if provider_name == "acp" else []
        current_acp_server = (
            getattr(model, "acp_server", "agy") if provider_name == "acp" else ""
        )
        context_sizes = tuple(
            _context_size_payload(size, is_default=size == capability.default.context_size)
            for size in capability.prices.sizes(settings.service_tier)
        )
        return {
            "status": "ready",
            "agentLabel": agent.runtime.tool_context.label,
            "availableScopes": ["current-agent"],
            "selectedScope": "current-agent",
            "provider": {
                "providers": list(team.catalog.factories),
                "currentProvider": provider_name,
                "accounts": accounts,
                "currentAccount": agent.account or team.catalog.account or "default",
                "acpServers": acp_servers,
                "currentAcpServer": current_acp_server,
            },
            "model": {
                "models": list(provider),
                "currentModel": capability.model_id,
            },
            "capability": {
                "modelId": capability.model_id,
                "contextSizes": list(context_sizes),
                "thinkingEfforts": _setting_values(capability, "thinking_effort"),
                "thinkingBudgets": _setting_values(capability, "thinking_budget"),
                "thinkingOutputs": _setting_values(capability, "thinking_output"),
                "serviceTiers": _setting_values(capability, "service_tier"),
                "cacheTtlSeconds": _setting_values(capability, "cache_ttl_sec"),
                "manageContextServerSide": _setting_values(
                    capability,
                    "manage_context_server_side",
                ),
                "defaults": {
                    "context": _context_size_payload(
                        capability.default.context_size,
                        is_default=True,
                    ),
                    "thinkingEffort": capability.default.thinking_effort,
                    "thinkingBudget": capability.default.thinking_budget,
                    "thinkingOutput": capability.default.thinking_output,
                    "serviceTier": capability.default.service_tier,
                    "cacheTtlSeconds": capability.default.cache_ttl_sec,
                    "manageContextServerSide": capability.default.manage_context_server_side,
                },
                "limits": {
                    "maxRequestTokens": capability.limits.max_request_tokens,
                    "maxResponseTokens": capability.limits.max_response_tokens,
                    "maxContextTokens": capability.limits.context_tokens,
                },
            },
            "settings": {
                "context": _context_size_payload(settings.context_size),
                "thinkingEffort": settings.thinking_effort,
                "thinkingBudget": settings.thinking_budget,
                "thinkingOutput": settings.thinking_output,
                "serviceTier": settings.service_tier,
                "cacheTtlSeconds": settings.cache_ttl_sec,
                "manageContextServerSide": settings.manage_context_server_side,
            },
            "writableSettings": [
                "thinking.effort",
                "thinking.budget",
                "thinking.output",
                "service_tier",
            ],
            "diagnostics": {
                "label": agent.runtime.tool_context.label,
                "model": f"{provider_name}/{capability.model_id}",
                "context": settings.context_size.display_name,
                "thinking": {
                    "budget": settings.thinking_budget,
                    "output": settings.thinking_output,
                    "effort": settings.thinking_effort,
                },
                "serviceTier": settings.service_tier,
                "cacheTtlSeconds": settings.cache_ttl_sec,
                "limits": {
                    "maxRequestTokens": capability.limits.max_request_tokens,
                    "maxResponseTokens": capability.limits.max_response_tokens,
                    "maxContextTokens": capability.limits.context_tokens,
                },
                "cwd": str(agent.runtime.tool_context.cwd),
                "tools": list(agent.tools),
            },
        }

    async def _set_model_settings(self, params: Mapping[str, object]) -> dict[str, object]:
        """Apply a supported settings change through Sagent's AgentSelf tool.

        AgentSelf owns model succession and capability validation. The VS Code
        adapter only translates the typed UI intent into its existing argument
        shape; it does not reproduce that logic.
        """
        scope = params.get("scope", "current-agent")
        if scope != "current-agent":
            raise ValueError("only current-agent settings are supported")
        agent = self._agent_for_label(params)
        action = _required_object(params.get("action"), "action")
        arguments = _model_settings_arguments(action)
        if arguments:
            tool = agent.tools.get("AgentSelf")
            if not isinstance(tool, AgentSelf):
                raise RuntimeError("AgentSelf is not enabled for this Sagent agent")
            await tool.call(
                agent.runtime.tool_context,
                cast("Mapping[str, PlainTree]", arguments),
            )
        return self._model_settings({"agentLabel": agent.runtime.tool_context.label})

    def _session_snapshot(self, agent: Agent | None = None) -> dict[str, object]:
        """Project the current Sagent session window using the TUI's record rules."""
        target_agent = agent if agent is not None else self._require_agent()
        items: list[dict[str, object]] = []
        calls: dict[str, dict[str, object]] = {}
        effective_model: str | None = None
        turn = 0
        for index, record in enumerate(target_agent.session):
            if isinstance(record, TurnContext):
                if record.model is not None:
                    effective_model = record.model
                continue
            if isinstance(record, (UserMessage, AgentToAgentMessage)):
                turn += 1
                item = {
                    "id": f"record-{index}",
                    "type": "agent"
                    if isinstance(record, AgentToAgentMessage)
                    else "user",
                    "turnId": f"turn-{turn}",
                    "text": record.content or "",
                    **_record_metadata(record),
                }
                if effective_model is not None:
                    item["model"] = effective_model
                if isinstance(record, AgentToAgentMessage):
                    item.update(
                        {"sender": record.sender, "recipient": record.recipient}
                    )
                items.append(item)
            elif isinstance(record, AssistantMessage):
                if (text := record.content or "").strip():
                    items.append(
                        {
                            "id": f"record-{index}",
                            "type": "assistant",
                            "turnId": f"turn-{turn}",
                            "text": text,
                            **_record_metadata(record),
                        },
                    )
            elif isinstance(record, Thinking):
                if (text := record.content or "").strip():
                    items.append(
                        {
                            "id": f"record-{index}",
                            "type": "thinking",
                            "turnId": f"turn-{turn}",
                            "text": text,
                            **_record_metadata(record),
                        },
                    )
            elif isinstance(record, ToolCall):
                if record.call_id.endswith(types.Markers().arrival):
                    continue
                tool = target_agent.tools.get(record.name)
                summary = (
                    str(record.extra.get("title") or "")
                    or (tool.summary(record.arguments) if tool else None)
                    or str(record.arguments.get("toolSummary") or "")
                    or str(record.arguments.get("CommandLine") or "")
                    or record.name
                )
                item = {
                    "id": f"tool-{record.call_id}",
                    "type": "tool",
                    "turnId": f"turn-{turn}",
                    "callId": record.call_id,
                    "name": record.name,
                    "summary": summary,
                    "arguments": dict(record.arguments),
                    "status": "running",
                    **_record_metadata(record),
                }
                items.append(item)
                calls[record.call_id] = item
            elif isinstance(record, ToolResult):
                call_id = record.call_id.removesuffix(types.Markers().arrival)
                item = calls.get(call_id)
                if item is None:
                    # Match TUI replay: an orphaned successful result is not a
                    # conversation item; an orphaned failure remains visible.
                    if record.is_error is not True:
                        continue
                    orphaned_item: dict[str, object] = {
                        "id": f"tool-{call_id}",
                        "type": "tool",
                        "turnId": f"turn-{turn}",
                        "callId": call_id,
                        "name": "unknown",
                        "summary": "Tool result",
                        "status": "failed",
                    }
                    items.append(orphaned_item)
                    calls[call_id] = orphaned_item
                    item = orphaned_item
                item.update(self._tool_result_params(record))
        return {"version": 1, "items": items}

    def _require_agent(self) -> Agent:
        if self._agent is None:
            raise RuntimeError("no Sagent session is open")
        return self._agent

    def _require_team(self) -> Team:
        if self._team is None:
            raise RuntimeError("no Sagent team is open")
        return self._team

    def _agent_for_label(self, params: Mapping[str, object]) -> Agent:
        team = self._require_team()
        label = params.get("agentLabel", "Agent")
        if not isinstance(label, str) or not label:
            raise ValueError("agentLabel must be a non-empty string")
        if label not in team:
            raise ValueError(f"unknown agent {label!r}")
        return team[label]

    def _notify(self, event: str, params: Mapping[str, object]) -> None:
        payload = {"type": event, **params}
        agent_label = self._notification_agent_label or self._active_agent_label
        if agent_label is not None:
            payload.setdefault("agentLabel", agent_label)
        self.output.put_nowait({"method": "sagent/event", "params": payload})

    def _respond(self, request_id: object, result: Mapping[str, object]) -> None:
        self.output.put_nowait({"id": request_id, "result": dict(result)})

    def _respond_error(self, request_id: object, message: str) -> None:
        self.output.put_nowait(
            {"id": request_id, "error": {"code": "sagent_error", "message": message}},
        )


async def serve(
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    cwd: Path,
    model_name: str,
    allowed_providers: str = "",
    tool_names: Sequence[str] | None = None,
    system_prompt: str | None = None,
    append_system_prompt: str = "",
    extra_prompt: str = "",
    skill_directories: Sequence[Path] = (),
) -> None:
    """Serve newline-delimited requests until stdin closes or shutdown is requested."""
    output: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    server = SagentServer(
        cwd=cwd,
        model_name=model_name,
        allowed_providers=allowed_providers,
        tool_names=tool_names,
        system_prompt=system_prompt,
        append_system_prompt=append_system_prompt,
        extra_prompt=extra_prompt,
        skill_directories=skill_directories,
        output=output,
    )
    stop = asyncio.Event()

    async def write_messages() -> None:
        while True:
            message = await output.get()
            output_stream.write(json_dumps(message) + "\n")
            output_stream.flush()
            if message.get("method") == "sagent/stop":
                return

    async def dispatch(line: str) -> None:
        try:
            request = _required_object(json_loads(line), "request")
            await server.handle(request)
            if request.get("method") == "shutdown":
                stop.set()
        except Exception as error:  # noqa: BLE001 -- malformed input must not kill the host.
            output.put_nowait(
                {"error": {"code": "invalid_request", "message": str(error)}},
            )

    writer = asyncio.create_task(write_messages())
    pending: set[asyncio.Task[None]] = set()
    lines = _read_lines(input_stream)
    try:
        while not stop.is_set():
            read = asyncio.create_task(anext(lines, None))
            stopped = asyncio.create_task(stop.wait())
            done, _ = await asyncio.wait(
                (read, stopped),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stopped in done:
                read.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await read
                break
            stopped.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stopped
            line = read.result()
            if line is None:
                break
            task = asyncio.create_task(dispatch(line))
            pending.add(task)
            task.add_done_callback(pending.discard)
    finally:
        await lines.aclose()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await server.close()
        output.put_nowait({"method": "sagent/stop", "params": {}})
        await writer


async def _read_lines(input_stream: TextIO) -> AsyncGenerator[str, None]:
    """Read a real pipe without a thread that can survive process shutdown.

    ``StringIO`` keeps protocol tests deterministic. The extension process uses
    the pipe branch, whose transport can be closed when the server receives
    ``shutdown``.
    """
    pipe = getattr(input_stream, "buffer", None)
    if pipe is None:
        while line := input_stream.readline():
            yield line
        return
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_running_loop()
    transport, _ = await loop.connect_read_pipe(lambda: protocol, pipe)
    try:
        async for line in reader:
            yield line.decode()
    finally:
        transport.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the VS Code process surface from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="default")
    parser.add_argument("--allow-providers", default="")
    parser.add_argument("--tools", nargs="+", default=None)
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt replacing the default one, or path to prompt file",
    )
    parser.add_argument(
        "--append-system-prompt",
        "--extra-prompt",
        dest="append_system_prompt",
        default="",
        help="Instructions or path to prompt file to append to the system prompt",
    )
    parser.add_argument(
        "--skill-dir",
        action="append",
        default=[],
        help="Additional directory containing SKILL.md bundles",
    )
    args = parser.parse_args(argv)
    asyncio.run(
        serve(
            input_stream=sys.stdin,
            output_stream=sys.stdout,
            cwd=Path.cwd(),
            model_name=args.model,
            allowed_providers=args.allow_providers,
            tool_names=args.tools,
            system_prompt=args.system_prompt,
            append_system_prompt=args.append_system_prompt,
            skill_directories=tuple(Path(path) for path in args.skill_dir),
        ),
    )
    return 0


def _required_string(request: Mapping[str, object], key: str) -> str:
    value = request.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_object(value: object, name: str) -> Mapping[str, object]:
    """Return a protocol object or explain which envelope was malformed."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return cast("Mapping[str, object]", value)


def _comma_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _agent_label(params: Mapping[str, object]) -> str:
    label = params.get("agentLabel", "Agent")
    if not isinstance(label, str) or not label:
        raise ValueError("agentLabel must be a non-empty string")
    return label


def _agent_model_identity(agent: Agent) -> dict[str, str]:
    """Project the live model that will answer this turn.

    The Agent owns the model object used by its runtime. Reading this at the
    turn boundary keeps the UI label tied to Sagent's execution source rather
    than to a separately maintained settings snapshot.
    """
    model = agent.model
    return {
        "modelProvider": model.provider.name,
        "modelId": model.capability.model_id,
    }


def _setting_values(capability: ModelCapability, name: ModelSettingName) -> list[object]:
    """Return native JSON values from Sagent's capability catalog.

    ``typed_text`` is intentionally not used here: it formats values for the
    terminal command line, while this protocol must preserve numbers and
    booleans for the webview's typed contract.
    """
    return list(capability.values(name))


def _context_size_payload(
    size: object,
    *,
    is_default: bool = False,
) -> dict[str, object]:
    context_name = getattr(size, "context_name", None)
    tokens = getattr(size, "tokens", None)
    display_name = getattr(size, "display_name", None)
    if not isinstance(context_name, str) or not isinstance(tokens, int) or not isinstance(display_name, str):
        raise TypeError("invalid Sagent context size")
    return {
        "name": context_name,
        "tokens": tokens,
        "displayName": display_name,
        "isDefault": is_default,
    }


def _model_settings_arguments(action: Mapping[str, object]) -> dict[str, object]:
    action_type = action.get("type")
    if action_type == "select-provider":
        # A provider does not necessarily serve the model currently selected.
        # Ask AgentSelf to resolve that provider's native default role in the
        # same atomic succession, so the returned snapshot is immediately
        # scoped to the new provider's catalog.
        return {
            "provider": _required_action_string(action, "provider"),
            "model": "default",
        }
    if action_type == "select-acp-server":
        # Reset to default model under the ACP provider when server changes
        return {
            "provider": "acp",
            "model": "default",
        }
    if action_type == "select-model":
        return {"model": _required_action_string(action, "model")}
    if action_type == "set-setting":
        name = action.get("name")
        value = action.get("value")
        if not isinstance(name, str):
            raise ValueError("settings action name must be a string")
        if name not in {
            "thinking.effort",
            "thinking.budget",
            "thinking.output",
            "service_tier",
        }:
            raise ValueError(f"setting {name!r} is not writable through AgentSelf")
        if not isinstance(value, str):
            raise ValueError(f"setting {name!r} requires a string value")
        return {name.rsplit(".", 1)[-1] if name.startswith("thinking.") else name: value}
    if action_type in {"select-scope", "refresh"}:
        return {}
    raise ValueError(f"unknown model settings action {action_type!r}")


def _required_action_string(action: Mapping[str, object], key: str) -> str:
    value = action.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"settings action {key} must be a non-empty string")
    return value


def _event_name(event: types.RuntimeEvent) -> str:
    return type(event).__name__


def _event_id(record: object) -> str | None:
    """Expose Sagent's record context id when the runtime assigned one."""
    context_id = getattr(record, "context_id", None)
    return f"record-{context_id}" if context_id is not None else None


def _record_metadata(
    record: object,
    *,
    fallback_timestamp: str | None = None,
) -> dict[str, object]:
    """Return record metadata, preserving native timestamps when available."""
    metadata: dict[str, object] = {}
    context_id = getattr(record, "context_id", None)
    timestamp = getattr(record, "timestamp", None)
    if context_id is not None:
        metadata["contextId"] = context_id
    if timestamp is not None:
        metadata["timestamp"] = timestamp
    elif fallback_timestamp is not None:
        metadata["timestamp"] = fallback_timestamp
    return metadata


def _snapshot_turn_count(snapshot: Mapping[str, object]) -> int:
    """Return the highest hydrated turn number used by a session snapshot."""
    items = snapshot.get("items", [])
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return 0
    numbers = [
        int(turn_id.removeprefix("turn-"))
        for item in items
        if isinstance(item, Mapping)
        and isinstance(turn_id := item.get("turnId"), str)
        and turn_id.startswith("turn-")
        and turn_id.removeprefix("turn-").isdigit()
    ]
    return max(numbers, default=0)


def _file_change_payload(result: ToolResult) -> dict[str, object]:
    """Serialize Sagent's typed file result without parsing display text.

    This is an adapter projection only. Edit and Write remain implemented by
    Sagent; the VS Code client receives the structured result fields that those
    tools already produce and can render them without depending on Git.
    """
    if isinstance(result, FileEditResult):
        return {
            "fileChange": {
                "kind": "edit",
                "edits": [
                    {
                        "before": splice.before,
                        "after": splice.after,
                        "start": splice.start,
                        "count": splice.count,
                    }
                    for splice in result.edits
                ],
            },
        }
    if isinstance(result, FileWriteResult):
        return {
            "fileChange": {
                "kind": "write",
                "content": result.content,
            },
        }
    return {}
