"""Long-lived headless Sagent daemon and RPC protocol process.

This is a transport adapter, not another agent implementation. Sagent owns the
model loop, tools, team, journal, and runtime; this module only turns requests
into calls on those objects and turns runtime events into frontend notifications.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, cast

import argparse
import asyncio
import contextlib
import sys

from treekle import json_dumps, json_loads

from sagent.runtime import types
from sagent.server.commands import SagentCommandService
from sagent.server.model_settings import (
    apply as apply_model_settings,
    snapshot as model_settings_snapshot,
    usage as provider_usage,
)
from sagent.server.protocol import (
    NATIVE_EVENT_METHOD,
    NATIVE_NOTIFICATION_METHOD,
    NATIVE_PROTOCOL,
)
from sagent.server.sessions import ManagedSession, SessionManager
from sagent.server.wire import (
    runtime_event_message,
    session_notification_message,
    session_record_message,
)
from sagent.session.controller import SubmissionMode
from sagent.session.host import AgentSession
from sagent.session.projects import list_sessions, new_session_path, project_dir
from sagent.session.snapshot import SessionSnapshotProjector
from sagent.tools.skill import Skill


if TYPE_CHECKING:
    from sagent.agent.agent import Agent
    from sagent.agent.team import Team
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


def _implementation_info() -> dict[str, str]:
    """Return enough identity for a client to verify which daemon it reached."""
    try:
        package_version = version("sagent")
    except PackageNotFoundError:
        package_version = "unknown"
    return {
        "name": "sagent",
        "version": package_version,
        # A source path is useful for local development, where two checkouts can
        # expose the same package version. This daemon is local-only JSONL RPC.
        "serverPath": str(Path(__file__).resolve()),
    }


class SagentServer:
    """Host independent native sessions through one daemon process."""

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
        self.cwd = cwd
        self.model_name = model_name
        self.allowed_providers = allowed_providers
        self.tool_names = tuple(tool_names) if tool_names is not None else None
        self.system_prompt = _resolve_prompt_text(system_prompt, cwd)
        prompt_to_append = append_system_prompt or extra_prompt
        self.append_system_prompt = _resolve_prompt_text(prompt_to_append, cwd) or ""
        self.skill_directories = tuple(
            path if path.is_absolute() else cwd / path for path in skill_directories
        )
        self.output = output
        self._sessions = SessionManager()
        # These aliases preserve the small private surface used by existing
        # daemon tests and by callers that have only one open session. New
        # protocol requests resolve through SessionManager by sessionId.
        self._session: AgentSession | None = None
        self._command_service: SagentCommandService | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._active_agent_label: str | None = None
        self._notification_agent_label: str | None = None
        self._snapshot_projector = SessionSnapshotProjector()
        self._session_id: str | None = None
        self._event_sequence = 0
        self._subscribed_agent_labels: set[str] = set()
        self._team_watch_task: asyncio.Task[None] | None = None

    @property
    def _agent(self) -> Agent | None:
        return self._session.agent if self._session is not None else None

    @property
    def _team(self) -> Team | None:
        return self._session.team if self._session is not None else None

    @property
    def _session_path(self) -> Path | None:
        return self._session.path if self._session is not None else None

    def _entry_for(self, params: Mapping[str, object]) -> ManagedSession | None:
        """Resolve a request to one session without changing global server state."""
        requested = params.get("sessionId")
        if requested is not None:
            if not isinstance(requested, str) or not requested:
                raise ValueError("sessionId must be a non-empty string")
            entry = self._sessions.get(requested)
            if entry is None:
                raise ValueError(f"unknown session {requested!r}")
            return entry
        if self._session_id is not None:
            entry = self._sessions.get(self._session_id)
            if entry is not None:
                return entry
        if len(self._sessions) == 1:
            return next(self._sessions.values())
        # Private unit tests and old single-session callers may attach an
        # already-configured native session directly. Keep that compatibility
        # path out of the real multi-session registry.
        if self._session is not None and not self._sessions:
            return ManagedSession(
                self._session_id or "legacy",
                self._session,
                command_service=self._command_service,
                active_agent_label=self._active_agent_label,
            )
        return None

    def _require_entry(self, params: Mapping[str, object]) -> ManagedSession:
        entry = self._entry_for(params)
        if entry is None:
            raise RuntimeError("no Sagent session is open")
        return entry

    def _select_entry(self, entry: ManagedSession) -> None:
        """Keep legacy aliases pointed at the most recently used session."""
        self._session = entry.session
        self._session_id = entry.session_id
        self._command_service = entry.command_service
        self._active_agent_label = entry.active_agent_label
        self._notification_agent_label = entry.notification_agent_label
        self._event_sequence = entry.event_sequence
        self._subscribed_agent_labels = entry.subscribed_agent_labels
        self._team_watch_task = entry.team_watch_task

    def _sync_entry_aliases(self, entry: ManagedSession) -> None:
        if self._session_id == entry.session_id or (
            not self._sessions and self._session is entry.session
        ):
            if not self._sessions and self._session_id is None:
                entry.session_id = "legacy"
            self._select_entry(entry)

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
        if method == "system.initialize":
            capabilities = {
                "streaming": True,
                "sessions": True,
                "team": True,
                "agentSessions": True,
                "modelSettings": True,
                "commands": True,
                "nativeWorkspace": True,
                "skills": True,
                "nativeEvents": True,
                "nativeRecords": True,
            }
            return {
                "protocol": NATIVE_PROTOCOL,
                "implementation": _implementation_info(),
                "capabilities": capabilities,
            }
        if method == "catalog.skills":
            return self._get_skills()
        if method == "session.open":
            return await self._open_session(params)
        if method == "session.list":
            return self._list_sessions()
        if method == "session.state":
            return self._session_state(params)
        if method == "session.team.state":
            return self._team_state(params)
        if method == "session.agent.state":
            return self._agent_session_state(params)
        if method == "catalog.modelSettings":
            return await self._model_settings(params)
        if method == "catalog.usage":
            return await self._get_provider_usage(params)
        if method == "catalog.modelSettings.set":
            return await self._set_model_settings(params)
        if method == "commands.list":
            return await self._get_commands(params)
        if method == "commands.execute":
            return await self._execute_command(params)
        if method == "session.records":
            return self._session_records(params)
        if method == "session.submit":
            return self._submit_native(params)
        if method == "session.halt":
            return self._cancel_turn(params)
        if method == "session.close":
            await self._close_session(params)
            return {"closed": True}
        if method == "system.shutdown":
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
            return self._session_state({"sessionId": self._session_id})

        session_id = path.parent.name
        existing = self._sessions.get(session_id)
        if existing is not None:
            if existing.session.path != path:
                raise ValueError(f"session id {session_id!r} is already open")
            self._select_entry(existing)
            return self._session_state({"sessionId": session_id})

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

        session: AgentSession | None = None
        try:
            session = await AgentSession.open(
                cwd=self.cwd,
                path=path,
                model_name=model_name,
                allowed_providers=self.allowed_providers,
                tool_names=cast("Sequence[str] | None", tool_names),
                system_prompt=self.system_prompt,
                append_system_prompt=self.append_system_prompt,
                skill_directories=self.skill_directories,
            )
        except BaseException:
            if session is not None:
                await session.close()
            raise

        assert session is not None
        entry = ManagedSession(session_id=session_id, session=session)
        entry.command_service = SagentCommandService(
            agent=session.agent,
            team=session.team,
            catalog=session.catalog,
            output=lambda label, output: self._emit_command_output(
                session_id, label, output,
            ),
            focus=lambda label: self._focus_agent(session_id, label),
        )
        self._sessions.add(entry)
        self._select_entry(entry)
        self._subscribe_team_events(entry)
        entry.team_watch_task = asyncio.create_task(self._watch_team(entry))
        self._sync_entry_aliases(entry)
        return self._session_state({"sessionId": session_id})

    def _subscribe_agent_events(self, entry: ManagedSession, label: str) -> None:
        remover = entry.session.subscribe(
            lambda agent_label, event: self._forward_runtime_event(
                entry.session_id, agent_label, event,
            ),
            agent_label=label,
        )
        entry.unsubscribers.append(remover)
        entry.subscribed_agent_labels.add(label)

    def _subscribe_team_events(self, entry: ManagedSession | None = None) -> None:
        """Attach the daemon observer to every currently live team agent."""
        entry = entry or self._entry_for({})
        if entry is None:
            return
        for label in tuple(entry.session.team):
            if label not in entry.subscribed_agent_labels:
                self._subscribe_agent_events(entry, label)

    async def _watch_team(self, entry: ManagedSession) -> None:
        """Follow live Team membership like the TUI's periodic view watch."""
        while self._sessions.get(entry.session_id) is entry:
            self._subscribe_team_events(entry)
            await asyncio.sleep(0.2)

    def _forward_runtime_event(
        self, session_id: str, agent_label: str, event: types.RuntimeEvent
    ) -> None:
        self._observe(event, agent_label, session_id=session_id)

    def _notify_usage_soon(self, session_id: str, agent_label: str) -> None:
        async def _notify() -> None:
            with contextlib.suppress(Exception):
                entry = self._sessions.get(session_id)
                if entry is not None and agent_label in entry.session.team:
                    usage = await provider_usage(entry.session.team[agent_label])
                    self._notify(
                        "usage.updated",
                        {
                            "agentLabel": agent_label,
                            **usage,
                        },
                        session_id=session_id,
                    )

        task = asyncio.create_task(_notify())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _submit_native(self, params: Mapping[str, object]) -> dict[str, object]:
        """Submit native runtime input without creating a frontend turn."""
        entry = self._require_entry(params)
        team = entry.session.team
        session = entry.session
        self._subscribe_team_events(entry)
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        agent_label = params.get("agentLabel", "Agent")
        if not isinstance(agent_label, str) or agent_label not in team:
            raise ValueError(f"unknown agent {agent_label!r}")
        mode = params.get("mode", "immediate")
        if mode not in ("immediate", "queued", "deferred"):
            raise ValueError(f"unknown submission mode {mode!r}")
        entry.active_agent_label = agent_label
        self._sync_entry_aliases(entry)
        submission = session.submit(
            text,
            agent_label=agent_label,
            mode=cast(SubmissionMode, mode),
        )
        return {
            "accepted": True,
            "submissionId": submission.id,
            "agentLabel": submission.agent_label,
            "mode": submission.mode,
        }

    def _cancel_turn(
        self, params: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        params = params or {}
        entry = self._entry_for(params)
        if entry is None:
            return {"cancelled": False}
        requested_label = params.get("agentLabel")
        if requested_label is not None and not isinstance(requested_label, str):
            raise TypeError("agentLabel must be a string")
        active_label = requested_label or entry.active_agent_label or "Agent"
        session = entry.session
        if session is None or active_label not in session.agents:
            return {"cancelled": False}
        return {"cancelled": session.halt(agent_label=active_label)}

    async def close(self) -> None:
        """Close every session and all resources in dependency order."""
        for session_id in self._sessions.ids():
            await self._close_session({"sessionId": session_id})
        # Preserve the legacy direct-attachment cleanup path used by callers
        # that never opened through SessionManager.
        if self._session is not None and not self._sessions:
            session = self._session
            if self._active_agent_label and self._active_agent_label in session.agents:
                session.halt(agent_label=self._active_agent_label)
            await session.close()
        self._session = None
        self._command_service = None
        self._session_id = None
        self._event_sequence = 0
        self._subscribed_agent_labels.clear()
        self._active_agent_label = None
        self._notification_agent_label = None

    async def _close_session(self, params: Mapping[str, object]) -> None:
        entry = self._entry_for(params)
        if entry is None:
            return
        if entry.active_agent_label and entry.active_agent_label in entry.session.agents:
            entry.session.halt(agent_label=entry.active_agent_label)
        if entry.team_watch_task is not None:
            entry.team_watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await entry.team_watch_task
            entry.team_watch_task = None
        for unsubscribe in entry.unsubscribers:
            unsubscribe()
        removed = self._sessions.remove(entry.session_id)
        if removed is None:
            # A directly attached legacy session is not in SessionManager.
            await entry.session.close()
        if self._session_id == entry.session_id:
            remaining = next(self._sessions.values(), None)
            if remaining is not None:
                self._select_entry(remaining)
            else:
                self._session = None
                self._command_service = None
                self._session_id = None
                self._event_sequence = 0
                self._subscribed_agent_labels.clear()
                self._active_agent_label = None
                self._notification_agent_label = None

    async def _get_commands(self, params: Mapping[str, object]) -> dict[str, object]:
        entry = self._require_entry(params)
        service = entry.command_service
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
        entry = self._require_entry(params)
        service = entry.command_service
        if service is None:
            raise RuntimeError("no Sagent session is open")
        line = params.get("line")
        if not isinstance(line, str) or not line.strip():
            raise ValueError("line must be a non-empty string")
        label = _agent_label(params)
        done = await service.execute_line(label=label, line=line)
        return {"done": done}

    def _emit_command_output(self, session_id: str, label: str, output: CommandOutput) -> None:
        self._notify(
            "command.output",
            {
                "agentLabel": label,
                "text": output.text,
                "tone": output.tone,
                "command": output.command,
            },
            session_id=session_id,
        )

    def _focus_agent(self, session_id: str, label: str) -> None:
        self._notify("agent.focused", {"agentLabel": label}, session_id=session_id)

    def _observe(
        self,
        event: types.RuntimeEvent,
        agent_label: str,
        *,
        session_id: str | None = None,
    ) -> None:
        session_id = session_id or self._session_id
        entry = self._sessions.get(session_id) if session_id is not None else None
        if entry is not None:
            self._subscribe_team_events(entry)
            entry.notification_agent_label = agent_label
        self._notification_agent_label = agent_label
        try:
            self._emit_observed_event(event, agent_label, session_id=session_id)
        finally:
            if entry is not None:
                entry.notification_agent_label = None
            self._notification_agent_label = None

    def _emit_observed_event(
        self,
        event: types.RuntimeEvent,
        agent_label: str,
        *,
        session_id: str | None = None,
    ) -> None:
        """Forward native runtime events as daemon notifications."""
        entry = self._sessions.get(session_id) if session_id is not None else None
        if entry is not None:
            entry.event_sequence += 1
            sequence = entry.event_sequence
        else:
            self._event_sequence += 1
            sequence = self._event_sequence
        message = runtime_event_message(
            event,
            agent_label=agent_label,
            session_id=session_id or self._session_id,
            sequence=sequence,
        )
        self.output.put_nowait({"method": NATIVE_EVENT_METHOD, "params": message})
        if isinstance(event, types.AgentIdle):
            if session_id is not None:
                self._notify_usage_soon(session_id, agent_label)

    def _session_state(self, params: Mapping[str, object] | None = None) -> dict[str, object]:
        """Return authoritative state from the live native session objects."""
        entry = self._entry_for(params or {})
        session = entry.session if entry is not None else None
        if session is None:
            return {
                "open": False,
                "sessionId": None,
                "snapshot": {"version": 1, "items": []},
                "records": [],
                "team": {"open": False, "agents": []},
            }
        label = session.agent.runtime.tool_context.label
        return {
            "open": True,
            "path": str(session.path) if session.path is not None else None,
            "sessionId": entry.session_id if entry is not None else self._session_id,
            "agentLabel": label,
            "tools": list(session.agent.tools),
            "running": session.agent.running,
            "snapshot": self._session_snapshot(entry, agent_label=label),
            "records": self._native_records(entry, label),
            "agent": self._agent_state(session.team, label),
            "team": self._team_state({"sessionId": entry.session_id} if entry else {}),
        }

    def _team_state(self, params: Mapping[str, object] | None = None) -> dict[str, object]:
        """Project the live native Team without creating a second team model."""
        entry = self._entry_for(params or {})
        team = entry.session.team if entry is not None else None
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
            "runtime": {
                "idle": agent.runtime.is_idle,
                "acceptsUserDispatch": agent.runtime.accepts_user_dispatch,
                "acceptsDeferredDispatch": agent.runtime.accepts_deferred_dispatch,
                "modelCall": agent.runtime.model_call is not None,
                "compacting": agent.runtime.compact_task is not None,
                "cohortSize": len(agent.runtime.cohort),
                "runningTools": len(agent.runtime.running_tools),
                "jobs": list(agent.runtime.jobs),
            },
        }

    def _agent_session_state(self, params: Mapping[str, object]) -> dict[str, object]:
        """Return one live agent's native state and journal records."""
        entry = self._require_entry(params)
        team = entry.session.team
        label = params.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string")
        if label not in team:
            raise ValueError(f"unknown agent {label!r}")
        agent = team[label]
        journal_path = agent.session.journal.path
        return {
            "label": label,
            "sessionId": entry.session_id,
            "path": str(journal_path) if journal_path is not None else None,
            "agent": self._agent_state(team, label),
            "snapshot": self._session_snapshot(entry, agent_label=label),
            "records": self._native_records(entry, label),
        }

    def _native_records(self, entry: ManagedSession, label: str) -> list[dict[str, object]]:
        """Serialize the live journal window without assigning UI semantics."""
        session = entry.session
        try:
            agent = session.team[label]
        except KeyError:
            raise ValueError(f"unknown agent {label!r}") from None
        return [session_record_message(record) for record in agent.session]

    async def _model_settings(self, params: Mapping[str, object]) -> dict[str, object]:
        entry = self._require_entry(params)
        return await model_settings_snapshot(
            self._agent_for_label(entry, params),
            entry.session.team,
        )

    async def _get_provider_usage(
        self, params: Mapping[str, object]
    ) -> dict[str, object]:
        entry = self._require_entry(params)
        return await provider_usage(self._agent_for_label(entry, params))

    async def _set_model_settings(
        self, params: Mapping[str, object]
    ) -> dict[str, object]:
        entry = self._require_entry(params)
        return await apply_model_settings(
            self._agent_for_label(entry, params),
            entry.session.team,
            params,
        )

    def _session_snapshot(
        self,
        entry: ManagedSession | None = None,
        *,
        agent_label: str = "Agent",
    ) -> dict[str, object]:
        """Project a native session through the shared session projector."""
        session = (entry or self._require_entry({})).session
        try:
            agent = session.team[agent_label]
        except KeyError:
            raise ValueError(f"unknown agent {agent_label!r}") from None
        return self._snapshot_projector.project(agent)

    def _session_records(self, params: Mapping[str, object]) -> dict[str, object]:
        entry = self._require_entry(params)
        label = params.get("agentLabel", "Agent")
        if not isinstance(label, str):
            raise TypeError("agentLabel must be a string")
        if not label:
            raise ValueError("agentLabel must be a non-empty string")
        return {
            "agentLabel": label,
            "records": self._native_records(entry, label),
        }

    def _require_agent(self) -> Agent:
        if self._agent is None:
            raise RuntimeError("no Sagent session is open")
        return self._agent

    def _require_team(self) -> Team:
        if self._team is None:
            raise RuntimeError("no Sagent team is open")
        return self._team

    def _agent_for_label(self, entry: ManagedSession, params: Mapping[str, object]) -> Agent:
        team = entry.session.team
        label = params.get("agentLabel", "Agent")
        if not isinstance(label, str) or not label:
            raise ValueError("agentLabel must be a non-empty string")
        if label not in team:
            raise ValueError(f"unknown agent {label!r}")
        return team[label]

    def _notify(
        self,
        event: str,
        params: Mapping[str, object],
        *,
        session_id: str | None = None,
    ) -> None:
        payload = {"type": event, **params}
        entry = self._sessions.get(session_id) if session_id is not None else None
        agent_label = (
            (entry.notification_agent_label or entry.active_agent_label)
            if entry is not None
            else (self._notification_agent_label or self._active_agent_label)
        )
        if agent_label is not None:
            payload.setdefault("agentLabel", agent_label)
        if entry is not None:
            entry.event_sequence += 1
            sequence = entry.event_sequence
        else:
            self._event_sequence += 1
            sequence = self._event_sequence
        message = session_notification_message(
            event,
            payload,
            agent_label=agent_label,
            session_id=session_id or self._session_id,
            sequence=sequence,
        )
        self.output.put_nowait(
            {"method": NATIVE_NOTIFICATION_METHOD, "params": message},
        )

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
            if request.get("method") == "system.shutdown":
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


def _agent_label(params: Mapping[str, object]) -> str:
    label = params.get("agentLabel", "Agent")
    if not isinstance(label, str) or not label:
        raise ValueError("agentLabel must be a non-empty string")
    return label


def _event_id(record: object) -> str | None:
    """Expose Sagent's record context id when the runtime assigned one."""
    context_id = getattr(record, "context_id", None)
    return f"record-{context_id}" if context_id is not None else None
