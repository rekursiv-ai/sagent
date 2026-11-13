"""UI-neutral control over live Sagent agents.

This module is deliberately non-owning.  ``Agent`` and ``Team`` remain the
owners of resources, journals, and runtime loops; a controller only provides a
stable surface for a UI or transport to submit native runtime events and
observe native runtime events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from sagent.runtime import types


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from sagent.agent.agent import Agent
    from sagent.session.types import Attachment


__all__ = ["AgentSessionController", "Submission", "SubmissionMode"]


type SubmissionMode = Literal["immediate", "queued", "deferred"]
"""How user input enters a native Sagent runtime."""


@dataclass(frozen=True, slots=True)
class Submission:
    """A transport-level receipt for input accepted by one agent.

    ``id`` identifies the submission for the caller.  It is not a replacement
    for Sagent's model rounds or persisted session records; those remain
    authoritative in the runtime and journal.
    """

    id: str
    agent_label: str
    text: str
    mode: SubmissionMode


class AgentSessionController:
    """Dispatch native input and expose native events for a group of agents.

    The controller does not create, open, or close agents.  The caller owns the
    surrounding ``Team``/``AsyncExitStack`` and can use this controller for a
    TUI, daemon, test harness, or another frontend.
    """

    def __init__(self, agents: Mapping[str, Agent]) -> None:
        self._agents = agents
        self._sequence = 0
        self._subscriptions: dict[
            str,
            dict[
                Callable[[types.RuntimeEvent], None],
                Callable[[types.RuntimeEvent], None],
            ],
        ] = {}

    @property
    def agents(self) -> Mapping[str, Agent]:
        """Return the live agent registry owned by the caller."""
        return self._agents

    def send(
        self,
        text: str,
        *,
        agent_label: str = "Agent",
        mode: SubmissionMode = "immediate",
        attachments: tuple[Attachment, ...] = (),
    ) -> Submission:
        """Submit text through the runtime's native inbox semantics.

        This method does not wait for a response.  ``Agent.runtime`` owns
        streaming, tools, retries, queued/deferred handling, and idleness.
        """
        if not text.strip():
            raise ValueError("text must be a non-empty string")
        agent = self._agent(agent_label)
        if mode == "immediate":
            event: types.RuntimeEvent = types.UserMessage(
                content=text,
                attachments=attachments,
            )
        elif mode == "queued":
            event = types.UserQueuedMessage(text=text, attachments=attachments)
        elif mode == "deferred":
            event = types.UserDeferredMessage(text=text, attachments=attachments)
        else:
            raise ValueError(f"unknown submission mode {mode!r}")
        self._sequence += 1
        submission = Submission(
            id=f"submission-{self._sequence}",
            agent_label=agent_label,
            text=text,
            mode=mode,
        )
        agent.runtime.push(event)
        return submission

    def halt(self, *, agent_label: str = "Agent") -> bool:
        """Ask one native runtime to halt and report whether it was active."""
        agent = self._agent(agent_label)
        if agent.runtime.is_idle:
            return False
        agent.runtime.halt()
        return True

    def subscribe(
        self,
        observer: Callable[[str, types.RuntimeEvent], None],
        *,
        agent_label: str | None = None,
    ) -> Callable[[], None]:
        """Subscribe to native runtime events and return an idempotent remover."""
        labels = (agent_label,) if agent_label is not None else tuple(self._agents)
        entries: list[tuple[str, Callable[[types.RuntimeEvent], None]]] = []
        for label in labels:
            agent = self._agent(label)

            existing_forward = self._subscriptions.get(label, {}).get(observer)
            if existing_forward is not None:
                continue

            def forward(event: types.RuntimeEvent, label: str = label) -> None:
                observer(label, event)

            agent.runtime.observers.append(forward)
            self._subscriptions.setdefault(label, {})[observer] = forward
            entries.append((label, forward))

        def remove() -> None:
            for label, forward in entries:
                agent = self._agent(label)
                if forward in agent.runtime.observers:
                    agent.runtime.observers.remove(forward)
                subscriptions = self._subscriptions.get(label)
                if subscriptions is not None and subscriptions.get(observer) is forward:
                    subscriptions.pop(observer, None)
                    if not subscriptions:
                        self._subscriptions.pop(label, None)

        return remove

    def unsubscribe_all(self) -> None:
        """Remove every runtime observer registered through this controller."""
        for label, subscriptions in tuple(self._subscriptions.items()):
            agent = self._agent(label)
            for forward in tuple(subscriptions.values()):
                if forward in agent.runtime.observers:
                    agent.runtime.observers.remove(forward)
            subscriptions.clear()
        self._subscriptions.clear()

    def _agent(self, label: str) -> Agent:
        try:
            return self._agents[label]
        except KeyError:
            raise ValueError(f"unknown agent {label!r}") from None
