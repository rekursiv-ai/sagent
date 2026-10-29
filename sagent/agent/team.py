"""A team: the agents of one session, by label, and who spawned whom.

::

    Agent                  the root, joined by the front end
    ├── Agent_0            a child; lives until stopped
    └── reviewer           a child named when spawned
        └── reviewer_0     its child

Each agent's label is ``runtime.tool_context.label``, so a tool knows who
called it. A persistent child is opened by the team and closed with it. One
given a ``ChildSpec`` stands in its parent's journal until it is stopped, so a
resumed parent can start it again; closing the team, as a session ends, leaves
it standing.

A message one agent sends another is a request, which the receiver owes a
reply, unless the sender owed the receiver one: then it is that reply. An agent
that goes idle owing replies is nudged, at every idle, until it replies by
AgentSend; one with no AgentSend answers with that error. One that fails, is
halted, or is stopped answers for itself.

A child that stops tells its parent ``[<label> is idle] <last text>``, unless
its last tool call was an AgentSend to that parent, or its reply to the parent
says it already; one that fails or is halted says so instead. The last text is
what it said since it last stopped. Only an idle after work counts: nothing
wakes as a session resumes, until the user's first message nudges each other
agent still owing.

A watcher is a root that watches another agent: it is that agent's ``parent``
by name, though the team never closes the one for the other.

Typical usage example:

  async with Team(catalog=catalog) as team, Agent(model=opus, tools=tools) as root:
      team.join(root, label="Agent")
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, override

import logging

from treekle import from_plain

from sagent.agent.agent import Agent
from sagent.runtime import types
from sagent.runtime.exceptions import describe
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    ToolCall,
    UncategorizedRecord,
    UserMessage,
)
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from treekle import PlainTree

    from sagent.providers.types import ModelCatalog


__all__ = [
    "RESERVED",
    "ChildSpec",
    "Team",
]


RESERVED: Final = frozenset(
    {"parent", "children", "ancestors", "descendants", "all", "user"},
)
"""Names that address agents by relation, or the user; never a label."""


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildSpec:
    """What starts a persistent child again: stood in its parent's journal.

    Its model, settings and system prompt stand in its own journal.
    """

    label: str
    journal: str
    """The child's own journal, which a restart resumes."""

    tools: tuple[str, ...]
    """Its parent's tools it was given, by name."""

    max_rounds: int | None = None
    max_depth: int | None = None

    model: str = ""
    """``PROVIDER/MODEL``, for a spec stood before its child's journal kept the
    model: a resume starts there, and what the journal stood wins. ``""`` since."""


class Team(Mapping[str, "Agent"], AsyncResource):
    """The live agents of one session, by label.

    Args:
      catalog: Where a spawn finds the models it may switch to.

    """

    def __init__(self, *, catalog: ModelCatalog) -> None:
        self.catalog = catalog
        self.to_user: Callable[[str, str], None] | None = None
        """Shows the user what an agent sent ``user``: its label, then the text."""

        self._members: dict[str, _Member] = {}
        # Debtor, then creditor, then the first words of what was asked.
        self._owed: dict[str, dict[str, str]] = {}
        # Who watches whom: the watched agent, then its watcher.
        self._watchers: dict[str, str] = {}
        self.watching: dict[str, bool] = {}
        """Whether each watcher is sent what its agent does, by the watcher's label."""

        self.choices: dict[str, dict[str, str]] = {}
        """Each agent's own setting choices, as typed: label, setting, then value.
        A setting it has not chosen is its parent's, or the nearest ancestor's."""

        self.awaited: set[str] = set()
        """Children whose spawn waits on their run: its result tells the parent,
        so their idle does not."""

        # Whether the user has said anything yet; their first message wakes debtors.
        self._woken = False

    @override
    def __getitem__(self, label: str) -> Agent:
        return self._members[label].agent

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._members)

    @override
    def __len__(self) -> int:
        return len(self._members)

    def join(
        self,
        agent: Agent,
        *,
        label: str = "",
        parent: str | None = None,
        max_depth: int | None = None,
        closing: AsyncExitStack | None = None,
        spec: ChildSpec | None = None,
    ) -> str:
        """Add ``agent`` under ``parent``; return the label it now answers to.

        Args:
          agent: The agent.
          label: What to call it; ``""`` is ``<parent>_<n>``, or ``Agent`` for a root.
          parent: Who spawned it; ``None`` for the root.
          max_depth: How many generations may follow it; ``None`` inherits the
            parent's cap.
          closing: For a persistent child, what closes it and its journal when it
            stops; ``None`` when its spawner closes it.
          spec: For a persistent child, what starts it again, stood in its
            parent's journal under the label it joins as; a ``spec.label``
            other than that, as a resume under a new label gives, is withdrawn.

        Returns:
          label: Its label, also set as its ``tool_context.label``.

        Raises:
          ValueError: ``label`` is taken, ``parent`` is no member, or a ``spec``
            is given for no persistent child.

        """
        if parent is not None and parent not in self._members:
            raise ValueError(f"No agent {parent!r} to spawn from.")
        if label in RESERVED:
            raise ValueError(f"{label!r} names a relation; pick another label.")
        if label in self._members:
            raise ValueError(f"An agent is already labelled {label!r}.")
        if spec is not None and (parent is None or closing is None):
            raise ValueError("Only a persistent child has a spec to stand.")
        up = self._members.get(parent) if parent is not None else None
        depth = 0 if up is None else up.depth + 1
        caps = [
            c
            for c in (
                up.max_depth if up is not None else None,
                depth + max_depth if max_depth is not None else None,
            )
            if c is not None
        ]
        label = label or self.label_for(parent)
        dun = _Dun(team=self, label=label)
        self._members[label] = _Member(
            agent=agent,
            parent=parent,
            depth=depth,
            max_depth=min(caps) if caps else None,
            closing=closing,
            dun=dun,
        )
        agent.runtime.observers.append(dun)
        agent.runtime.tool_context.label = label
        self.restore(label)
        if spec is not None and parent is not None:
            session = self._members[parent].agent.session
            if spec.label != label:
                session.stand(_key(spec.label), None)
            payload = _payload(spec)
            payload["label"] = label
            session.stand(
                _key(label),
                UncategorizedRecord(kind="persistent_agent", payload=payload),
            )
        return label

    def restore(self, label: str) -> None:
        """Take up what ``label``'s journal says it owes and chose, once it is open.

        Args:
          label: A member whose session is open.

        """
        for record in self._members[label].agent.session.standing():
            held = {k: v for k, v in record.payload.items() if isinstance(v, str)}
            if record.kind == "owed":
                self._owed[label] = held
            elif record.kind == "choices":
                self.choices[label] = held

    def choose(self, label: str, setting: str, value: str | None) -> None:
        """Note ``label``'s own choice of ``setting``; ``None`` gives it up.

        Args:
          label: Who chose.
          setting: The setting's dotted name, such as ``model.thinking.effort``.
          value: What was typed; ``None`` to take the ancestors' again.

        """
        mine = self.choices.setdefault(label, {})
        if value is None:
            mine.pop(setting, None)
        else:
            mine[setting] = value
        self._members[label].agent.session.stand(
            "choices",
            UncategorizedRecord(kind="choices", payload=dict(mine)) if mine else None,
        )

    def inherited(self, label: str) -> dict[str, str]:
        """Return the choices ``label`` takes from its ancestors, nearest winning.

        Args:
          label: A member.

        Returns:
          choices: Setting to value, each from the nearest ancestor that chose it.

        """
        found: dict[str, str] = {}
        above = self._members[label].parent
        while above is not None:
            for setting, value in self.choices.get(above, {}).items():
                found.setdefault(setting, value)
            above = self._members[above].parent
        return found

    def label_for(self, parent: str | None) -> str:
        """Return the first free label for a child of ``parent``.

        Args:
          parent: Who spawns; ``None`` for a root.

        Returns:
          label: ``<parent>_<n>`` for the least free ``n``, or ``Agent``.

        """
        if parent is None:
            return "Agent"
        n = 0
        while f"{parent}_{n}" in self._members:
            n += 1
        return f"{parent}_{n}"

    def leave(self, label: str) -> Agent:
        """Remove ``label`` from the team; return its agent, still open.

        Args:
          label: Who leaves.

        Returns:
          agent: Its agent, for the caller to close.

        Raises:
          KeyError: No agent has ``label``.

        """
        member = self._members.pop(label)
        self._forget(label)
        observers = member.agent.runtime.observers
        if member.dun in observers:
            observers.remove(member.dun)
        return member.agent

    def parent(self, label: str) -> str | None:
        """Return who spawned ``label``; ``None`` for the root."""
        return self._members[label].parent

    def above(self, label: str) -> str | None:
        """Return ``label``'s ``parent`` by name: who spawned it, or a root's watcher."""
        spawner = self._members[label].parent
        return spawner if spawner is not None else self._watchers.get(label)

    def children(self, label: str) -> list[str]:
        """Return who ``label`` spawned, in the order they joined."""
        return [c for c, m in self._members.items() if m.parent == label]

    def resolve(self, name: str, *, by: str) -> list[str]:
        """Return the labels ``name`` addresses, seen from ``by``.

        Args:
          name: A label, or ``parent``, ``children``, ``ancestors``, or
            ``descendants``.
          by: Who addresses them.

        Returns:
          labels: Nearest first; ``[]`` when there are none.

        """
        if name == "parent":
            above = self.above(by) if by in self._members else None
            return [] if above is None else [above]
        if name == "children":
            return self.children(by)
        if name == "ancestors":
            chain: list[str] = []
            above = self.above(by) if by in self._members else None
            while above is not None and above not in chain:
                chain.append(above)
                above = self.above(above)
            return chain
        if name == "descendants":
            below: list[str] = []
            for child in self.children(by):
                below.extend([child, *self.resolve("descendants", by=child)])
            return below
        return [name] if name in self._members else []

    def watch(self, watcher: str, watched: str, *, journal: str | None = None) -> None:
        """Make ``watcher`` the ``parent``, by name, of the root ``watched``.

        Args:
          watcher: A root, which watches.
          watched: A root, watched.
          journal: The watcher's own journal, stood in ``watched``'s so a resume
            starts it again; ``None`` keeps none.

        Raises:
          ValueError: ``watched`` is no root, or already watched.

        """
        if self._members[watched].parent is not None:
            raise ValueError(f"{watched!r} has a parent; only a root is watched.")
        if watched in self._watchers:
            raise ValueError(f"{watched!r} is watched by {self._watchers[watched]!r}.")
        self._watchers[watched] = watcher
        self.watching[watcher] = True
        if journal is not None:
            self._members[watched].agent.session.stand(
                _watch_key(watcher),
                UncategorizedRecord(
                    kind="watcher",
                    payload={"label": watcher, "journal": journal},
                ),
            )

    def watched(self, watcher: str) -> str | None:
        """Return whom ``watcher`` watches; ``None`` for no one."""
        return next((w for w, by in self._watchers.items() if by == watcher), None)

    def send(self, sender: str, recipient: str, content: str) -> None:
        """Keep the books for ``content``, then deliver it.

        Args:
          sender: Who sends.
          recipient: A member, or ``user``.
          content: What it says.

        Raises:
          ValueError: ``recipient`` is no member, or ``user`` when no one shows it.

        """
        self.book(sender, recipient=recipient, content=content)
        self._deliver(sender, recipient=recipient, content=content)

    def book(self, sender: str, recipient: str, content: str) -> None:
        """Keep the books for ``content``, sent but not yet delivered.

        A message to one the sender owes a reply is that reply, and settles it;
        any other is a request, which the recipient now owes. ``user`` owes
        nothing, and a message to oneself is neither.

        Args:
          sender: Who sends.
          recipient: A member, or ``user``.
          content: What it says.

        Raises:
          ValueError: ``recipient`` is no member, or ``user`` when no one shows it.

        """
        self._check(recipient)
        owed = self._owed.get(sender, {})
        if recipient in owed:
            del owed[recipient]
            self._book(sender)
        elif recipient not in ("user", sender):
            self._owed.setdefault(recipient, {})[sender] = _head(content)
            self._book(recipient)

    def deliver(self, sender: str, recipient: str, content: str) -> None:
        """Put ``content``, already booked, before ``recipient``.

        Args:
          sender: Who sent it.
          recipient: A member, or ``user``.
          content: What it says.

        Raises:
          ValueError: ``recipient`` is no member, or ``user`` when no one shows it.

        """
        self._check(recipient)
        self._deliver(sender, recipient=recipient, content=content)

    def ask(self, debtor: str, content: str) -> None:
        """Note that the user asked ``debtor`` ``content``: it owes ``user`` a reply."""
        self._owed.setdefault(debtor, {})["user"] = _head(content)
        self._book(debtor)

    def owes(self, label: str) -> dict[str, str]:
        """Return whom ``label`` owes a reply, each with the first words asked."""
        return dict(self._owed.get(label, {}))

    def answer(self, label: str, content: str) -> None:
        """Send ``content`` to each one ``label`` owes, settling every debt."""
        owed = self._owed.pop(label, {})
        if owed:
            self._book(label)
        for creditor in owed:
            self._deliver(label, creditor, content)

    def tell(self, sender: str, recipient: str, content: str) -> None:
        """Deliver ``content`` as news: no request, so no one owes a reply.

        Args:
          sender: Who it is about.
          recipient: A member.
          content: What it says.

        """
        self._deliver(sender, recipient, content)

    def wake(self, addressed: str) -> None:
        """Nudge each agent still owing, the first time the user says anything.

        Args:
          addressed: The agent the user said it to, at work on it already.

        """
        if self._woken:
            return
        self._woken = True
        for label, owed in list(self._owed.items()):
            if owed and label != addressed:
                self._members[label].dun.nudge(owed)

    def _book(self, debtor: str) -> None:
        """Stand what ``debtor`` owes in its journal, so a resume still owes it."""
        owed = self._owed.get(debtor, {})
        self._members[debtor].agent.session.stand(
            "owed",
            UncategorizedRecord(kind="owed", payload=dict(owed)) if owed else None,
        )

    def can_spawn(self, label: str) -> bool:
        """Return whether ``label`` may spawn a child under its depth cap."""
        member = self._members[label]
        return member.max_depth is None or member.depth < member.max_depth

    def standing(self, label: str) -> list[ChildSpec]:
        """Return the persistent children ``label``'s journal says to start again.

        Args:
          label: The parent.

        Returns:
          specs: Each child stood and not stopped, in the order it was stood; a
            record that no longer reads is skipped.

        """
        specs: list[ChildSpec] = []
        for record in self._members[label].agent.session.standing():
            if record.kind != "persistent_agent":
                continue
            # A spec stood before its child's journal kept the model held it here.
            payload = {k: v for k, v in record.payload.items() if k not in _MOVED}
            provider, model = (
                record.payload.get("provider"),
                record.payload.get("model"),
            )
            if isinstance(provider, str) and isinstance(model, str):
                payload["model"] = f"{provider}/{model}"
            try:
                specs.append(from_plain(payload, ChildSpec))
            except ValueError:
                logger.warning("skipping an unreadable child record: %s", record)
        return specs

    def persistent(self) -> list[str]:
        """Return the labels of the persistent children, in the order they joined."""
        return [label for label, m in self._members.items() if m.closing is not None]

    async def stop(self, label: str) -> None:
        """Remove a persistent child and close it, its own children first.

        Each one stood is withdrawn from its parent's journal: a resume does not
        start it again.

        Args:
          label: Who stops.

        Raises:
          KeyError: No agent has ``label``.

        """
        await self._close(label, withdraw=True)

    @override
    async def aclose(self) -> None:
        for label in reversed(self.persistent()):
            if label in self._members:
                try:
                    await self._close(label, withdraw=False)
                except Exception:
                    logger.warning("closing agent %r failed", label, exc_info=True)

    async def _close(self, label: str, *, withdraw: bool) -> None:
        """Close ``label`` and its children; ``withdraw`` their standing records too."""
        for child in self.children(label):
            await self._close(child, withdraw=withdraw)
        # A session's end leaves debts and watchers standing, for a resume.
        if withdraw:
            self.answer(label, f"[{label} stopped]")
            watched = self.watched(label)
            if watched is not None:
                self._members[watched].agent.session.stand(_watch_key(label), None)
        member = self._members.pop(label)
        self._forget(label)
        if withdraw and member.closing is not None and member.parent is not None:
            self._members[member.parent].agent.session.stand(_key(label), None)
        if member.closing is not None:
            await member.closing.aclose()

    def _forget(self, label: str) -> None:
        """Drop what ``label`` owed, what was owed it, and whom it watched."""
        self._owed.pop(label, None)
        for owed in self._owed.values():
            owed.pop(label, None)
        self._watchers = {
            w: by for w, by in self._watchers.items() if label not in (w, by)
        }
        self.watching.pop(label, None)
        self.choices.pop(label, None)
        self.awaited.discard(label)

    def _check(self, recipient: str) -> None:
        """Raise ``ValueError`` unless ``recipient`` can be reached."""
        if recipient != "user" and recipient not in self._members:
            raise ValueError(f"Unknown agent {recipient!r}. Active: {sorted(self)}")
        if recipient == "user" and self.to_user is None:
            raise ValueError("No user is watching; there is no one to tell.")

    def _deliver(self, sender: str, recipient: str, content: str) -> None:
        """Put ``content`` before ``recipient``: the user, or a member's inbox."""
        if recipient == "user":
            if self.to_user is not None:
                self.to_user(sender, content)
            return
        # Only members are owed or sent to: one that leaves is forgotten as it goes.
        self._members[recipient].agent.runtime.push(
            AgentToAgentMessage(sender=sender, recipient=recipient, content=content),
        )


@dataclass(slots=True, kw_only=True, eq=False)
class _Dun:
    """Holds an agent to the replies it owes, and tells its parent when it stops."""

    team: Team
    label: str

    _said: str = ""
    # Whether the model answered since the last idle: an idle without is no stop.
    _worked: bool = field(default=False, repr=False)
    _last_call: ToolCall | None = field(default=None, repr=False)

    def __call__(self, event: types.RuntimeEvent) -> None:
        """Nudge, answer for the agent, or tell its parent, as ``event`` calls for."""
        if isinstance(event, types.ModelResponseComplete):
            records = event.response.records
            said = "".join(
                r.content or "" for r in records if isinstance(r, AssistantMessage)
            )
            self._said = said or self._said
            self._last_call = next(
                (r for r in reversed(records) if isinstance(r, ToolCall)),
                self._last_call,
            )
            self._worked = True
            return
        if isinstance(event, UserMessage) or (
            isinstance(event, AgentToAgentMessage) and event.sender == "user"
        ):
            self.team.wake(self.label)
            return
        owed = self.team.owes(self.label)
        if isinstance(event, types.ModelResponseError):
            self._ended(f"[{self.label} failed] {describe(event.exception)}", owed)
        elif isinstance(event, types.ModelResponseCancelled):
            self._ended(f"[{self.label} halted]", owed)
        elif isinstance(event, types.AgentIdle) and self._worked:
            # pragma: no mutate start -- only its truth is read; None is as false.
            self._worked = False
            # pragma: no mutate end
            last, self._last_call = self._last_call, None
            self._idled(owed, last)

    def nudge(self, owed: dict[str, str]) -> None:
        """Wake the agent with whom it owes a reply, in a note only it sees.

        Args:
          owed: Each creditor, with the first words it asked.

        """
        asked = "; ".join(f"{c} ({what!r})" for c, what in owed.items())
        self.team[self.label].runtime.push(
            types.Nudge(
                content=f"You owe a reply to: {asked}. Answer each message"
                " with AgentSend. Do not acknowledge this note.",
            ),
        )

    # A debtor is nudged at every idle until it replies; only one that cannot reply,
    # having no AgentSend, is answered for, with the error.
    def _idled(self, owed: dict[str, str], last: ToolCall | None) -> None:
        """Nudge a debtor, or answer for one without AgentSend; else tell its parent."""
        if owed and "AgentSend" in self.team[self.label].tools:
            self.nudge(owed)
            return
        if owed:
            self._ended(
                f"[{self.label} failed] it has no AgentSend to reply with",
                owed,
            )
            return
        said, self._said = self._said, ""
        if not (
            last is not None
            and last.name == "AgentSend"
            and last.arguments.get("to") in (self.team.parent(self.label), "parent")
        ):
            self._tell_parent(f"[{self.label} is idle] {said}".rstrip(), owed)

    # The runtime then waits for the user, so no idle follows to end the run.
    def _ended(self, said: str, owed: dict[str, str]) -> None:
        """Answer each creditor, and tell the parent, that the run ended so."""
        self._said = ""
        self._last_call = None
        if owed:
            self.team.answer(self.label, said)
        self._tell_parent(said, owed)

    def _tell_parent(self, said: str, owed: dict[str, str]) -> None:
        """Tell the parent ``said``, unless it was answered, or a spawn tells it."""
        parent = self.team.parent(self.label)
        if parent is None or parent in owed or self.label in self.team.awaited:
            return
        self.team.tell(self.label, parent, said)


def _head(text: str, words: int = 8) -> str:
    """Return the first ``words`` words of ``text``, ``…`` after when cut."""
    split = text.split()
    return " ".join(split[:words]) + ("…" if len(split) > words else "")


def _watch_key(label: str) -> str:
    """Return what a watcher's record stands under, in the journal it watches."""
    return f"watcher:{label}"


def _key(label: str) -> str:
    """Return what a child's record stands under."""
    return f"agent:{label}"


# What a spec once held that its child's journal now keeps, or that no longer is:
# every child now owes its replies, so none asks to be told when it idles.
_MOVED = frozenset(
    {
        "provider",
        "model",
        "system",
        "effort",
        "budget",
        "output",
        "thinking",
        "notify_on_asleep",
    },
)


def _payload(spec: ChildSpec) -> dict[str, PlainTree]:
    """Return ``spec`` as a record's payload."""
    return {
        "label": spec.label,
        "journal": spec.journal,
        "tools": list(spec.tools),
        "max_rounds": spec.max_rounds,
        "max_depth": spec.max_depth,
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class _Member:
    """One agent's place in the team."""

    agent: Agent
    parent: str | None
    depth: int
    max_depth: int | None
    """The deepest any descendant may be; ``None`` for no cap."""

    closing: AsyncExitStack | None
    """What closes a persistent child; ``None`` for one its spawner closes."""

    dun: _Dun
    """What holds it to the replies it owes."""
