"""What the screen shows of each agent of the team, and which one it shows now.

Each agent the user has focused has a ``View``: its panes, status line, keys
and commands. Its panes and status line stay on its runtime while another is
shown, so a queued message is still sent and its counts stay whole; the
scrollback and the title follow the focus. An agent that leaves the team
hands the focus to its parent. Each agent has its own screen settings, which
the printer takes up as it is shown; one that joins takes up the settings its
ancestors chose.

Typical usage example:

  views = Views(agent, team=team, printer=printer, catalog=catalog)
  views.focus("reviewer")      # Its session replays; what is typed goes to it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import asyncio
import logging

from sagent.agent.types import Agent
from sagent.tui.console import Title
from sagent.tui.prompt import (
    Deferred,
    Navigation,
    Panes,
    commit_panes,
    key_bindings,
)
from sagent.tui.render import Scrollback, error_text
from sagent.tui.replay import replay
from sagent.tui.slash import Commands
from sagent.tui.status import Status


if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyBindings

    from sagent.agent.team import Team
    from sagent.providers.types import ModelCatalog
    from sagent.runtime.types import RuntimeEvent
    from sagent.tui.tree import Lineage
    from sagent.tui.types import Printer, ThinkingView
else:
    from wrapt import lazy_import

    KeyBindings = lazy_import("prompt_toolkit.key_binding", "KeyBindings")  # 80 ms.


__all__ = [
    "View",
    "Views",
]


logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True)
class View:
    """The screen's parts for one agent."""

    agent: Agent
    label: str
    parent: str | None
    """Who spawned it, kept for when it has left the team."""

    panes: Panes
    deferred: Deferred
    status: Status
    scrollback: Scrollback
    keys: KeyBindings
    commands: Commands
    uncommit: Callable[[], None]


class Views:
    """Each focused agent's view, and the one the screen shows.

    Args:
      root: The session's agent, shown first; its session replays now.
      team: The agents it may show; ``None`` for ``root`` alone.
      printer: Where the scrollback goes.
      catalog: Where each view's model settings find models.
      commands: Builds each view's commands.
      switched: Called after the focus moves.

    """

    def __init__(
        self,
        root: Agent,
        *,
        team: Team | None,
        printer: Printer,
        catalog: ModelCatalog,
        commands: Callable[..., Commands] = Commands,
        switched: Callable[[], None] = lambda: None,
    ) -> None:
        self.team = team
        self.printer = printer
        self.catalog = catalog
        self.title = Title()
        self.switched = switched
        self._make = commands
        self._root = root
        self._root_label = root.runtime.tool_context.label
        self._views: dict[str, View] = {}
        self._held: list[RuntimeEvent] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._thinking = printer.show_thinking
        self._screens: dict[str, _Screen] = {}
        self._joined: set[str] = set()
        if team is not None:
            team.to_user = self._to_user
        view = self._view(self._root_label, root)
        replay(root, view.scrollback)
        self.current = view
        self._show(view)
        self._adopt()

    @property
    def lineage(self) -> Lineage:
        """The agents the tree lists: the team, or the root alone."""
        return (
            self.team if self.team is not None else _Alone(self._root_label, self._root)
        )

    def toolbar(self) -> str:
        """Return the shown agent's status line."""
        return self.current.status.frame()

    def screen(self, label: str) -> ThinkingView:
        """Return ``label``'s screen settings; the printer follows the shown agent's.

        Args:
          label: An agent of the team.

        Returns:
          screen: Its settings, made at the screen's start-up value on first ask.

        """
        screen = self._screens.get(label)
        if screen is None:
            screen = self._screens[label] = _Screen(
                self,
                label,
                show_thinking=self._thinking,
            )
        return screen

    def focus(self, label: str) -> None:
        """Show ``label``: its session replays, and what is typed goes to it.

        Args:
          label: An agent of the team.

        Raises:
          KeyError: No agent has ``label``.

        """
        agent = self.lineage[label]
        if label == self.current.label and agent is self.current.agent:
            return
        self._hide(self.current)
        view = self._views.get(label)
        if view is None or view.agent is not agent:
            view = self._view(label, agent)
        self.current = view
        self.printer.show_thinking = self.screen(label).show_thinking
        self.printer.line(f"── {label} ──")
        replay(agent, view.scrollback)
        self._show(view)
        self.switched()

    def follow(self) -> None:
        """Drop the views of agents that left; show a gone agent's parent instead."""
        lineage = self.lineage
        gone = [
            v
            for label, v in self._views.items()
            if label not in lineage or lineage[label] is not v.agent
        ]
        for view in gone:
            del self._views[view.label]
            self._screens.pop(view.label, None)
            self._close(view)
        self._joined &= set(lineage)
        self._adopt()
        if self.current in gone:
            left = self.current
            self.printer.note(f"[{left.label} stopped]")
            parent = left.parent
            self.focus(
                parent
                if parent is not None and parent in lineage
                else self._root_label,
            )

    async def watch(self, interval_sec: float = 0.2) -> None:
        """Call ``follow`` each ``interval_sec``, until cancelled."""
        while True:
            await asyncio.sleep(interval_sec)
            self.follow()

    def hold(self) -> None:
        """Keep what the shown agent does from printing, until ``release``."""
        view = self.current
        self._hide(view)
        held: list[RuntimeEvent] = []
        self._held = held
        view.agent.runtime.observers.append(held.append)

    def release(self, *, show: bool) -> None:
        """Print what was held, if ``show``; then let the shown agent print again.

        Args:
          show: Whether the held events print; ``False`` drops them.

        """
        held, self._held = self._held, None
        view = self.current
        if held is not None:
            observers = view.agent.runtime.observers
            if held.append in observers:
                observers.remove(held.append)
            if show:
                for event in held:
                    view.scrollback(event)
        self._show(view)

    def stop(self, label: str) -> str:
        """Stop the persistent agent ``label`` and its children; say what happened.

        Args:
          label: An agent of the team.

        Returns:
          said: That it is stopping, or why it is not.

        """
        if self.team is None or label not in self.team.persistent():
            return (
                f"{label} ends with /quit"
                if label == self._root_label
                else f"{label} is not a persistent agent"
            )
        self._track(asyncio.create_task(self.team.stop(label)))
        return f"stopping {label}"

    def cancel(self, label: str, call_id: str) -> str:
        """Cancel the job ``call_id`` of ``label``; say what happened.

        Args:
          label: The agent running it.
          call_id: Its call.

        Returns:
          said: That it is cancelled, or that it already ended.

        """
        job = self.lineage[label].runtime.jobs.get(call_id)
        if job is None:
            return f"{call_id} already ended"
        job.cancel()
        return f"cancelled {call_id}"

    async def aclose(self) -> None:
        """Take every view off its runtime; wait for its commands and the stops."""
        if self.team is not None and self.team.to_user == self._to_user:
            self.team.to_user = None
        self.release(show=False)
        self._hide(self.current)
        for view in self._views.values():
            self._close(view)
            await view.commands.aclose()
        self._views.clear()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def _to_user(self, sender: str, text: str) -> None:
        """Print what ``sender`` sent the user, naming the user as its recipient."""
        self.printer.agent(sender, recipient="user", text=text)

    def _adopt(self) -> None:
        """Have each agent new to the team take up the settings its ancestors chose."""
        if self.team is None:
            return
        root = self._views[self._root_label].commands
        for label in self.team:
            if label not in self._joined:
                self._joined.add(label)
                self._track(asyncio.create_task(self._inherit(root, label)))

    async def _inherit(self, root: Commands, label: str) -> None:
        """Take up, for ``label``, what its ancestors chose."""
        if self.team is not None and label in self.team:
            _ = await root.commands_for(self.team, label)

    def _track(self, task: asyncio.Task[None]) -> None:
        """Keep ``task`` until it ends; print why, if it fails."""
        self._tasks.add(task)
        task.add_done_callback(self._stopped)

    def _view(self, label: str, agent: Agent) -> View:
        """Return a new view of ``agent``, its panes and status line on its runtime."""
        runtime = agent.runtime
        panes = Panes()
        scrollback = Scrollback(
            self.printer,
            tools=agent.tools,
            markers=runtime.markers,
            home=runtime.tool_context.origin,
        )
        status = Status(runtime, spend=agent.spend_usd)
        runtime.observers.append(status)
        view = View(
            agent=agent,
            label=label,
            parent=self.team.parent(label) if self.team is not None else None,
            panes=panes,
            deferred=Deferred(runtime, panes),
            status=status,
            scrollback=scrollback,
            keys=key_bindings(runtime, panes, Navigation()),
            commands=self._make(
                agent,
                catalog=self.catalog,
                say=scrollback.show,
                screen=self.screen(label),
                panes=panes,
                team=self.team,
                focus=self.focus,
                screens=self.screen,
            ),
            uncommit=commit_panes(runtime, panes),
        )
        self._views[label] = view
        return view

    def _show(self, view: View) -> None:
        """Put the scrollback and the title on ``view``'s runtime."""
        observers = view.agent.runtime.observers
        for observer in (view.scrollback, self.title):
            if observer not in observers:
                observers.append(observer)

    def _hide(self, view: View) -> None:
        """Take the scrollback and the title off ``view``'s runtime."""
        observers = view.agent.runtime.observers
        for observer in (view.scrollback, self.title):
            if observer in observers:
                observers.remove(observer)

    def _close(self, view: View) -> None:
        """Take every part of ``view`` off its runtime."""
        self._hide(view)
        view.uncommit()
        observers = view.agent.runtime.observers
        if view.status in observers:
            observers.remove(view.status)

    def _stopped(self, task: asyncio.Task[None]) -> None:
        """Forget a finished stop; print why it failed."""
        self._tasks.discard(task)
        if not task.cancelled() and (e := task.exception()) is not None:
            logger.warning("stopping an agent failed", exc_info=e)
            self.printer.error(error_text(e))


class _Screen:
    """One agent's screen settings; the printer takes them up while it is shown."""

    def __init__(self, views: Views, label: str, *, show_thinking: bool) -> None:
        self._views = views
        self._label = label
        self._show = show_thinking

    @property
    def show_thinking(self) -> bool:
        """Whether this agent's reasoning shows."""
        return self._show

    @show_thinking.setter
    def show_thinking(self, show: bool) -> None:
        self._show = show
        if self._views.current.label == self._label:
            self._views.printer.show_thinking = show


class _Alone(Mapping[str, "Agent"]):
    """A lone agent, as the tree reads a team."""

    def __init__(self, label: str, agent: Agent) -> None:
        self._label = label
        self._agent = agent

    @override
    def __getitem__(self, label: str) -> Agent:
        if label != self._label:
            raise KeyError(label)
        return self._agent

    @override
    def __iter__(self) -> Iterator[str]:
        return iter((self._label,))

    @override
    def __len__(self) -> int:
        return 1

    def parent(self, label: str) -> str | None:
        """Return ``None``: a lone agent is a root."""
        del label
        return None
