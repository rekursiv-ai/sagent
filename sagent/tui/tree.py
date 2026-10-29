"""The tree screen: every agent of the team, each one's running jobs under it.

::

     Agent                            busy      $1.20
     ├─ Bash  uv run pytest …         running   42s
     ├─ ● reviewer                    idle      $0.31
     │  └─ reviewer_0                 busy      $0.05
     └─ Agent_0                       busy      $0.12

``●`` marks the agent the screen shows. Up and Down move the cursor, Enter
picks the agent of its row, Right goes back unchanged, and Delete stops the
agent of its row, or cancels its job. A spawn's job is left out: the child it
started is its row.

Typical usage example:

  label = await pick(team, focused="Agent", stop=views.stop)   # None: Right.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import asyncio
import time

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from sagent.tui.prompt import repaint_on_change
from sagent.tui.status import elapsed


if TYPE_CHECKING:
    from collections.abc import Iterator

    from prompt_toolkit.key_binding import KeyPressEvent

    from sagent.agent.types import Agent


__all__ = [
    "Lineage",
    "Row",
    "pick",
    "render",
    "rows",
]


class Lineage(Protocol):
    """The agents of a team by label, and who spawned whom."""

    def __iter__(self) -> Iterator[str]:
        """Yield each label, in the order it joined."""
        ...

    def __getitem__(self, label: str) -> Agent:
        """Return the agent ``label`` names."""
        ...

    def parent(self, label: str) -> str | None:
        """Return who spawned ``label``; ``None`` for a root."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Row:
    """One line of the tree: an agent, or one of its jobs."""

    label: str
    """The agent the row belongs to: Enter focuses it."""

    branch: str
    """The tree's lines before the text, such as ``│  └─ ``."""

    text: str
    state: str
    detail: str
    """An agent's spend, or how long a job has run."""

    call_id: str = ""
    """The job's call; ``""`` on an agent's row."""


def rows(team: Lineage, *, focused: str, now: float | None = None) -> list[Row]:
    """Return the tree's rows: each root, then what is under it, depth first.

    Args:
      team: The agents.
      focused: The agent the screen shows; its row is marked ``●``.
      now: The epoch time jobs are timed to; the clock's when omitted.

    Returns:
      rows: An agent's row, its jobs' rows, then its children's, in join order.

    """
    at = time.time() if now is None else now
    under: dict[str | None, list[str]] = {}
    for label in team:
        under.setdefault(team.parent(label), []).append(label)
    out: list[Row] = []
    _walk(team, under, None, lead="", top=True, focused=focused, now=at, out=out)
    return out


def render(shown: list[Row], *, cursor: int, note: str = "") -> FormattedText:
    """Return the screen: the rows, the cursor's reversed, then the keys and ``note``.

    Args:
      shown: The rows.
      cursor: The row the cursor is on.
      note: What the last key did, such as a refusal; ``""`` for nothing.

    Returns:
      text: One line per row, then a blank line and the key help.

    """
    width = max((len(r.branch) + len(r.text) for r in shown), default=0)
    width = min(max(width, 32), 72)
    out: list[tuple[str, str]] = []
    for i, row in enumerate(shown):
        head = f"{row.branch}{row.text}"
        head = head if len(head) <= width else f"{head[: width - 1]}…"
        line = f" {head:<{width}}  {row.state:<9} {row.detail:>8} "
        out.extend([("reverse" if i == cursor else "", line), ("", "\n")])
    keys = "↑↓ move · Enter focus · Del stop · → back"
    out.append(("fg:ansibrightblack", f"\n {keys}{f' · {note}' if note else ''}\n"))
    return FormattedText(out)


async def pick(
    team: Lineage,
    *,
    focused: str,
    stop: Callable[[str], str],
    cancel: Callable[[str, str], str],
    interval_sec: float = 0.2,
) -> str | None:
    """Show the tree full screen until a row is picked or the user goes back.

    Args:
      team: The agents.
      focused: The agent the screen shows; the cursor starts on it.
      stop: Called with an agent's label on Delete; returns what to say.
      cancel: Called with a job's agent and call id on Delete; returns what to say.
      interval_sec: How often a changed tree is redrawn.

    Returns:
      label: The agent picked with Enter; ``None`` for Right or Ctrl+C.

    """
    screen = _Screen(team=team, focused=focused, stop=stop, cancel=cancel)
    app: Application[str | None] = Application(
        layout=Layout(
            Window(
                FormattedTextControl(screen.text, focusable=True, show_cursor=False),
            ),
        ),
        key_bindings=screen.keys(),
        full_screen=True,
    )
    redraw = asyncio.create_task(
        repaint_on_change(app.invalidate, screen.text, interval_sec),
    )
    try:
        return await app.run_async()
    finally:
        redraw.cancel()


@dataclass(slots=True, kw_only=True)
class _Screen:
    """The tree's cursor, and what its keys do."""

    team: Lineage
    focused: str
    stop: Callable[[str], str]
    cancel: Callable[[str, str], str]
    cursor: int = 0
    note: str = ""

    # Set here, not at the first draw: keys typed ahead run before it.
    def __post_init__(self) -> None:
        self.cursor = next(
            (
                i
                for i, r in enumerate(rows(self.team, focused=self.focused))
                if r.label == self.focused and not r.call_id
            ),
            0,
        )

    def text(self) -> FormattedText:
        """Return the screen now, the cursor kept on the rows.

        Returns:
          text: The rows, the cursor's reversed, then the key help.

        """
        shown = rows(self.team, focused=self.focused)
        self.cursor = min(self.cursor, max(len(shown) - 1, 0))
        return render(shown, cursor=self.cursor, note=self.note)

    def keys(self) -> KeyBindings:
        """Return the tree's keys.

        Returns:
          keys: Up, Down, Enter, Right, Ctrl+C, and Delete.

        """
        keys = KeyBindings()
        keys.add("up")(self._up)
        keys.add("down")(self._down)
        keys.add("enter")(self._enter)
        keys.add("right")(self._back)
        keys.add("c-c")(self._back)
        keys.add("delete")(self._delete)
        return keys

    # An agent may leave while the cursor sits past it; the cursor falls back a row.
    def _row(self) -> Row | None:
        shown = rows(self.team, focused=self.focused)
        self.cursor = min(self.cursor, len(shown) - 1)
        return shown[self.cursor] if shown else None

    def _up(self, event: KeyPressEvent) -> None:
        del event
        self.cursor = max(self.cursor - 1, 0)

    def _down(self, event: KeyPressEvent) -> None:
        del event
        self.cursor = min(
            self.cursor + 1,
            len(rows(self.team, focused=self.focused)) - 1,
        )

    def _enter(self, event: KeyPressEvent) -> None:
        row = self._row()
        if row is not None:
            event.app.exit(result=row.label)

    def _back(self, event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    def _delete(self, event: KeyPressEvent) -> None:
        del event
        row = self._row()
        if row is None:
            return
        self.note = (
            self.cancel(row.label, row.call_id) if row.call_id else self.stop(row.label)
        )


def _walk(
    team: Lineage,
    under: dict[str | None, list[str]],
    parent: str | None,
    *,
    lead: str,
    top: bool,
    focused: str,
    now: float,
    out: list[Row],
) -> None:
    """Add each child of ``parent`` to ``out``: its row, its jobs, then its children."""
    labels = under.get(parent, [])
    for n, label in enumerate(labels):
        last = n == len(labels) - 1
        branch = "" if top else lead + ("└─ " if last else "├─ ")
        below = "" if top else lead + ("   " if last else "│  ")
        agent = team[label]
        jobs = [
            (call_id, job)
            for call_id, job in agent.runtime.jobs.items()
            if job.call.name != "AgentSpawn"
        ]
        out.append(
            Row(
                label=label,
                branch=branch,
                text=f"{'● ' if label == focused else ''}{label}",
                state="busy" if _busy(agent) else "idle",
                detail=f"${agent.spend_usd:.2f}",
            ),
        )
        children = under.get(label, [])
        for k, (call_id, job) in enumerate(jobs):
            twig = "└─ " if k == len(jobs) - 1 and not children else "├─ "
            tool = agent.tools.get(job.call.name)
            out.append(
                Row(
                    label=label,
                    branch=below + twig,
                    text=tool.summary(job.call.arguments)
                    if tool is not None
                    else job.call.name,
                    state=job.state,
                    detail=elapsed(max(now - job.started, 0.0)),
                    call_id=call_id,
                ),
            )
        _walk(
            team,
            under,
            label,
            lead=below,
            top=False,
            focused=focused,
            now=now,
            out=out,
        )


def _busy(agent: Agent) -> bool:
    """Return whether ``agent`` is calling its model, compacting, or running tools."""
    rt = agent.runtime
    return rt.model_call is not None or rt.compact_task is not None or bool(rt.cohort)
