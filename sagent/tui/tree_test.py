"""Tests for the tree screen: its rows, how they draw, and its keys."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from io import StringIO
from typing import TYPE_CHECKING

import asyncio

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output
from treekle import PlainTree

import pytest

from sagent.agent.agent import Agent
from sagent.agent.team import Team
from sagent.lib.userdirs import config_dir
from sagent.model.scripted import ScriptedProvider, call, reply
from sagent.providers.catalog import ModelCatalog
from sagent.session.types import UncategorizedToolResult, UserMessage
from sagent.tools.display import ToolDisplay
from sagent.tui import tree
from sagent.tui.prompt import repaint_on_change
from sagent.tui.theme import load_theme
from sagent.tui.tree import Row, pick, render, rows


if TYPE_CHECKING:
    from collections.abc import Callable

    from sagent.tools.types import ToolContext
    from sagent.tui.theme import Theme


@dataclass(slots=True, kw_only=True)
class _Wait:
    """A tool that runs until cancelled."""

    name: str = "Wait"
    description: str = "Waits."
    schema: Mapping[str, PlainTree] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": dict[str, PlainTree](),
        },
    )
    display: ToolDisplay = field(default_factory=ToolDisplay)
    started: asyncio.Event = field(default_factory=asyncio.Event)

    def summary(self, arguments: Mapping[str, PlainTree]) -> str:
        return f"{self.name} {arguments.get('on', '')}".strip()

    def serialize_key(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> str | None:
        del context, arguments
        return None

    async def call(
        self,
        context: ToolContext,
        arguments: Mapping[str, PlainTree],
    ) -> UncategorizedToolResult:
        del context, arguments
        self.started.set()
        await asyncio.Event().wait()
        return UncategorizedToolResult(call_id="", content="never")


def _team() -> Team:
    """Return ``Agent``, its children ``kid`` and ``other``, and ``kid``'s ``grand``."""
    team = Team(catalog=ModelCatalog(factories={}))
    for label, parent in (
        ("Agent", None),
        ("kid", "Agent"),
        ("other", "Agent"),
        ("grand", "kid"),
    ):
        _ = team.join(
            Agent(model=ScriptedProvider()["scripted"]),
            label=label,
            parent=parent,
        )
    return team


def test_each_agent_follows_its_parent_depth_first_with_the_shown_one_marked() -> None:
    team = _team()
    team["kid"].spend_usd = 0.25

    shown = rows(team, focused="kid", now=0.0)

    assert [(r.branch, r.text, r.state, r.detail) for r in shown] == [
        ("", "Agent", "idle", "$0.00"),
        ("├─ ", "● kid", "idle", "$0.25"),
        ("│  └─ ", "grand", "idle", "$0.00"),
        ("└─ ", "other", "idle", "$0.00"),
    ]
    assert {r.call_id for r in shown} == {""}


def test_an_agent_calling_its_model_or_compacting_alone_is_busy() -> None:
    team = _team()

    async def never() -> None:
        await asyncio.Event().wait()

    async def check() -> None:
        kid, other = team["kid"].runtime, team["other"].runtime
        kid.model_call = asyncio.create_task(never())
        other.compact_task = asyncio.create_task(never())
        try:
            assert [r.state for r in rows(team, focused="kid", now=0.0)] == [
                "idle",
                "busy",
                "idle",
                "busy",
            ]
        finally:
            for task in (kid.model_call, other.compact_task):
                _ = task.cancel()

    asyncio.run(check())


@pytest.mark.asyncio
async def test_a_last_child_s_job_hangs_from_a_blank_stem_and_never_runs_backwards() -> (
    None
):
    wait = _Wait()
    team = Team(catalog=ModelCatalog(factories={}))
    _ = team.join(Agent(model=ScriptedProvider()["scripted"]))
    provider = ScriptedProvider(replies=[reply(calls=[call("Wait", "w1")])])
    async with Agent(model=provider["scripted"], tools=[wait]) as kid:
        _ = team.join(kid, label="kid", parent="Agent")
        kid.runtime.push(UserMessage(content="go"))
        await asyncio.wait_for(wait.started.wait(), timeout=5)
        started = kid.runtime.jobs["w1"].started

        shown = rows(team, focused="Agent", now=started - 5)

        assert [(r.branch, r.text, r.detail) for r in shown] == [
            ("", "● Agent", "$0.00"),
            ("└─ ", "kid", "$0.00"),
            ("   └─ ", "Wait", "0s"),
        ]
        kid.runtime.halt()


@pytest.mark.asyncio
async def test_a_busy_agent_lists_its_jobs_but_not_its_spawns() -> None:
    wait, spawn = _Wait(), _Wait(name="AgentSpawn")
    team = Team(catalog=ModelCatalog(factories={}))
    provider = ScriptedProvider(
        replies=[
            reply(calls=[call("Wait", "w1", on="tests"), call("AgentSpawn", "s1")]),
        ],
    )
    async with Agent(model=provider["scripted"], tools=[wait, spawn]) as agent:
        _ = team.join(agent)
        _ = team.join(
            Agent(model=ScriptedProvider()["scripted"]),
            label="kid",
            parent="Agent",
        )
        agent.runtime.push(UserMessage(content="go"))
        await asyncio.wait_for(wait.started.wait(), timeout=5)
        await asyncio.wait_for(spawn.started.wait(), timeout=5)
        started = agent.runtime.jobs["w1"].started

        shown = rows(team, focused="Agent", now=started + 42)

        assert [(r.branch, r.text, r.state, r.detail, r.call_id) for r in shown] == [
            ("", "● Agent", "busy", "$0.00", ""),
            ("├─ ", "Wait tests", "running", "42s", "w1"),
            ("└─ ", "kid", "idle", "$0.00", ""),
        ]
        agent.runtime.halt()


@pytest.mark.asyncio
async def test_a_job_whose_tool_is_gone_shows_its_name() -> None:
    wait = _Wait()
    team = Team(catalog=ModelCatalog(factories={}))
    provider = ScriptedProvider(replies=[reply(calls=[call("Wait", "w1")])])
    async with Agent(model=provider["scripted"], tools=[wait]) as agent:
        _ = team.join(agent)
        agent.runtime.push(UserMessage(content="go"))
        await asyncio.wait_for(wait.started.wait(), timeout=5)
        agent.tools = {}

        (_, job) = rows(team, focused="Agent")

        assert (job.branch, job.text) == ("└─ ", "Wait")
        agent.runtime.halt()


def test_the_cursor_row_is_reversed_and_long_text_is_cut() -> None:
    shown = [
        Row(label="a", branch="", text="a", state="idle", detail="$0.00"),
        Row(label="b", branch="└─ ", text="b" * 100, state="busy", detail="$1.00"),
    ]

    lines = render(shown, cursor=1, note="stopping b")

    styles = [part[0] for part in lines if part[1].strip()]
    texts = [part[1] for part in lines]
    assert styles == ["", "class:tree.selected", "class:tree.help"]
    assert texts[0] == f" {'a':<72}  idle         $0.00 "
    assert texts[2] == f" {'└─ ' + 'b' * 68}…  busy         $1.00 "
    assert texts[-1].endswith("→ back · stopping b\n")


def test_short_rows_draw_at_least_32_wide_and_a_full_row_is_not_cut() -> None:
    assert render(
        [
            Row(label="a", branch="", text="a", state="idle", detail=""),
            Row(label="b", branch="", text="b" * 33, state="idle", detail=""),
        ],
        cursor=0,
    ) == FormattedText(
        [
            ("class:tree.selected", f" {'a':<33}  idle               "),
            ("", "\n"),
            ("", f" {'b' * 33}  idle               "),
            ("", "\n"),
            ("class:tree.help", "\n ↑↓ move · Enter focus · Del stop · → back\n"),
        ],
    )
    (first, *_) = render(
        [Row(label="a", branch="", text="a", state="idle", detail="")],
        cursor=0,
    )
    assert first[1] == f" {'a':<32}  idle               "


def test_with_no_rows_only_the_keys_are_drawn() -> None:
    assert render([], cursor=0) == FormattedText(
        [("class:tree.help", "\n ↑↓ move · Enter focus · Del stop · → back\n")],
    )


@dataclass(slots=True, kw_only=True)
class _Said:
    """What Delete asked for; a stop takes the agent out of ``team``, as a real one does."""

    team: Team | None = None
    stopped: list[str] = field(default_factory=list[str])
    cancelled: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    def stop(self, label: str) -> str:
        self.stopped.append(label)
        if self.team is not None:
            _ = self.team.leave(label)
        return f"stopping {label}"

    def cancel(self, label: str, call_id: str) -> str:
        self.cancelled.append((label, call_id))
        return f"cancelled {call_id}"


async def _pick(
    team: Team,
    keys: str,
    *,
    focused: str = "kid",
    said: _Said | None = None,
) -> tuple[str | None, _Said]:
    said = said if said is not None else _Said()
    with (
        create_pipe_input() as pipe,
        create_app_session(input=pipe, output=DummyOutput()),
    ):
        pipe.send_text(keys)
        picked = await asyncio.wait_for(
            pick(team, focused=focused, stop=said.stop, cancel=said.cancel),
            timeout=5,
        )
    return picked, said


# ``then`` changes the tree mid-way; with ``team_leaves``, an agent Delete stops leaves
# ``team``.
async def _drawn(
    team: Team,
    keys: str,
    *,
    focused: str,
    then: Callable[[], None] | None = None,
    painted: asyncio.Event | None = None,
    team_leaves: bool = False,
    theme: Theme | None = None,
) -> tuple[str, str]:
    """Return what ``pick`` writes to a terminal before ``keys``, and in all."""
    written = StringIO()
    output = Vt100_Output(written, lambda: Size(rows=24, columns=80), term="xterm")
    said = _Said(team=team if team_leaves else None)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=output):
        picking = asyncio.ensure_future(
            pick(
                team,
                focused=focused,
                stop=said.stop,
                cancel=said.cancel,
                theme=theme,
            ),
        )
        if then is not None:
            assert painted is not None
            await asyncio.wait_for(painted.wait(), timeout=5)
            painted.clear()
            then()
            await asyncio.wait_for(painted.wait(), timeout=5)
        before = written.getvalue()
        pipe.send_text(keys)
        _ = await asyncio.wait_for(picking, timeout=5)
    return before, written.getvalue()


_UP, _DOWN, _RIGHT, _DELETE = "\x1b[A", "\x1b[B", "\x1b[C", "\x1b[3~"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("keys", "picked"),
    [
        ("\r", "kid"),
        (f"{_DOWN}\r", "grand"),
        (f"{_DOWN * 9}\r", "other"),
        (f"{_UP * 9}\r", "Agent"),
        (f"{_DOWN}{_DOWN}{_UP}\r", "grand"),
        (f"{_DOWN * 9}{_UP}\r", "grand"),
        (_RIGHT, None),
        ("\x03", None),
    ],
)
async def test_the_cursor_starts_on_the_shown_agent_and_enter_picks(
    keys: str,
    picked: str | None,
) -> None:
    assert (await _pick(_team(), keys))[0] == picked


@pytest.mark.asyncio
async def test_a_shown_agent_not_in_the_tree_starts_the_cursor_on_the_first_row() -> (
    None
):
    assert (await _pick(_team(), "\r", focused="gone"))[0] == "Agent"


@pytest.mark.asyncio
async def test_the_tree_takes_the_full_screen_with_the_cursor_hidden() -> None:
    _, drawn = await _drawn(_team(), _RIGHT, focused="kid")

    opened, _, rest = drawn.partition("\x1b[?1049h")
    shown, _, closed = rest.partition("\x1b[?1049l")
    assert "\x1b[0;7m ├─ ● kid " in shown
    assert "\x1b[?25h" not in shown
    assert closed.endswith("\x1b[?25h")
    assert opened


@pytest.mark.asyncio
async def test_custom_theme_colors_reach_the_live_picker() -> None:
    path = config_dir() / "rekursiv-ai" / "sagent" / "themes" / "tree-test.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        "preset: terminal\nstyles:\n  selected: magenta\n  status: cyan\n",
    )
    _, drawn = await _drawn(
        _team(),
        _RIGHT,
        focused="kid",
        theme=load_theme("tree-test"),
    )
    assert "\x1b[0;35m ├─ ● kid " in drawn
    assert "\x1b[0;36m" in drawn
    assert "↑↓ move" in drawn


@pytest.mark.asyncio
async def test_a_tree_that_changes_while_shown_is_redrawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team = _team()
    painted = asyncio.Event()

    def after_render(app: Application[str | None]) -> None:
        del app
        painted.set()

    monkeypatch.setattr(
        tree,
        "Application",
        partial(Application, after_render=after_render),
    )
    monkeypatch.setattr(
        tree,
        "repaint_on_change",
        partial(repaint_on_change, interval_sec=0),
    )

    def spend() -> None:
        team["other"].spend_usd = 0.5

    before_keys, _ = await _drawn(
        team,
        _RIGHT,
        focused="kid",
        then=spend,
        painted=painted,
    )

    # The first frame ends with the cursor sent back up the screen; a redraw
    # writes only what changed: ``$0.00`` became ``$0.50``.
    first, _, redrawn = before_keys.partition("\x1b[23A")
    assert "$0.00" in first
    assert "5" in redrawn


@pytest.mark.asyncio
async def test_what_delete_did_is_said_and_the_cursor_falls_back_a_row() -> None:
    team = _team()

    _, drawn = await _drawn(
        team,
        f"{_DELETE}{_RIGHT}",
        focused="other",
        team_leaves=True,
    )

    # Before the Delete, grand hangs from ``│  ``: these are drawn only after.
    assert "\x1b[0;7m    └─ grand " in drawn
    assert "→ back · stopping other" in drawn


@pytest.mark.asyncio
async def test_a_lone_root_is_drawn_under_the_cursor() -> None:
    team = Team(catalog=ModelCatalog(factories={}))
    _ = team.join(Agent(model=ScriptedProvider()["scripted"]))

    _, drawn = await _drawn(team, _RIGHT, focused="Agent")

    assert "\x1b[0;7m ● Agent " in drawn


@pytest.mark.asyncio
async def test_delete_stops_the_agent_of_its_row() -> None:
    _, said = await _pick(_team(), f"{_DOWN}{_DELETE}{_RIGHT}")
    assert (said.stopped, said.cancelled) == (["grand"], [])


@pytest.mark.asyncio
async def test_delete_on_a_job_cancels_it() -> None:
    wait = _Wait()
    team = Team(catalog=ModelCatalog(factories={}))
    provider = ScriptedProvider(replies=[reply(calls=[call("Wait", "w1")])])
    async with Agent(model=provider["scripted"], tools=[wait]) as agent:
        _ = team.join(agent)
        agent.runtime.push(UserMessage(content="go"))
        await asyncio.wait_for(wait.started.wait(), timeout=5)

        _, said = await _pick(team, f"{_DOWN}{_DELETE}{_RIGHT}", focused="Agent")

        assert (said.stopped, said.cancelled) == ([], [("Agent", "w1")])
        agent.runtime.halt()


@pytest.mark.asyncio
async def test_when_the_last_row_leaves_the_cursor_falls_back_a_row() -> None:
    team = _team()
    picked, said = await _pick(
        team,
        f"{_DELETE}\r",
        focused="other",
        said=_Said(team=team),
    )
    assert (picked, said.stopped) == ("grand", ["other"])


@pytest.mark.asyncio
async def test_with_every_agent_gone_enter_and_delete_do_nothing() -> None:
    team = Team(catalog=ModelCatalog(factories={}))
    _ = team.join(Agent(model=ScriptedProvider()["scripted"]))
    picked, said = await _pick(
        team,
        f"{_DELETE}{_DELETE}\r{_RIGHT}",
        focused="Agent",
        said=_Said(team=team),
    )
    assert (picked, said.stopped) == (None, ["Agent"])


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
