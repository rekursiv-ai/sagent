"""Tests for the tree screen: its rows, how they draw, and its keys."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import asyncio

from prompt_toolkit.formatted_text import FormattedText, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl

import pytest

from sagent.agent.types import Agent
from sagent.session.types import ToolCall
from sagent.tui import tree
from sagent.tui.theme import load_theme
from sagent.tui.tree import Row, pick, render, rows


if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(slots=True, kw_only=True)
class _Runtime:
    jobs: dict[str, SimpleNamespace] = field(default_factory=dict[str, SimpleNamespace])
    cohort: set[str] = field(default_factory=set[str])
    model_call: object = None
    compact_task: object = None


class _Agent(MagicMock):
    runtime: _Runtime
    tools: dict[str, SimpleNamespace]
    spend_usd: float


@dataclass(slots=True, kw_only=True)
class _Team:
    agents: dict[str, _Agent] = field(default_factory=dict[str, _Agent])
    parents: dict[str, str | None] = field(default_factory=dict[str, str | None])

    def __iter__(self) -> Iterator[str]:
        return iter(self.agents)

    def __getitem__(self, label: str) -> _Agent:
        return self.agents[label]

    def parent(self, label: str) -> str | None:
        return self.parents[label]


def _team() -> _Team:
    parents = {"Agent": None, "kid": "Agent", "other": "Agent", "grand": "kid"}
    return _Team(
        parents=parents,
        agents={
            label: _Agent(
                spec=Agent,
                spend_usd=0.0,
                tools={},
                runtime=_Runtime(),
            )
            for label in parents
        },
    )


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
    team["kid"].runtime.model_call = object()
    team["other"].runtime.compact_task = object()
    assert [r.state for r in rows(team, focused="kid", now=0.0)] == [
        "idle",
        "busy",
        "idle",
        "busy",
    ]


def test_a_last_child_s_job_hangs_from_a_blank_stem_and_never_runs_backwards() -> None:
    team = _team()
    del team.agents["grand"], team.agents["other"]
    team["kid"].runtime.jobs = {
        "w1": SimpleNamespace(
            call=ToolCall(call_id="w1", name="Wait", arguments={}),
            started=10.0,
            state="running",
        ),
    }
    team["kid"].tools = {
        "Wait": SimpleNamespace(summary=MagicMock(return_value="Wait")),
    }
    shown = rows(team, focused="Agent", now=5.0)
    assert [(r.branch, r.text, r.detail) for r in shown] == [
        ("", "● Agent", "$0.00"),
        ("└─ ", "kid", "$0.00"),
        ("   └─ ", "Wait", "0s"),
    ]


def test_a_busy_agent_lists_its_jobs_but_not_its_spawns() -> None:
    team = _team()
    del team.agents["grand"], team.agents["other"]
    team["Agent"].runtime.cohort = {"w1", "s1"}
    team["Agent"].runtime.jobs = {
        call_id: SimpleNamespace(
            call=ToolCall(call_id=call_id, name=name, arguments={"on": "tests"}),
            started=10.0,
            state="running",
        )
        for call_id, name in (("w1", "Wait"), ("s1", "AgentSpawn"))
    }
    summary = MagicMock(return_value="Wait tests")
    team["Agent"].tools = {"Wait": SimpleNamespace(summary=summary)}
    shown = rows(team, focused="Agent", now=52.0)
    assert [(r.branch, r.text, r.state, r.detail, r.call_id) for r in shown] == [
        ("", "● Agent", "busy", "$0.00", ""),
        ("├─ ", "Wait tests", "running", "42s", "w1"),
        ("└─ ", "kid", "idle", "$0.00", ""),
    ]
    summary.assert_called_once_with({"on": "tests"})


def test_a_job_whose_tool_is_gone_shows_its_name() -> None:
    team = _team()
    team.agents = {"Agent": team["Agent"]}
    team["Agent"].runtime.jobs = {
        "w1": SimpleNamespace(
            call=ToolCall(call_id="w1", name="Wait", arguments={}),
            started=0.0,
            state="detached",
        ),
    }
    _, job = rows(team, focused="Agent")
    assert (job.branch, job.text, job.state) == ("└─ ", "Wait", "detached")


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
    first, *_ = render(
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
    team: _Team | None = None
    stopped: list[str] = field(default_factory=list[str])
    cancelled: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    def stop(self, label: str) -> str:
        self.stopped.append(label)
        if self.team is not None:
            del self.team.agents[label]
        return f"stopping {label}"

    def cancel(self, label: str, call_id: str) -> str:
        self.cancelled.append((label, call_id))
        return f"cancelled {call_id}"


def _press(screen: tree._Screen, *keys: str) -> MagicMock:
    exiting = MagicMock()
    event = MagicMock(spec=KeyPressEvent, app=SimpleNamespace(exit=exiting))
    bindings = screen.keys()
    for key in keys:
        (binding,) = bindings.get_bindings_for_keys((key,))
        binding.handler(event)
    return exiting


@pytest.mark.parametrize(
    ("keys", "picked"),
    [
        (("c-m",), "kid"),
        (("down", "c-m"), "grand"),
        ((*["down"] * 4, "c-m"), "other"),
        ((*["up"] * 2, "c-m"), "Agent"),
        (("down", "down", "up", "c-m"), "grand"),
        ((*["down"] * 4, "up", "c-m"), "grand"),
        (("right",), None),
        (("c-c",), None),
    ],
)
def test_the_cursor_starts_on_the_shown_agent_and_enter_picks(
    keys: tuple[str, ...],
    picked: str | None,
) -> None:
    said = _Said()
    screen = tree._Screen(
        team=_team(),
        focused="kid",
        stop=said.stop,
        cancel=said.cancel,
    )
    _press(screen, *keys).assert_called_once_with(result=picked)


def test_a_shown_agent_not_in_the_tree_starts_the_cursor_on_the_first_row() -> None:
    said = _Said()
    screen = tree._Screen(
        team=_team(),
        focused="gone",
        stop=said.stop,
        cancel=said.cancel,
    )
    _press(screen, "c-m").assert_called_once_with(result="Agent")


@pytest.mark.asyncio
async def test_the_tree_takes_the_full_screen_with_the_cursor_hidden() -> None:
    team = _team()
    said = _Said()
    before = asyncio.all_tasks()
    invalidate = MagicMock()
    made = SimpleNamespace(
        run_async=AsyncMock(return_value="kid"),
        invalidate=invalidate,
    )
    repaint = AsyncMock()
    with (
        patch.object(tree, "Application", return_value=made) as application,
        patch.object(tree, "repaint_on_change", repaint),
    ):
        assert (
            await pick(team, focused="kid", stop=said.stop, cancel=said.cancel) == "kid"
        )
    options = application.call_args.kwargs
    assert options["full_screen"] is True
    window = cast(Layout, options["layout"]).container
    assert isinstance(window, Window)
    assert window.always_hide_cursor()
    control = window.content
    assert isinstance(control, FormattedTextControl)
    assert any(
        part[0] == "class:tree.selected" and "├─ ● kid" in part[1]
        for part in to_formatted_text(control.text)
    )
    keys = cast(KeyBindings, options["key_bindings"])
    (down,) = keys.get_bindings_for_keys(("down",))
    down.handler(MagicMock(spec=KeyPressEvent))
    assert any(
        part[0] == "class:tree.selected" and "grand" in part[1]
        for part in to_formatted_text(control.text)
    )
    (delete,) = keys.get_bindings_for_keys(("delete",))
    delete.handler(MagicMock(spec=KeyPressEvent))
    assert said.stopped == ["grand"]
    team["grand"].runtime.jobs = {
        "w1": SimpleNamespace(
            call=ToolCall(call_id="w1", name="Wait", arguments={}),
            started=0.0,
            state="running",
        ),
    }
    down.handler(MagicMock(spec=KeyPressEvent))
    delete.handler(MagicMock(spec=KeyPressEvent))
    assert said.cancelled == [("grand", "w1")]
    repaint.assert_called_once_with(invalidate, control.text)
    assert options["style"] is load_theme().prompt_style()
    (redraw,) = asyncio.all_tasks() - before
    assert redraw.cancelling() == 1
    with pytest.raises(asyncio.CancelledError):
        await redraw


@pytest.mark.asyncio
async def test_custom_theme_colors_reach_the_picker() -> None:
    said = _Said()
    theme = load_theme("terminal")
    before = asyncio.all_tasks()
    made = SimpleNamespace(
        run_async=AsyncMock(side_effect=RuntimeError("closed")),
        invalidate=MagicMock(),
    )
    with (
        patch.object(tree, "Application", return_value=made) as application,
        pytest.raises(RuntimeError, match="closed"),
    ):
        await pick(
            _team(),
            focused="kid",
            stop=said.stop,
            cancel=said.cancel,
            theme=theme,
        )
    assert application.call_args.kwargs["style"] is theme.prompt_style()
    (redraw,) = asyncio.all_tasks() - before
    assert redraw.cancelling() == 1
    with pytest.raises(asyncio.CancelledError):
        await redraw


def test_a_tree_that_changes_while_shown_is_redrawn() -> None:
    team = _team()
    said = _Said()
    screen = tree._Screen(team=team, focused="kid", stop=said.stop, cancel=said.cancel)
    before = screen.text()
    team["other"].spend_usd = 0.5
    after = screen.text()
    assert before != after
    assert "$0.00" in before[-3][1]
    assert "$0.50" in after[-3][1]


def test_what_delete_did_is_said_and_the_cursor_falls_back_a_row() -> None:
    team = _team()
    said = _Said(team=team)
    screen = tree._Screen(
        team=team,
        focused="other",
        stop=said.stop,
        cancel=said.cancel,
    )
    _press(screen, "delete")
    drawn = screen.text()
    assert drawn[-3][0] == "class:tree.selected"
    assert "   └─ grand" in drawn[-3][1]
    assert drawn[-1][1].endswith("→ back · stopping other\n")


def test_a_lone_root_is_drawn_under_the_cursor() -> None:
    team = _team()
    team.agents = {"Agent": team["Agent"]}
    said = _Said()
    screen = tree._Screen(
        team=team,
        focused="Agent",
        stop=said.stop,
        cancel=said.cancel,
    )
    first, *_ = screen.text()
    assert first == ("class:tree.selected", f" {'● Agent':<32}  idle         $0.00 ")


def test_delete_stops_the_agent_of_its_row() -> None:
    said = _Said()
    screen = tree._Screen(
        team=_team(),
        focused="kid",
        stop=said.stop,
        cancel=said.cancel,
    )
    _press(screen, "down", "delete", "right").assert_called_once_with(result=None)
    assert (said.stopped, said.cancelled) == (["grand"], [])


def test_delete_on_a_job_cancels_it() -> None:
    team = _team()
    team["Agent"].runtime.jobs = {
        "w1": SimpleNamespace(
            call=ToolCall(call_id="w1", name="Wait", arguments={}),
            started=0.0,
            state="running",
        ),
    }
    said = _Said()
    screen = tree._Screen(
        team=team,
        focused="Agent",
        stop=said.stop,
        cancel=said.cancel,
    )
    _press(screen, "down", "delete", "right").assert_called_once_with(result=None)
    assert (said.stopped, said.cancelled) == ([], [("Agent", "w1")])


def test_when_the_last_row_leaves_the_cursor_falls_back_a_row() -> None:
    team = _team()
    said = _Said(team=team)
    screen = tree._Screen(
        team=team,
        focused="other",
        stop=said.stop,
        cancel=said.cancel,
    )
    _press(screen, "delete", "c-m").assert_called_once_with(result="grand")
    assert said.stopped == ["other"]


def test_with_every_agent_gone_enter_and_delete_do_nothing() -> None:
    team = _team()
    team.agents = {"Agent": team["Agent"]}
    said = _Said(team=team)
    screen = tree._Screen(
        team=team,
        focused="Agent",
        stop=said.stop,
        cancel=said.cancel,
    )
    _press(screen, "delete", "delete", "c-m").assert_not_called()
    assert screen.text() == render([], cursor=0, note="stopping Agent")
    _press(screen, "right").assert_called_once_with(result=None)
    assert said.stopped == ["Agent"]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
