"""Tests for persistent-browser tool forwarding and presentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import asyncio
import threading

from treekle import from_plain, json_loads
from wesearch import browser
from wesearch.browser import (
    BrowserElement,
    BrowserError,
    BrowserMode,
    BrowserResult,
    BrowserTab,
)
from wesearch.fetch import FetchSession

import pytest

from sagent.agent.team import Team
from sagent.model.scripted import ScriptedProvider
from sagent.providers.catalog import ModelCatalog
from sagent.session.types import WebBrowserResult
from sagent.tools.builtin import builders
from sagent.tools.remote import RemoteSession
from sagent.tools.types import ToolContext


if TYPE_CHECKING:
    from pathlib import Path

    from sagent.tools.types import Tool


@dataclass(slots=True, kw_only=True)
class Calls:
    """Record calls at the browser boundary without starting Chrome."""

    seen: list[dict[str, object]] = field(default_factory=list)
    result: BrowserResult = field(default_factory=BrowserResult)
    error: Exception | None = None

    def __call__(self, action: str, **kwargs: object) -> BrowserResult:
        self.seen.append({"action": action, **kwargs})
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Calls:
    calls = Calls()
    monkeypatch.setattr(browser, "browser", calls)
    return calls


@pytest.fixture
def tool(tmp_path: Path) -> Tool:
    built = builders(
        cwd=tmp_path,
        team=Team(catalog=ModelCatalog(factories={"scripted": ScriptedProvider})),
        remote=RemoteSession([]),
    )
    assert "WebBrowser" in built
    return built["WebBrowser"]()


def test_schema_and_secret_safe_summary(tool: Tool) -> None:
    props = from_plain(tool.schema["properties"], dict[str, object])
    assert set(props) == {
        "action",
        "tab_id",
        "url",
        "selector",
        "text",
        "key",
        "max_chars",
        "timeout_sec",
        "mode",
        "restart",
    }
    assert tool.schema["required"] == ["action"]
    assert (
        tool.summary(
            {
                "action": "fill",
                "tab_id": "t",
                "selector": "#password",
                "text": "secret",
                "key": "secret",
            },
        )
        == "WebBrowser fill t #password"
    )
    assert all(
        word in tool.description
        for word in ("approval", "purchases", "terms", "account")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    [
        "tabs",
        "open",
        "navigate",
        "inspect",
        "click",
        "fill",
        "press",
        "screenshot",
        "close",
    ],
)
async def test_forwarding_and_session_reuse(
    tool: Tool,
    calls: Calls,
    action: str,
) -> None:
    context = ToolContext(fetch_session=FetchSession())
    arguments = {
        "action": action,
        "tab_id": "t",
        "url": "",
        "selector": "#p",
        "text": "secret",
        "key": "",
        "max_chars": 17,
        "timeout_sec": 4.5,
    }
    for _ in range(2):
        await tool.call(context, arguments)
    assert (
        calls.seen
        == [
            {
                **arguments,
                "mode": None,
                "restart": False,
                "session": context.fetch_session,
            },
        ]
        * 2
    )


@pytest.mark.asyncio
async def test_inspection_is_structured_and_image_is_an_attachment(
    tool: Tool,
    calls: Calls,
) -> None:
    tab = BrowserTab(tab_id="t", url="https://example.com", title="Page")
    calls.result = BrowserResult(
        tab=tab,
        tabs=(tab,),
        text="bounded",
        truncated=True,
        elements=(
            BrowserElement(
                selector="#p",
                tag="input",
                role="textbox",
                label="Password",
                kind="password",
            ),
        ),
        screenshot=b"png bytes",
    )
    result = await tool.call(ToolContext(), {"action": "screenshot", "tab_id": "t"})
    assert isinstance(result, WebBrowserResult)
    assert result.action == "screenshot"
    body = json_loads(result.text(), dict[str, object])
    assert body == {
        "tab": {"tab_id": "t", "url": "https://example.com", "title": "Page"},
        "tabs": [{"tab_id": "t", "url": "https://example.com", "title": "Page"}],
        "text": "bounded",
        "truncated": True,
        "mode": None,
        "restarted": False,
        "elements": [
            {
                "selector": "#p",
                "tag": "input",
                "role": "textbox",
                "label": "Password",
                "kind": "password",
            },
        ],
    }
    assert [(a.mime_descriptor, a.data) for a in result.attachments] == [
        ("image/png", b"png bytes"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [ValueError("Unknown tab"), BrowserError("Unknown tab"), OSError("Unknown tab")],
)
async def test_errors_are_tool_results(
    tool: Tool,
    calls: Calls,
    error: Exception,
) -> None:
    calls.error = error
    result = await tool.call(ToolContext(), {"action": "inspect", "tab_id": "gone"})
    assert isinstance(result, WebBrowserResult)
    assert result.action == "inspect"
    assert result.call_id == ""
    assert result.is_error is True
    assert "Unknown tab" in result.text()


def test_serialization_is_per_tab_not_global(tool: Tool) -> None:
    context = ToolContext()
    keys = [
        tool.serialize_key(context, {"action": action, "tab_id": "t"})
        for action in ("inspect", "fill", "close")
    ]
    assert keys[0] is not None
    assert keys[0] != "*"
    assert keys == [keys[0]] * 3
    assert (
        tool.serialize_key(context, {"action": "inspect", "tab_id": "other"}) != keys[0]
    )
    assert (
        tool.serialize_key(context, {"action": "open", "url": "https://example.com"})
        is None
    )
    assert tool.serialize_key(context, {"action": "tabs"}) is None
    assert tool.serialize_key(context, {"action": "tabs", "tab_id": ""}) is None
    assert tool.serialize_key(context, {"action": "inspect", "tab_id": 3}) is None


@pytest.mark.asyncio
async def test_independent_tabs_do_not_block_the_event_loop(
    tool: Tool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2, timeout=2)

    def act(action: str, **kwargs: object) -> BrowserResult:
        del action, kwargs
        barrier.wait()
        return BrowserResult()

    monkeypatch.setattr(browser, "browser", act)
    context = ToolContext()
    results = await asyncio.gather(
        *(
            tool.call(context, {"action": "inspect", "tab_id": tab})
            for tab in ("a", "b")
        ),
    )
    assert all(result.is_error is not True for result in results)


@pytest.mark.asyncio
async def test_defaults_and_no_image(tool: Tool, calls: Calls) -> None:
    context = ToolContext()
    result = await tool.call(context, {"action": "tabs"})
    assert result.call_id == ""
    assert result.attachments == ()
    assert calls.seen == [
        {
            "action": "tabs",
            "tab_id": "",
            "url": "",
            "selector": "",
            "text": "",
            "key": "",
            "mode": None,
            "restart": False,
            "max_chars": 20_000,
            "timeout_sec": 30,
            "session": context.fetch_session,
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "headed", "headless"])
@pytest.mark.parametrize("restart", [False, True])
async def test_mode_control_forwards_without_owning_lifecycle(
    tool: Tool,
    calls: Calls,
    mode: BrowserMode | None,
    restart: bool,
) -> None:
    calls.result = BrowserResult(mode=mode, restarted=restart)
    context = ToolContext()
    result = await tool.call(
        context,
        {"action": "mode", "mode": mode, "restart": restart},
    )
    body = json_loads(result.text(), dict[str, object])
    assert body["mode"] == mode
    assert body["restarted"] is restart
    assert calls.seen[0]["mode"] == mode
    assert calls.seen[0]["restart"] is restart
    assert calls.seen[0]["session"] is context.fetch_session


def test_mode_summary_and_restart_serialization(tool: Tool) -> None:
    args = {"action": "mode", "mode": "headed", "restart": True, "text": "secret"}
    assert tool.summary(args) == "WebBrowser mode headed"
    assert tool.serialize_key(ToolContext(), args) == "*"
    assert tool.serialize_key(ToolContext(), {"action": "mode"}) is None
    assert (
        tool.serialize_key(ToolContext(), {"action": "mode", "mode": "headed"}) is None
    )
    assert "consent" in tool.description
    assert "restart" in tool.description


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
