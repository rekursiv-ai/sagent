"""Tests for the remote actions: what ``sagent remote VERB`` and ``/remote.VERB`` run."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from sagent.settings.harvest import harvest
from sagent.tools.remote import Endpoint, Link, SavedRemote, saved
from sagent.tools.remote_settings import RemoteSettings


if TYPE_CHECKING:
    from sagent.settings.types import Setting


_PIN = "SHA256:" + "a" * 64


def _actions(connect_sec: float = 3.0) -> dict[str, Setting]:
    owner = RemoteSettings(connect_sec=connect_sec)
    return {".".join(s.name): s for s in harvest([(("remote",), owner)])}


def test_every_verb_is_an_action_the_command_line_reaches() -> None:
    actions = _actions()
    assert list(actions) == ["remote.add", "remote.del", "remote.list"]
    for action in actions.values():
        assert action.cli is True
        assert action.help_long.startswith(f"usage: sagent {' '.join(action.name)}")
    assert "SAGENT_URL" in actions["remote.add"].usage


@pytest.mark.asyncio
async def test_add_saves_a_string_and_replaces_one_at_that_address() -> None:
    add = _actions()["remote.add"]
    said = await add.apply(f"sagent://gpu1:7777/{_PIN}/one --tools Bash,Read")
    assert said == "Saved gpu1:7777; sessions started from now reach it."
    _ = await add.apply(f"sagent://gpu2:7777/{_PIN}/two")
    _ = await add.apply(f"sagent://gpu1:7777/{_PIN}/three")
    assert saved() == [
        SavedRemote(url=f"sagent://gpu2:7777/{_PIN}/two"),
        SavedRemote(url=f"sagent://gpu1:7777/{_PIN}/three"),
    ]


@pytest.mark.asyncio
async def test_del_forgets_a_host_and_names_the_saved_ones() -> None:
    actions = _actions()
    _ = await actions["remote.add"].apply(f"sagent://gpu1:7777/{_PIN}/one")
    with pytest.raises(ValueError, match="No saved server 'gpu9'; saved: gpu1"):
        _ = await actions["remote.del"].apply("gpu9")
    assert await actions["remote.del"].apply("gpu1") == (
        "Forgot gpu1; sessions started from now go without it."
    )
    assert saved() == []


@pytest.mark.asyncio
async def test_a_bad_url_is_a_setting_error() -> None:
    with pytest.raises(ValueError, match="is no sagent://"):
        _ = await _actions()["remote.add"].apply("http://gpu1")


@pytest.mark.asyncio
async def test_list_says_which_answer() -> None:
    actions = _actions(connect_sec=0.5)
    assert await actions["remote.list"].apply("") == (
        "No servers. Save one with `sagent remote add`; see its --help."
    )
    _ = await actions["remote.add"].apply(f"sagent://127.0.0.1:7777/{_PIN}/one")
    _ = await actions["remote.add"].apply(f"sagent://127.0.0.1:1/{_PIN}/two")
    wait_up, wait_down = AsyncMock(), AsyncMock()
    up = Mock(
        spec=Endpoint,
        where="127.0.0.1:7777",
        up=True,
        tools={"Read": "digest"},
        until=wait_up,
    )
    down = Mock(
        spec=Endpoint,
        where="127.0.0.1:1",
        up=False,
        error="connection refused",
        until=wait_down,
    )
    session = Mock(endpoints=Mock(return_value={"gpu1": up, "127.0.0.1": down}))
    enter, exit_session = AsyncMock(return_value=session), AsyncMock()
    lifecycle = MagicMock(__aenter__=enter, __aexit__=exit_session)
    with patch(
        "sagent.tools.remote_settings.RemoteSession",
        return_value=lifecycle,
    ) as build:
        listed = (await actions["remote.list"].apply("")).splitlines()
    build.assert_called_once()
    links = cast(list[Link], build.call_args.args[0])
    assert [(link.where, link.token) for link in links] == [
        ("127.0.0.1:7777", "one"),
        ("127.0.0.1:1", "two"),
    ]
    assert build.call_args.kwargs == {"retry_sec": 0.5}
    wait_up.assert_awaited_once()
    wait_down.assert_awaited_once()
    enter.assert_awaited_once()
    exit_session.assert_awaited_once()
    assert listed == [
        "gpu1  127.0.0.1:7777  up: Read",
        "127.0.0.1  127.0.0.1:1  down: connection refused",
    ]


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
