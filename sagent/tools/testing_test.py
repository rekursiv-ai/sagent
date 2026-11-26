"""Tests for the stdio MCP server tests run."""

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict

import json
import subprocess
import sys

from treekle import PlainTree, from_plain, json_loads

from sagent.tools.testing import SHOUT_SERVER


if TYPE_CHECKING:
    from pathlib import Path


class _Error(TypedDict):
    code: int


class _Reply(TypedDict):
    """One JSON-RPC reply: a result, or an error."""

    id: int
    result: NotRequired[dict[str, PlainTree]]
    error: NotRequired[_Error]


class _Tool(TypedDict):
    name: str


class _Listed(TypedDict):
    tools: list[_Tool]


class _Initialized(TypedDict):
    protocolVersion: str


class _Called(TypedDict):
    content: list[dict[str, str]]


def test_the_server_answers_each_request_and_refuses_an_unknown_method(
    tmp_path: Path,
) -> None:
    script = tmp_path / "server.py"
    script.write_text(SHOUT_SERVER)
    requests = [
        {"id": 1, "method": "initialize", "params": {"protocolVersion": "v"}},
        {"method": "notifications/initialized"},
        {"id": 2, "method": "tools/list"},
        {"id": 3, "method": "tools/call", "params": {"arguments": {"text": "hi"}}},
        {"id": 4, "method": "ping"},
        {"id": 5, "method": "resources/list"},
    ]
    done = subprocess.run(  # noqa: S603 -- fixed interpreter and script.
        [sys.executable, str(script)],
        input="".join(json.dumps(r) + "\n" for r in requests),
        capture_output=True,
        text=True,
        check=True,
    )
    initialized, listed, called, pinged, refused = (
        json_loads(line, _Reply) for line in done.stdout.splitlines()
    )

    assert [r["id"] for r in (initialized, listed, called, pinged, refused)] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert from_plain(initialized.get("result"), _Initialized) == {
        "protocolVersion": "v",
    }
    tools = from_plain(listed.get("result"), _Listed)["tools"]
    assert [t["name"] for t in tools] == ["shout"]
    assert from_plain(called.get("result"), _Called)["content"] == [
        {"type": "text", "text": "HI"},
    ]
    assert pinged.get("result") == {}
    assert refused.get("error") == {"code": -32_601}
    assert done.stderr == "server started\n"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
