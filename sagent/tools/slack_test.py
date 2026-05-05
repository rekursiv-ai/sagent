"""Tests for tools.slack."""

from __future__ import annotations

from typing import Any

import json as _json

import pytest

from sagent.custom_types import (
    JsonMessage,
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.json import JSON, json_freeze
from sagent.tools import slack as slack_mod


def _msg(directive: JSON) -> Message:
    return MultipartMessage(
        (JsonMessage(directive, "application/x-tool-slack"),),
        "multipart/x-tool-call",
    )


class _Calls:
    """Tracks mock fetch invocations."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Any]] = []


def _patch_client(
    monkeypatch: pytest.MonkeyPatch,
    canned: dict[str, dict[str, Any]],
) -> _Calls:
    """Mock ``fetch`` for Slack API calls."""
    tracker = _Calls()

    def _mock_fetch(
        url: str, *, method: str = "GET", json: Any = None, **kwargs: object
    ) -> bytes:
        del kwargs
        slack_method = url.rsplit("/api/", maxsplit=1)[-1].split("?", maxsplit=1)[0]
        tracker.calls.append((method, url, json))
        if slack_method in canned:
            return _json.dumps({"ok": True, **canned[slack_method]}).encode()
        return _json.dumps({"ok": False, "error": "no_canned"}).encode()

    monkeypatch.setattr("sagent.tools.slack.fetch", _mock_fetch)
    return tracker


_TEST_TOKEN = "xoxb-" + "test"


def _tool(
    *,
    token: str = _TEST_TOKEN,
    username: str = "",
    icon_url: str = "",
) -> slack_mod.Slack:
    return slack_mod.Slack(token=token, username=username, icon_url=icon_url)


class TestSend:
    @pytest.mark.anyio
    async def test_requires_channel_and_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(monkeypatch, {})
        resp = await _tool().run(_msg(json_freeze({"operation": "send"})))
        assert resp.descriptor == "text/x-error"
        assert "required" in str(resp.content)

    @pytest.mark.anyio
    async def test_sends(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            {"chat.postMessage": {"ts": "12345.67", "channel": "C1"}},
        )
        resp = await _tool().run(
            _msg(json_freeze({"operation": "send", "channel": "C1", "text": "hi"}))
        )
        assert isinstance(resp, TextMessage)
        assert "12345.67" in resp.content
        _, _, body = mock.calls[0]
        assert body["channel"] == "C1"
        assert body["text"] == "hi"

    @pytest.mark.anyio
    async def test_thread(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            {"chat.postMessage": {"ts": "t", "channel": "C1"}},
        )
        await _tool().run(
            _msg(
                json_freeze(
                    {
                        "operation": "send",
                        "channel": "C1",
                        "text": "reply",
                        "thread_ts": "111.222",
                    }
                )
            )
        )
        _, _, body = mock.calls[0]
        assert body["thread_ts"] == "111.222"

    @pytest.mark.anyio
    async def test_identity_fields(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            {"chat.postMessage": {"ts": "t", "channel": "C1"}},
        )
        tool = _tool(username="Sara", icon_url="https://example.com/sara.png")
        await tool.run(
            _msg(json_freeze({"operation": "send", "channel": "C1", "text": "hi"}))
        )
        _, _, body = mock.calls[0]
        assert body["username"] == "Sara"
        assert body["icon_url"] == "https://example.com/sara.png"

    @pytest.mark.anyio
    async def test_no_identity_when_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            {"chat.postMessage": {"ts": "t", "channel": "C1"}},
        )
        await _tool().run(
            _msg(json_freeze({"operation": "send", "channel": "C1", "text": "hi"}))
        )
        _, _, body = mock.calls[0]
        assert "username" not in body
        assert "icon_url" not in body


class TestListChannels:
    @pytest.mark.anyio
    async def test_renders(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "conversations.list": {
                    "channels": [
                        {"id": "C1", "name": "general", "num_members": 5},
                    ]
                }
            },
        )
        resp = await _tool().run(_msg(json_freeze({"operation": "list_channels"})))
        assert isinstance(resp, TextMessage)
        assert "C1" in resp.content
        assert "#general" in resp.content


class TestMessages:
    @pytest.mark.anyio
    async def test_list_messages_requires_channel(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(monkeypatch, {})
        resp = await _tool().run(_msg(json_freeze({"operation": "list_messages"})))
        assert resp.descriptor == "text/x-error"
        assert "required" in str(resp.content)

    @pytest.mark.anyio
    async def test_list_messages(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "conversations.history": {
                    "messages": [
                        {"ts": "1", "user": "U1", "text": "hello"},
                        {"ts": "2", "user": "U2", "text": "world"},
                    ]
                }
            },
        )
        resp = await _tool().run(
            _msg(json_freeze({"operation": "list_messages", "channel": "C1"}))
        )
        assert isinstance(resp, TextMessage)
        assert "hello" in resp.content
        assert "world" in resp.content

    @pytest.mark.anyio
    async def test_read_thread(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "conversations.replies": {
                    "messages": [
                        {"ts": "1", "user": "U1", "text": "parent"},
                        {"ts": "1.1", "user": "U2", "text": "reply"},
                    ]
                }
            },
        )
        resp = await _tool().run(
            _msg(
                json_freeze(
                    {"operation": "read_thread", "channel": "C1", "thread_ts": "1"}
                )
            )
        )
        assert isinstance(resp, TextMessage)
        assert "parent" in resp.content
        assert "reply" in resp.content


class TestUsers:
    @pytest.mark.anyio
    async def test_list(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(
            monkeypatch,
            {
                "users.list": {
                    "members": [
                        {"id": "U1", "name": "alice", "real_name": "Alice A"},
                        {"id": "U2", "name": "bob", "deleted": True},
                    ]
                }
            },
        )
        resp = await _tool().run(_msg(json_freeze({"operation": "list_users"})))
        assert isinstance(resp, TextMessage)
        assert "alice" in resp.content
        assert "bob" not in resp.content


class TestCreateChannel:
    @pytest.mark.anyio
    async def test_requires_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_client(monkeypatch, {})
        resp = await _tool().run(_msg(json_freeze({"operation": "create_channel"})))
        assert resp.descriptor == "text/x-error"
        assert "required" in str(resp.content)

    @pytest.mark.anyio
    async def test_creates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock = _patch_client(
            monkeypatch,
            {"conversations.create": {"channel": {"id": "C99", "name": "test-log"}}},
        )
        resp = await _tool().run(
            _msg(
                json_freeze({"operation": "create_channel", "channel_name": "test-log"})
            )
        )
        assert isinstance(resp, TextMessage)
        assert "C99" in resp.content
        _, _, body = mock.calls[0]
        assert body["name"] == "test-log"


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
