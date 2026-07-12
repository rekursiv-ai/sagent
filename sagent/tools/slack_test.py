"""Tests for ``tools.slack``: Slack Web API integration."""

from __future__ import annotations

from unittest.mock import patch

import asyncio
import json

from sagent.lib.web.errors import FetchError
from sagent.lib.web.fetch import FetchSession
from sagent.tools.slack import Slack
from sagent.types.runtime import ToolResult


_TOKEN = "test-token-placeholder"  # noqa: S105 -- fake test token


def _ok(payload: dict[str, object]) -> bytes:
    body: dict[str, object] = {"ok": True}
    body.update(payload)
    return json.dumps(body).encode()


def _not_ok(error: str) -> bytes:
    return json.dumps({"ok": False, "error": error}).encode()


def test_slack_metadata() -> None:
    s = Slack(token=_TOKEN)
    assert s.name == "Slack"
    assert s.tool_id == "application/x-tool-slack"


def test_summary_with_channel() -> None:
    s = Slack(token=_TOKEN)
    assert s.summary({"operation": "send", "channel": "C1"}) == "Slack send:C1"


def test_summary_no_channel() -> None:
    s = Slack(token=_TOKEN)
    assert s.summary({"operation": "list_channels"}) == "Slack list_channels"


def test_summary_result_none() -> None:
    assert (
        Slack(token=_TOKEN).summary_result(ToolResult(call_id="", content="ok")) is None
    )


def test_prompt_empty() -> None:
    assert Slack(token=_TOKEN).prompt() == ""


def test_send_requires_channel_and_text() -> None:
    s = Slack(token=_TOKEN)
    result = asyncio.run(s.run({"operation": "send", "channel": "C"}))
    assert result.is_error
    assert "'channel' and 'text' required" in result.content


def test_send_success() -> None:
    payload = _ok({"ts": "1.0", "channel": "C1"})
    with patch(
        "sagent.tools.slack.fetch",
        return_value=(payload, FetchSession()),
    ) as mock_fetch:
        result = asyncio.run(
            Slack(token=_TOKEN, username="bot", icon_url="https://i").run(
                {
                    "operation": "send",
                    "channel": "C1",
                    "text": "hi",
                    "thread_ts": "thr1",
                }
            ),
        )
    assert not result.is_error
    assert "Sent." in result.content
    # Verify the POST mode + payload include username/icon_url/thread.
    _, kwargs = mock_fetch.call_args
    payload_json = kwargs["request"].json
    assert payload_json["channel"] == "C1"
    assert payload_json["text"] == "hi"
    assert payload_json["thread_ts"] == "thr1"
    assert payload_json["username"] == "bot"
    assert payload_json["icon_url"] == "https://i"


def test_send_api_error() -> None:
    with patch(
        "sagent.tools.slack.fetch",
        return_value=(_not_ok("channel_not_found"), FetchSession()),
    ):
        result = asyncio.run(
            Slack(token=_TOKEN).run(
                {"operation": "send", "channel": "X", "text": "hi"}
            ),
        )
    assert result.is_error
    assert "channel_not_found" in result.content


def test_send_http_error() -> None:
    err = FetchError(
        url="https://slack.com/api/chat.postMessage",
        status=500,
        headers={},
        body=b"boom",
    )
    with patch("sagent.tools.slack.fetch", side_effect=err):
        result = asyncio.run(
            Slack(token=_TOKEN).run(
                {"operation": "send", "channel": "C", "text": "hi"}
            ),
        )
    assert result.is_error
    assert "Slack HTTP 500" in result.content


def test_list_channels_empty() -> None:
    payload = _ok({"channels": []})
    with patch(
        "sagent.tools.slack.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(Slack(token=_TOKEN).run({"operation": "list_channels"}))
    assert result.content == "(no channels)"


def test_list_channels_renders() -> None:
    payload = _ok(
        {
            "channels": [
                {"id": "C1", "name": "general", "num_members": 10},
                {"id": "C2", "name": "random"},
            ]
        }
    )
    with patch(
        "sagent.tools.slack.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(Slack(token=_TOKEN).run({"operation": "list_channels"}))
    assert "C1  #general" in result.content
    assert "members=10" in result.content
    assert "C2  #random" in result.content
    assert "members=?" in result.content


def test_list_messages_requires_channel() -> None:
    result = asyncio.run(Slack(token=_TOKEN).run({"operation": "list_messages"}))
    assert result.is_error
    assert "'channel' required" in result.content


def test_list_messages_empty() -> None:
    payload = _ok({"messages": []})
    with patch(
        "sagent.tools.slack.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(
            Slack(token=_TOKEN).run({"operation": "list_messages", "channel": "C1"}),
        )
    assert result.content == "(no messages)"


def test_list_messages_renders_with_reactions() -> None:
    payload = _ok(
        {
            "messages": [
                {
                    "ts": "1.0",
                    "user": "U1",
                    "text": "hello",
                    "reactions": [{"name": "tada", "count": 3}],
                },
                {"ts": "2.0", "bot_id": "B1", "text": "bot"},
            ]
        }
    )
    with patch(
        "sagent.tools.slack.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(
            Slack(token=_TOKEN).run({"operation": "list_messages", "channel": "C1"}),
        )
    assert "[1.0] <U1> hello" in result.content
    assert ":tada:x3" in result.content
    assert "[2.0] <B1> bot" in result.content


def test_read_thread_requires_channel_and_ts() -> None:
    result = asyncio.run(Slack(token=_TOKEN).run({"operation": "read_thread"}))
    assert result.is_error


def test_read_thread_renders() -> None:
    payload = _ok({"messages": [{"ts": "1.0", "user": "U1", "text": "parent"}]})
    with patch(
        "sagent.tools.slack.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(
            Slack(token=_TOKEN).run(
                {
                    "operation": "read_thread",
                    "channel": "C1",
                    "thread_ts": "1.0",
                }
            ),
        )
    assert "[1.0] <U1> parent" in result.content


def test_list_users_filters_deleted() -> None:
    payload = _ok(
        {
            "members": [
                {"id": "U1", "name": "alice", "real_name": "Alice"},
                {"id": "U2", "name": "bob", "deleted": True},
            ]
        }
    )
    with patch(
        "sagent.tools.slack.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(Slack(token=_TOKEN).run({"operation": "list_users"}))
    assert "U1  @alice" in result.content
    assert "U2" not in result.content


def test_list_users_empty() -> None:
    payload = _ok({"members": []})
    with patch(
        "sagent.tools.slack.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(Slack(token=_TOKEN).run({"operation": "list_users"}))
    assert result.content == "(no users)"


def test_create_channel_requires_name() -> None:
    result = asyncio.run(Slack(token=_TOKEN).run({"operation": "create_channel"}))
    assert result.is_error
    assert "'channel_name' required" in result.content


def test_create_channel_success() -> None:
    payload = _ok({"channel": {"id": "C-new"}})
    with patch(
        "sagent.tools.slack.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(
            Slack(token=_TOKEN).run(
                {"operation": "create_channel", "channel_name": "newroom"}
            ),
        )
    assert result.content == "id=C-new"


def test_unknown_operation() -> None:
    result = asyncio.run(Slack(token=_TOKEN).run({"operation": "blah"}))
    assert result.is_error
    assert "Unknown operation" in result.content


def test_send_convenience_method() -> None:
    payload = _ok({"ts": "9.9", "channel": "C9"})
    with patch(
        "sagent.tools.slack.fetch",
        return_value=(payload, FetchSession()),
    ):
        result = asyncio.run(Slack(token=_TOKEN).send("C9", "hi"))
    assert isinstance(result, str)
    assert "Sent." in result


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
