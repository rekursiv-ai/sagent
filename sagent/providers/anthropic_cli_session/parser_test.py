"""Tests for the CLI session JSONL parser's coalescing behaviour."""

from __future__ import annotations

from pathlib import Path

import json

from sagent.providers.anthropic_cli_session.parser import parse_jsonl_to_messages
from sagent.types.runtime import AssistantMessage, ToolResult, UserMessage


def _write(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def _asst(
    uuid: str, parent: str | None, blocks: list[dict], rid: str = "req_x"
) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "requestId": rid,
        "sessionId": "s",
        "version": "test",
        "timestamp": "2026-06-09T00:00:00.000Z",
        "message": {"role": "assistant", "content": blocks},
    }


def test_consecutive_assistant_entries_coalesce_into_one(tmp_path: Path) -> None:
    """Claude splits one assistant turn into separate thinking/text/tool_use
    JSONL entries; the parser must merge that run into ONE AssistantMessage.

    Without this, consecutive assistant-role messages land on the tape and
    a later ContextSplice (compaction) raises ``InvalidPayloadError:
    payload violates role alternation``. Surfaced live 2026-06-09 when
    rehydrating SWE's 283-entry session on restart.
    """
    p = tmp_path / "s.jsonl"
    _write(
        p,
        [
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": None,
                "sessionId": "s",
                "version": "test",
                "timestamp": "2026-06-09T00:00:00.000Z",
                "message": {"role": "user", "content": "do it"},
            },
            # One assistant turn, split across 3 entries (claude's shape):
            _asst(
                "a1",
                "u1",
                [{"type": "thinking", "thinking": "plan", "signature": "sig"}],
            ),
            _asst("a2", "a1", [{"type": "text", "text": "Running it."}]),
            _asst(
                "a3",
                "a2",
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"cmd": "ls"},
                    }
                ],
            ),
            {
                "type": "user",
                "uuid": "u2",
                "parentUuid": "a3",
                "sessionId": "s",
                "version": "test",
                "timestamp": "2026-06-09T00:00:00.000Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "ok"}],
                        }
                    ],
                },
            },
        ],
    )
    msgs = parse_jsonl_to_messages(p)
    # user, ONE merged assistant, tool_result  → 3 messages (not 5)
    assert len(msgs) == 3, [type(m).__name__ for m in msgs]
    assert isinstance(msgs[0], UserMessage)
    assert isinstance(msgs[1], AssistantMessage)
    assert isinstance(msgs[2], ToolResult)
    # The merged assistant carries all three blocks
    a = msgs[1]
    assert a.text == "Running it."
    assert len(a.thinking_blocks) == 1
    assert len(a.tool_calls) == 1
    assert a.tool_calls[0].id == "toolu_1"
    # No consecutive assistants
    consec = sum(
        1
        for i in range(1, len(msgs))
        if isinstance(msgs[i], AssistantMessage)
        and isinstance(msgs[i - 1], AssistantMessage)
    )
    assert consec == 0


def test_text_only_assistants_still_distinct_across_a_user_turn(tmp_path: Path) -> None:
    """Two assistant turns SEPARATED by a user/tool entry stay distinct.

    Coalescing only fuses ADJACENT assistant entries; a user message
    between two assistant turns is a real turn boundary.
    """
    p = tmp_path / "s.jsonl"
    _write(
        p,
        [
            _asst("a1", None, [{"type": "text", "text": "first"}]),
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": "a1",
                "sessionId": "s",
                "version": "test",
                "timestamp": "2026-06-09T00:00:00.000Z",
                "message": {"role": "user", "content": "again"},
            },
            _asst("a2", "u1", [{"type": "text", "text": "second"}]),
        ],
    )
    msgs = parse_jsonl_to_messages(p)
    assert len(msgs) == 3
    assert isinstance(msgs[0], AssistantMessage)
    assert msgs[0].text == "first"
    assert isinstance(msgs[1], UserMessage)
    assert isinstance(msgs[2], AssistantMessage)
    assert msgs[2].text == "second"
