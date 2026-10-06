"""Tests for the shared CLI-subprocess respawn cadence."""

from __future__ import annotations

from pathlib import Path

import tempfile

import pytest

from sagent.providers.lib.cli_respawn import (
    CONTEXT_FRACTION_RESPAWN_THRESHOLD,
    TURN_RESPAWN_THRESHOLD,
    CLISubprocessModel,
    hash_system,
    merge_user_input,
    populated_tmpdir,
    respawn_for_cadence,
)
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)


def test_respawn_when_turn_cap_reached() -> None:
    assert respawn_for_cadence(
        turn_count=TURN_RESPAWN_THRESHOLD,
        last_input_tokens=0,
        max_request_tokens=1_000_000,
    )


def test_no_respawn_below_turn_cap_and_context_fraction() -> None:
    assert not respawn_for_cadence(
        turn_count=TURN_RESPAWN_THRESHOLD - 1,
        last_input_tokens=1,
        max_request_tokens=1_000_000,
    )


def test_respawn_when_context_fraction_crossed() -> None:
    max_tokens = 1_000_000
    over = int(max_tokens * CONTEXT_FRACTION_RESPAWN_THRESHOLD) + 1
    assert respawn_for_cadence(
        turn_count=0,
        last_input_tokens=over,
        max_request_tokens=max_tokens,
    )


def test_no_respawn_exactly_at_context_fraction() -> None:
    # Strict ``>``: exactly at the fraction does not trip.
    max_tokens = 1_000_000
    at = int(max_tokens * CONTEXT_FRACTION_RESPAWN_THRESHOLD)
    assert not respawn_for_cadence(
        turn_count=0,
        last_input_tokens=at,
        max_request_tokens=max_tokens,
    )


def test_merge_user_input_keeps_only_user_side_entries() -> None:
    merged = merge_user_input(
        [
            UserMessage(text="a"),
            AssistantMessage(
                text="",
                tool_calls=(ToolCall(id="c", name="T", args={}),),
            ),
            ToolResult(call_id="c", content="out"),
            AgentSendMessage(source="tl", text="b"),
        ],
    )
    assert isinstance(merged, UserMessage)
    assert merged.text == "a\n\n[from tl]: b"


def test_merge_user_input_none_without_user_side_entries() -> None:
    assert merge_user_input([AssistantMessage(text="x")]) is None


def test_populated_tmpdir_removed_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp(**kwargs: str) -> str:
        path = real_mkdtemp(dir=tmp_path, **kwargs)
        made.append(Path(path))
        return path

    def _fail(path: Path) -> None:
        (path / "creds").write_text("secret")
        raise ValueError("bad credentials")

    monkeypatch.setattr(tempfile, "mkdtemp", _tracking_mkdtemp)
    with pytest.raises(ValueError, match="bad credentials"):
        populated_tmpdir("x-", _fail)
    assert len(made) == 1
    assert not made[0].exists()


def test_hash_system_stable_and_distinguishes() -> None:
    assert hash_system("be brief") == hash_system("be brief")
    assert hash_system("be brief") != hash_system("be verbose")
    assert hash_system(None) == hash_system("")


def test_respawn_due_triggers() -> None:
    state = CLISubprocessModel()
    user = UserMessage(text="hi")
    state._system_hash = hash_system(None)
    assert not state._respawn_due([user], system=None, max_request_tokens=100)
    assert state._respawn_due([], system=None, max_request_tokens=100)
    assert state._respawn_due([user], system="new", max_request_tokens=100)
    state._sent_history_head = UserMessage(text="other")
    assert state._respawn_due([user], system=None, max_request_tokens=100)


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
