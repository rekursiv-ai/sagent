"""Tests for ``compaction.files``: re-attach helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sagent.compaction import files
from sagent.compaction.files import (
    CLEARED,
    reattach_files,
)
from sagent.types.compactor import ReattachPolicy
from sagent.types.runtime import (
    AssistantMessage,
    ModelContextEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from pathlib import Path


def _policy(*, count: int, max_tokens: int, budget_tokens: int) -> ReattachPolicy:
    """Re-attach caps for one case."""
    return ReattachPolicy(
        count=count, max_tokens=max_tokens, budget_tokens=budget_tokens
    )


def _estimate(text: str) -> int:
    """Four chars per token, matching the stub models."""
    return max(1, len(text) // 4)


@pytest.mark.asyncio
async def test_reattach_files_no_recent_noop(tmp_path: Path) -> None:
    del tmp_path
    history: list[ModelContextEvent] = []
    await reattach_files(
        history,
        [],
        estimate_tokens=_estimate,
        policy=_policy(count=3, max_tokens=250, budget_tokens=2500),
    )
    assert history == []


@pytest.mark.asyncio
async def test_reattach_files_inserts_user_message(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("contents of a")
    history: list[ModelContextEvent] = []
    await reattach_files(
        history,
        [str(f)],
        estimate_tokens=_estimate,
        policy=_policy(
            count=3,
            max_tokens=250,
            budget_tokens=2500,
        ),
    )
    assert len(history) == 1
    first = history[0]
    assert isinstance(first, UserMessage)
    assert "Recently accessed files" in first.text
    assert "contents of a" in first.text


@pytest.mark.asyncio
async def test_reattach_files_appends_to_first_user(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("file body")
    history: list[ModelContextEvent] = [UserMessage(text="original prompt")]
    await reattach_files(
        history,
        [str(f)],
        estimate_tokens=_estimate,
        policy=_policy(count=3, max_tokens=250, budget_tokens=2500),
    )
    assert len(history) == 1
    first = history[0]
    assert isinstance(first, UserMessage)
    assert first.text.startswith("original prompt")
    assert "file body" in first.text


@pytest.mark.asyncio
async def test_reattach_files_skips_already_read(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("body")
    resolved = str(f.resolve())
    # History shows the Read tool already pulled this file in.
    history: list[ModelContextEvent] = [
        UserMessage(text="hi"),
        AssistantMessage(
            tool_calls=(ToolCall(id="c1", name="Read", args={"file_path": resolved}),),
        ),
        ToolResult(call_id="c1", content="body"),
    ]
    before = list(history)
    await reattach_files(
        history,
        [str(f)],
        estimate_tokens=_estimate,
        policy=_policy(count=3, max_tokens=250, budget_tokens=2500),
    )
    # File is already inline -> no re-attachment.
    assert history == before


@pytest.mark.asyncio
async def test_reattach_files_skips_cleared(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("body")
    resolved = str(f.resolve())
    history: list[ModelContextEvent] = [
        UserMessage(text="hi"),
        AssistantMessage(
            tool_calls=(ToolCall(id="c1", name="Read", args={"file_path": resolved}),),
        ),
        ToolResult(call_id="c1", content=CLEARED),
    ]
    await reattach_files(
        history,
        [str(f)],
        estimate_tokens=_estimate,
        policy=_policy(count=3, max_tokens=250, budget_tokens=2500),
    )
    # Cleared content means the file is NOT inline; re-attach should occur.
    first = history[0]
    assert isinstance(first, UserMessage)
    assert "body" in first.text


@pytest.mark.asyncio
async def test_reattach_files_truncates_long_file(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    big = "x" * 10_000
    f.write_text(big)
    history: list[ModelContextEvent] = []
    await reattach_files(
        history,
        [str(f)],
        estimate_tokens=_estimate,
        policy=_policy(count=3, max_tokens=25, budget_tokens=2500),
    )
    first = history[0]
    assert isinstance(first, UserMessage)
    assert "(truncated for re-attachment)" in first.text


@pytest.mark.asyncio
async def test_reattach_files_budget_caps_total(tmp_path: Path) -> None:
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text("a" * 500)
    f2.write_text("b" * 500)
    history: list[ModelContextEvent] = []
    await reattach_files(
        history,
        [str(f1), str(f2)],
        estimate_tokens=_estimate,
        policy=_policy(
            count=3,
            max_tokens=250,
            budget_tokens=150,
        ),
    )
    first = history[0]
    assert isinstance(first, UserMessage)
    # Iteration is newest-first under budget, so the most-recent file wins.
    assert "b.py" in first.text
    assert "a.py" not in first.text


@pytest.mark.asyncio
async def test_reattach_files_skips_missing_files(tmp_path: Path) -> None:
    missing = tmp_path / "nope.py"
    history: list[ModelContextEvent] = []
    await reattach_files(
        history,
        [str(missing)],
        estimate_tokens=_estimate,
        policy=_policy(count=3, max_tokens=250, budget_tokens=2500),
    )
    assert history == []


@pytest.mark.asyncio
async def test_reattach_files_prefers_newest_under_budget(tmp_path: Path) -> None:
    """Iteration is newest-first so the freshest file always lands."""
    old = tmp_path / "old.py"
    mid = tmp_path / "mid.py"
    new = tmp_path / "new.py"
    old.write_text("o" * 5_000)
    mid.write_text("m" * 5_000)
    new.write_text("n" * 1_000)
    history: list[ModelContextEvent] = []
    await reattach_files(
        history,
        [str(old), str(mid), str(new)],
        estimate_tokens=_estimate,
        policy=_policy(
            count=3,
            max_tokens=2500,
            budget_tokens=1500,
        ),
    )
    first = history[0]
    assert isinstance(first, UserMessage)
    assert "new.py" in first.text


@pytest.mark.asyncio
async def test_reattach_files_skips_already_written(tmp_path: Path) -> None:
    """A Write tool result keeps content inline; don't re-attach atop it."""
    f = tmp_path / "a.py"
    f.write_text("body-on-disk")
    resolved = str(f.resolve())
    history: list[ModelContextEvent] = [
        UserMessage(text="hi"),
        AssistantMessage(
            tool_calls=(
                ToolCall(
                    id="c1",
                    name="Write",
                    args={"file_path": resolved, "content": "body-on-disk"},
                ),
            ),
        ),
        ToolResult(call_id="c1", content="ok"),
    ]
    before = list(history)
    await reattach_files(
        history,
        [str(f)],
        estimate_tokens=_estimate,
        policy=_policy(count=3, max_tokens=250, budget_tokens=2500),
    )
    assert history == before


@pytest.mark.asyncio
async def test_reattach_files_does_not_skip_edited_only_file(tmp_path: Path) -> None:
    """``Edit`` only embeds fragments; re-attach must refresh the full body."""
    f = tmp_path / "a.py"
    f.write_text("body-on-disk")
    resolved = str(f.resolve())
    history: list[ModelContextEvent] = [
        UserMessage(text="hi"),
        AssistantMessage(
            tool_calls=(
                ToolCall(
                    id="c1",
                    name="Edit",
                    args={
                        "file_path": resolved,
                        "old_string": "a",
                        "new_string": "b",
                    },
                ),
            ),
        ),
        ToolResult(call_id="c1", content="ok"),
    ]
    await reattach_files(
        history,
        [str(f)],
        estimate_tokens=_estimate,
        policy=_policy(count=3, max_tokens=250, budget_tokens=2500),
    )
    first = history[0]
    assert isinstance(first, UserMessage)
    assert "body-on-disk" in first.text


@pytest.mark.asyncio
async def test_reattach_files_escapes_quote_in_path(tmp_path: Path) -> None:
    """A path containing ``"`` must not break the ``<file>`` attribute."""
    f = tmp_path / 'a"b.py'
    f.write_text("body")
    history: list[ModelContextEvent] = [UserMessage(text="prompt")]
    await reattach_files(
        history,
        [str(f)],
        estimate_tokens=_estimate,
        policy=_policy(count=3, max_tokens=250, budget_tokens=2500),
    )
    first = history[0]
    assert isinstance(first, UserMessage)
    # Raw quote in path would terminate the attribute early -- must be escaped.
    assert f'path="{f}"' not in first.text
    assert "body" in first.text


@pytest.mark.asyncio
async def test_reattach_files_escapes_close_tag_in_body(tmp_path: Path) -> None:
    """A body containing ``</file>`` must not close the wrapper prematurely."""
    f = tmp_path / "a.py"
    f.write_text("before </file> after")
    history: list[ModelContextEvent] = [UserMessage(text="prompt")]
    await reattach_files(
        history,
        [str(f)],
        estimate_tokens=_estimate,
        policy=_policy(count=3, max_tokens=250, budget_tokens=2500),
    )
    first = history[0]
    assert isinstance(first, UserMessage)
    # Exactly one closing tag -- the framing's own. Body's literal is escaped.
    assert first.text.count("</file>") == 1


def test_files_reuses_history_append_helper_without_duplicate() -> None:
    # ``append_to_first_user`` is the canonical history mutator in
    # ``compaction.history``; ``files`` must reuse it, not keep a private
    # logic-identical ``_append_to_first_user`` copy (DRY; one rule everywhere).
    assert not hasattr(files, "_append_to_first_user")


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
