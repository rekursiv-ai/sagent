"""Tests for sagent.tools.result_storage."""

from __future__ import annotations

from pathlib import Path

from sagent.custom_types import Message, MultipartMessage, TextMessage
from sagent.tools.result_storage import (
    DEFAULT_PERSIST_THRESHOLD,
    MESSAGE_BUDGET_CHARS,
    PERSIST_EXEMPT_TOOLS,
    ReplacementState,
    _build_preview,
    _result_path,
    enforce_message_budget,
    inject_empty_marker,
    persist_result,
)


class TestResultPath:
    def test_deterministic(self) -> None:
        assert _result_path("id1") == _result_path("id1")

    def test_different_ids(self) -> None:
        assert _result_path("id1") != _result_path("id2")


class TestBuildPreview:
    def test_short_content(self) -> None:
        preview = _build_preview("short", Path("x.txt"))
        assert "persisted-output" in preview
        assert "short" in preview

    def test_long_content_truncated(self) -> None:
        content = "line\n" * 1000
        preview = _build_preview(content, Path("x.txt"))
        assert "..." in preview
        assert len(preview) < len(content)


class TestPersistResult:
    def test_under_threshold_returns_none(self) -> None:
        state = ReplacementState()
        assert persist_result("id1", "Bash", "small", state) is None

    def test_exempt_tool_returns_none(self) -> None:
        state = ReplacementState()
        big = "x" * (DEFAULT_PERSIST_THRESHOLD + 1)
        assert persist_result("id_read", "application/x-tool-read", big, state) is None
        assert "application/x-tool-read" in PERSIST_EXEMPT_TOOLS

    def test_over_threshold_returns_preview(self, tmp_path: Path) -> None:
        state = ReplacementState(storage_dir=tmp_path)
        content = "x" * (DEFAULT_PERSIST_THRESHOLD + 1)
        preview = persist_result("id_big", "Bash", content, state)
        assert preview is not None
        assert "persisted-output" in preview

    def test_dedup_no_overwrite(self, tmp_path: Path) -> None:
        state = ReplacementState(storage_dir=tmp_path)
        content = "x" * (DEFAULT_PERSIST_THRESHOLD + 1)
        p1 = persist_result("id_dup", "Bash", content, state)
        p2 = persist_result("id_dup", "Bash", content, state)
        assert p1 is not None
        assert p2 is not None


def _make_result(queue_id: str, text: str) -> Message:
    return MultipartMessage(
        (
            TextMessage(queue_id, "text/x-queue-id"),
            TextMessage(text, "text/plain"),
        ),
        "multipart/x-tool-result",
    )


def _text(result: Message) -> str:
    if isinstance(result, MultipartMessage):
        for p in result.content:
            if p.descriptor in (
                "text/plain",
                "text/x-error",
            ):
                return str(p.content)
    return result.content if isinstance(result, TextMessage) else ""


class TestEnforceMessageBudget:
    def test_under_budget_unchanged(self) -> None:
        results = [_make_result("t1", "short")]
        state = ReplacementState()
        out = enforce_message_budget(results, {"t1": "Bash"}, state)
        assert _text(out[0]) == "short"

    def test_reapply_cached(self) -> None:
        state = ReplacementState()
        state.replacements["t1"] = "cached_preview"
        state.seen_ids.add("t1")
        results = [_make_result("t1", "new_content")]
        out = enforce_message_budget(results, {"t1": "Bash"}, state)
        assert _text(out[0]) == "cached_preview"

    def test_fresh_persisted_when_over_budget(self, tmp_path: Path) -> None:
        big = "x" * (MESSAGE_BUDGET_CHARS + 1)
        results = [_make_result("t1", big)]
        state = ReplacementState(message_budget=100, storage_dir=tmp_path)
        out = enforce_message_budget(
            results,
            {"t1": "Bash"},
            state,
        )
        assert "persisted-output" in _text(out[0])
        assert "t1" in state.replacements

    def test_frozen_not_replaced(self) -> None:
        state = ReplacementState()
        state.seen_ids.add("t1")  # seen but not replaced
        results = [_make_result("t1", "frozen")]
        out = enforce_message_budget(results, {"t1": "Bash"}, state)
        assert _text(out[0]) == "frozen"

    def test_exempt_tool_treated_as_frozen(self) -> None:
        big = "x" * (MESSAGE_BUDGET_CHARS + 1)
        results = [_make_result("t1", big)]
        state = ReplacementState(message_budget=100)
        out = enforce_message_budget(
            results,
            {"t1": "application/x-tool-read"},
            state,
        )
        # Read is exempt - content unchanged even over budget.
        assert _text(out[0]) == big


class TestInjectEmptyMarker:
    def test_non_empty_passthrough(self) -> None:
        assert inject_empty_marker("Bash", "output") == "output"

    def test_empty_replaced(self) -> None:
        result = inject_empty_marker("Bash", "")
        assert "no output" in result

    def test_whitespace_replaced(self) -> None:
        result = inject_empty_marker("Bash", "   \n  ")
        assert "no output" in result


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
