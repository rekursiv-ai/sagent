"""Tests for lib.stop_reason."""

from __future__ import annotations

import pytest

from sagent.providers.lib.stop_reason import (
    BENIGN_STOP_REASONS,
    normalize_stop_reason,
)


class TestNormalizeOpenAI:
    def test_stop_to_model_finished(self) -> None:
        assert (
            normalize_stop_reason("stop", kind="openai", has_tool_use=False)
            == "model_finished"
        )

    def test_length_to_max_tokens(self) -> None:
        assert (
            normalize_stop_reason("length", kind="openai", has_tool_use=False)
            == "max_tokens"
        )

    def test_tool_calls_to_model_tool_use(self) -> None:
        assert (
            normalize_stop_reason("tool_calls", kind="openai", has_tool_use=True)
            == "model_tool_use"
        )

    def test_function_call_legacy(self) -> None:
        assert (
            normalize_stop_reason("function_call", kind="openai", has_tool_use=True)
            == "model_tool_use"
        )

    def test_content_filter_to_model_refusal(self) -> None:
        assert (
            normalize_stop_reason("content_filter", kind="openai", has_tool_use=False)
            == "model_refusal"
        )

    def test_stop_with_tool_use_upgraded(self) -> None:
        # Some streaming variants emit ``stop`` even when tool calls are
        # present; upgrade so the agent's gate sees the right canonical.
        assert (
            normalize_stop_reason("stop", kind="openai", has_tool_use=True)
            == "model_tool_use"
        )

    def test_none_defaults_to_model_finished(self) -> None:
        assert (
            normalize_stop_reason(None, kind="openai", has_tool_use=False)
            == "model_finished"
        )


class TestNormalizeGoogle:
    def test_stop_to_model_finished(self) -> None:
        assert (
            normalize_stop_reason("STOP", kind="google", has_tool_use=False)
            == "model_finished"
        )

    def test_max_tokens(self) -> None:
        assert (
            normalize_stop_reason("MAX_TOKENS", kind="google", has_tool_use=False)
            == "max_tokens"
        )

    @pytest.mark.parametrize(
        "reason",
        ["SAFETY", "RECITATION", "LANGUAGE", "BLOCKLIST", "PROHIBITED_CONTENT"],
    )
    def test_safety_categories_to_model_refusal(self, reason: str) -> None:
        assert (
            normalize_stop_reason(reason, kind="google", has_tool_use=False)
            == "model_refusal"
        )

    def test_stop_with_tool_use_upgraded(self) -> None:
        # Gemini's STOP doesn't differentiate; upgrade when tool calls present.
        assert (
            normalize_stop_reason("STOP", kind="google", has_tool_use=True)
            == "model_tool_use"
        )

    def test_none_defaults_to_model_finished(self) -> None:
        assert (
            normalize_stop_reason(None, kind="google", has_tool_use=False)
            == "model_finished"
        )


class TestNormalizeAnthropic:
    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            ("end_turn", "model_finished"),
            ("tool_use", "model_tool_use"),
            ("max_tokens", "max_tokens"),
            ("pause_turn", "model_continuing"),
            ("refusal", "model_refusal"),
            ("stop_sequence", "stop_sequence"),
        ],
    )
    def test_translation(self, reason: str, expected: str) -> None:
        assert (
            normalize_stop_reason(reason, kind="anthropic", has_tool_use=False)
            == expected
        )

    def test_none_defaults_to_model_finished(self) -> None:
        assert (
            normalize_stop_reason(None, kind="anthropic", has_tool_use=False)
            == "model_finished"
        )


def test_benign_set_is_canonical() -> None:
    """BENIGN_STOP_REASONS must use canonical vocabulary."""
    assert (
        frozenset(
            {"model_finished", "model_tool_use", "model_continuing"},
        )
        == BENIGN_STOP_REASONS
    )
