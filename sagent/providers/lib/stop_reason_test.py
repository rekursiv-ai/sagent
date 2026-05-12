"""Tests for ``providers.lib.stop_reason``: provider-to-canonical normalization."""

from __future__ import annotations

import pytest

from sagent.providers.lib.stop_reason import (
    BENIGN_STOP_REASONS,
    normalize_stop_reason,
)


def test_benign_stop_reasons_membership() -> None:
    assert "model_finished" in BENIGN_STOP_REASONS
    assert "model_tool_use" in BENIGN_STOP_REASONS
    assert "model_continuing" in BENIGN_STOP_REASONS
    assert "max_tokens" not in BENIGN_STOP_REASONS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", "model_finished"),
        ("pause_turn", "model_continuing"),
        ("tool_use", "model_tool_use"),
        ("refusal", "model_refusal"),
        ("stop_sequence", "stop_sequence"),
        ("max_tokens", "max_tokens"),
    ],
)
def test_anthropic_known_mappings(raw: str, expected: str) -> None:
    assert normalize_stop_reason(raw, kind="anthropic", has_tool_use=False) == expected


def test_anthropic_none_defaults_to_finished() -> None:
    assert (
        normalize_stop_reason(None, kind="anthropic", has_tool_use=False)
        == "model_finished"
    )


def test_anthropic_unknown_passthrough() -> None:
    assert (
        normalize_stop_reason("custom_x", kind="anthropic", has_tool_use=False)
        == "custom_x"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stop", "model_finished"),
        ("length", "max_tokens"),
        ("tool_calls", "model_tool_use"),
        ("function_call", "model_tool_use"),
        ("content_filter", "model_refusal"),
    ],
)
def test_openai_known_mappings(raw: str, expected: str) -> None:
    assert normalize_stop_reason(raw, kind="openai", has_tool_use=False) == expected


def test_openai_none_defaults_to_finished() -> None:
    assert (
        normalize_stop_reason(None, kind="openai", has_tool_use=False)
        == "model_finished"
    )


def test_openai_stop_upgrades_when_tool_use_present() -> None:
    """``stop`` + tool calls → ``model_tool_use``."""
    assert (
        normalize_stop_reason("stop", kind="openai", has_tool_use=True)
        == "model_tool_use"
    )


def test_openai_length_not_upgraded_with_tool_use() -> None:
    # max_tokens is not model_finished, so the tool-use upgrade does not apply.
    assert (
        normalize_stop_reason("length", kind="openai", has_tool_use=True)
        == "max_tokens"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("STOP", "model_finished"),
        ("MAX_TOKENS", "max_tokens"),
        ("SAFETY", "model_refusal"),
        ("RECITATION", "model_refusal"),
        ("LANGUAGE", "model_refusal"),
        ("BLOCKLIST", "model_refusal"),
        ("PROHIBITED_CONTENT", "model_refusal"),
        ("SPII", "model_refusal"),
        ("IMAGE_SAFETY", "model_refusal"),
        ("MALFORMED_FUNCTION_CALL", "model_refusal"),
        ("OTHER", "model_finished"),
        ("FINISH_REASON_UNSPECIFIED", "model_finished"),
    ],
)
def test_google_known_mappings(raw: str, expected: str) -> None:
    assert normalize_stop_reason(raw, kind="google", has_tool_use=False) == expected


def test_google_stop_upgrades_when_tool_use_present() -> None:
    assert (
        normalize_stop_reason("STOP", kind="google", has_tool_use=True)
        == "model_tool_use"
    )


def test_google_unknown_passthrough() -> None:
    assert (
        normalize_stop_reason("CUSTOM", kind="google", has_tool_use=False) == "CUSTOM"
    )


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
