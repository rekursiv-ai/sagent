"""Tests for canonical thinking-state parsing."""

from __future__ import annotations

import pytest

from sagent.thinking import (
    ThinkingCapability,
    ThinkingState,
    request_thinking,
    resolve_thinking_command,
    should_redact_thinking,
    should_show_thinking,
    valid_thinking_states,
)


def test_resolve_full_state() -> None:
    assert resolve_thinking_command("on-show") == "on-show"


@pytest.mark.parametrize(
    ("command", "state"),
    [
        ("adaptive", "adaptive-hide"),
        ("on", "on-hide"),
        ("off", "off-hide"),
        ("redact", "redact-hide"),
        ("show", "adaptive-show"),
        ("hide", "adaptive-hide"),
    ],
)
def test_resolve_startup_partial(command: str, state: str) -> None:
    assert resolve_thinking_command(command) == state


@pytest.mark.parametrize(
    ("command", "current", "state"),
    [
        ("adaptive", "on-show", "adaptive-show"),
        ("on", "adaptive-hide", "on-hide"),
        ("off", "adaptive-show", "off-hide"),
        ("redact", "adaptive-show", "redact-hide"),
        ("hide", "on-show", "on-hide"),
        ("show", "adaptive-hide", "adaptive-show"),
    ],
)
def test_resolve_live_partial(
    command: str, current: ThinkingState, state: ThinkingState
) -> None:
    assert resolve_thinking_command(command, current) == state


@pytest.mark.parametrize("current", ["off-hide", "redact-hide"])
def test_show_invalid_from_unshowable_states(current: ThinkingState) -> None:
    with pytest.raises(ValueError, match="cannot show"):
        resolve_thinking_command("show", current)


@pytest.mark.parametrize("current", ["off-hide", "redact-hide"])
def test_hide_invalid_from_unhidable_states(current: ThinkingState) -> None:
    """``hide`` must reject impossible transitions just like ``show``."""
    with pytest.raises(ValueError, match="cannot hide"):
        resolve_thinking_command("hide", current)


@pytest.mark.parametrize(
    ("state", "request_mode", "show", "redact"),
    [
        ("adaptive-show", "adaptive", True, False),
        ("adaptive-hide", "adaptive", False, False),
        ("on-show", "enabled", True, False),
        ("on-hide", "enabled", False, False),
        ("off-hide", None, False, False),
        ("redact-hide", "adaptive", False, True),
    ],
)
def test_state_projections(
    state: ThinkingState, request_mode: str | None, show: bool, redact: bool
) -> None:
    assert request_thinking(state) == request_mode
    assert should_show_thinking(state) is show
    assert should_redact_thinking(state) is redact


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        # No thinking (OpenAI-chat, SelfHosted, LlamaCpp): off only.
        (ThinkingCapability(supports_thinking=False), ("off-hide",)),
        # Readable, no redaction (Google, OpenAISub, DashScope): all but redact.
        (
            ThinkingCapability(supports_thinking=True),
            ("adaptive-show", "adaptive-hide", "on-show", "on-hide", "off-hide"),
        ),
        # Readable + redaction (plain Anthropic): all six.
        (
            ThinkingCapability(supports_thinking=True, supports_redaction=True),
            (
                "adaptive-show",
                "adaptive-hide",
                "on-show",
                "on-hide",
                "off-hide",
                "redact-hide",
            ),
        ),
        # No readable text (signed-but-empty blocks): every -show drops out.
        (
            ThinkingCapability(
                supports_thinking=True,
                readable_text=False,
                supports_redaction=True,
            ),
            ("adaptive-hide", "on-hide", "off-hide", "redact-hide"),
        ),
        # Adaptive-only (rejects enabled): on-* states drop out.
        (
            ThinkingCapability(
                supports_thinking=True,
                supports_enabled_mode=False,
                supports_redaction=True,
            ),
            ("adaptive-show", "adaptive-hide", "off-hide", "redact-hide"),
        ),
        # opus-4-8: adaptive-only AND no readable text -> only adaptive-hide,
        # off-hide, redact-hide survive (no -show, no on-*).
        (
            ThinkingCapability(
                supports_thinking=True,
                readable_text=False,
                supports_enabled_mode=False,
                supports_redaction=True,
            ),
            ("adaptive-hide", "off-hide", "redact-hide"),
        ),
    ],
)
def test_valid_thinking_states(
    capability: ThinkingCapability, expected: tuple[ThinkingState, ...]
) -> None:
    assert valid_thinking_states(capability) == expected


def test_valid_thinking_states_always_includes_off() -> None:
    """``off-hide`` is valid for every capability."""
    for cap in (
        ThinkingCapability(supports_thinking=False),
        ThinkingCapability(supports_thinking=True),
        ThinkingCapability(supports_thinking=True, readable_text=False),
        ThinkingCapability(supports_thinking=True, supports_redaction=True),
    ):
        assert "off-hide" in valid_thinking_states(cap)


if __name__ == "__main__":
    from sagent.lib.testing import test_main

    test_main(__file__)
