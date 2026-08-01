"""Tests for canonical thinking-state parsing."""

from __future__ import annotations

import pytest

from sagent.thinking import (
    ThinkingState,
    request_thinking,
    resolve_thinking_command,
    should_redact_thinking,
    should_show_thinking,
    thinking_mode_supported,
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
    ("mode", "valid_states", "expected"),
    [
        (None, ("off-hide",), True),  # off always supported
        ("adaptive", ("adaptive-hide", "off-hide"), True),
        ("adaptive", ("on-hide", "off-hide"), False),  # enabled-only model
        ("enabled", ("on-hide", "off-hide"), True),
        ("enabled", ("adaptive-hide", "off-hide"), False),  # adaptive-only model
        ("adaptive", ("off-hide",), False),  # thinking effectively off
    ],
)
def test_thinking_mode_supported(
    mode: str | None, valid_states: tuple[str, ...], expected: bool
) -> None:
    assert thinking_mode_supported(mode, valid_states) is expected


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
