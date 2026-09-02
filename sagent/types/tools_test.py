"""Tests for ``types.tools``: the result off-load policy."""

from __future__ import annotations

import pytest

from sagent.types.model import AgentSettings
from sagent.types.tools import ToolResultPolicy


def _settings(*, request: int = 200_000) -> AgentSettings:
    return AgentSettings(max_request_tokens=request, max_response_tokens=1_024)


def test_from_settings_scales_with_the_window() -> None:
    policy = ToolResultPolicy.from_settings(_settings(request=1_000_000))
    assert policy.persist_tokens == 250_000
    assert policy.message_budget_tokens == 500_000


def test_a_single_result_may_never_exceed_the_window() -> None:
    """No floor: a fixed 20k one let ONE result be 2.4x a gpt-4 context."""
    for window in (8_192, 128_000, 1_000_000):
        policy = ToolResultPolicy.from_settings(_settings(request=window))
        assert policy.persist_tokens < window
        assert policy.message_budget_tokens < window


def test_the_per_result_threshold_is_below_the_aggregate() -> None:
    policy = ToolResultPolicy.from_settings(_settings())
    assert policy.persist_tokens < policy.message_budget_tokens


def test_defaults_disable_off_loading() -> None:
    policy = ToolResultPolicy()
    assert policy.persist_tokens == 0
    assert policy.message_budget_tokens == 0


@pytest.mark.parametrize(
    ("field", "value"), [("persist_tokens", -1), ("message_budget_tokens", -1)]
)
def test_negative_thresholds_are_rejected(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        _ = ToolResultPolicy(**{field: value})


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
