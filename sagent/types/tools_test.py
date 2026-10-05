"""Tests for ``types.tools``: the result off-load policy."""

from __future__ import annotations

import pytest

from sagent.types.settings import AgentSettings
from sagent.types.tools import ToolResultPolicy


def _settings(*, request: int = 200_000) -> AgentSettings:
    return AgentSettings(max_request_tokens=request, max_response_tokens=1_024)


def test_from_settings_scales_only_the_aggregate_with_the_window() -> None:
    """A window-derived per-result cap let one Grep carry 113k tokens on 1M.

    The per-result bound is the fixed character cap plus the room left in
    context, both applied by the agent; the policy derives no per-result cap.
    """
    policy = ToolResultPolicy.from_settings(_settings(request=1_000_000))
    assert policy.persist_tokens == 0
    assert policy.message_budget_tokens == 500_000


def test_the_aggregate_may_never_exceed_the_window() -> None:
    for window in (8_192, 128_000, 1_000_000):
        policy = ToolResultPolicy.from_settings(_settings(request=window))
        assert policy.message_budget_tokens < window


def test_defaults_disable_off_loading() -> None:
    policy = ToolResultPolicy()
    assert policy.persist_tokens == 0
    assert policy.message_budget_tokens == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("persist_tokens", -1), ("message_budget_tokens", -1)],
)
def test_negative_thresholds_are_rejected(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        _ = ToolResultPolicy(**{field: value})


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
