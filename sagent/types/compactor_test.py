"""Tests for ``types.compactor``: the re-attach policy."""

from __future__ import annotations

import pytest

from sagent.types.compactor import ReattachPolicy
from sagent.types.model import AgentSettings


def _settings(*, request: int = 200_000) -> AgentSettings:
    return AgentSettings(max_request_tokens=request, max_response_tokens=1_024)


def test_from_settings_scales_with_the_window() -> None:
    policy = ReattachPolicy.from_settings(_settings(request=400_000))
    assert policy.max_tokens == 10_000
    assert policy.budget_tokens == 100_000


def test_from_settings_floors_a_small_window() -> None:
    policy = ReattachPolicy.from_settings(_settings(request=8_192))
    assert policy.max_tokens == 2_000
    assert policy.budget_tokens == 10_000


def test_the_per_file_cap_never_exceeds_the_total() -> None:
    for window in (8_192, 200_000, 1_000_000):
        policy = ReattachPolicy.from_settings(_settings(request=window))
        assert policy.max_tokens <= policy.budget_tokens


def test_both_caps_are_tokens_not_characters() -> None:
    names = set(ReattachPolicy.__dataclass_fields__)
    assert names == {"count", "max_tokens", "budget_tokens"}
    assert not {n for n in names if "char" in n}


def test_defaults_disable_both_caps() -> None:
    policy = ReattachPolicy()
    assert policy.max_tokens == 0
    assert policy.budget_tokens == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [("count", -1), ("max_tokens", -1), ("budget_tokens", -1)],
)
def test_negative_caps_are_rejected(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        _ = ReattachPolicy(**{field: value})


def test_zero_count_is_legal() -> None:
    assert ReattachPolicy(count=0).count == 0


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
