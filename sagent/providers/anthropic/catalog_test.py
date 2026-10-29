"""Each section of ``anthropic.yaml`` offers only the settings it sends."""

from __future__ import annotations

import pytest

from sagent.providers import anthropic


def test_a_section_is_read_once() -> None:
    assert anthropic.models("anthropic-cli") is anthropic.models("anthropic-cli")


def test_a_setting_the_section_does_not_send_offers_only_its_default() -> None:
    for row in anthropic.models("anthropic-cli").values():
        assert row.values("service_tier") == (row.default.value("service_tier"),)


def test_an_unknown_section_is_refused() -> None:
    with pytest.raises(ValueError, match=r"no provider 'nowhere'"):
        anthropic.models("nowhere")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
