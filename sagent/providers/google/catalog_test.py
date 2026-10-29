"""Each section of ``google.yaml`` offers only the settings it sends."""

from __future__ import annotations

import pytest

from sagent.providers import google


def test_a_section_is_read_once() -> None:
    assert google.models("google-cli") is google.models("google-cli")


def test_a_setting_the_section_does_not_send_offers_only_its_default() -> None:
    for row in google.models("google-cli").values():
        assert row.values("thinking_budget") == (row.default.value("thinking_budget"),)


def test_an_unknown_section_is_refused() -> None:
    with pytest.raises(ValueError, match=r"no provider 'nowhere'"):
        google.models("nowhere")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
