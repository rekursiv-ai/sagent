"""Each section of ``openai.yaml`` offers only the settings it sends."""

from __future__ import annotations

import pytest

from sagent.providers import openai


def test_a_section_is_read_once() -> None:
    assert openai.models("openai-sub") is openai.models("openai-sub")


def test_a_setting_the_section_sends_offers_the_catalogs_choices() -> None:
    rows = openai.models("openai-sub").values()
    assert any(len(row.values("thinking_effort")) > 1 for row in rows)


def test_an_unknown_section_is_refused() -> None:
    with pytest.raises(ValueError, match=r"no provider 'nowhere'"):
        openai.models("nowhere")


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
