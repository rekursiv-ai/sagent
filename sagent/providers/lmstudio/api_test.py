"""Tests for configurable local LM Studio Chat."""

from dataclasses import replace

import pytest

from sagent.model.types import Request
from sagent.providers.lmstudio.api import LMStudio
from sagent.providers.table import ModelTable


@pytest.mark.asyncio
async def test_lmstudio_accepts_caller_model_and_optional_authentication() -> None:
    row = replace(
        LMStudio.models()["default"],
        model_id="custom",
        wire_model_id="installed/model",
    )
    provider = LMStudio(models=ModelTable(rows=(row,), roles={"default": "custom"}))
    assert await provider.credentials() == {}
    assert await provider.endpoint() == "http://localhost:1234/v1"
    model = provider["default"]
    assert provider.encode(Request(), model=model)["model"] == "installed/model"
    assert await LMStudio(api_key="secret").credentials() == {
        "authorization": "Bearer secret",
    }


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
