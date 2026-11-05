"""Tests for local Ollama Chat and caller-defined model identifiers."""

from dataclasses import replace

import pytest

from sagent.model.types import Request
from sagent.providers.ollama.api import Ollama
from sagent.providers.table import ModelTable


@pytest.mark.asyncio
async def test_ollama_is_local_and_does_not_require_authentication() -> None:
    provider = Ollama()
    assert await provider.credentials() == {}
    assert await provider.endpoint() == "http://localhost:11434/v1"
    model = provider["default"]
    assert (
        provider.encode(Request(max_response_tokens=8), model=model)["max_tokens"] == 8
    )


def test_ollama_accepts_installed_model_identifiers() -> None:
    row = replace(
        Ollama.models()["default"],
        model_id="custom",
        wire_model_id="my-model:latest",
    )
    provider = Ollama(models=ModelTable(rows=(row,), roles={"default": "custom"}))
    model = provider["default"]
    assert provider.encode(Request(), model=model)["model"] == "my-model:latest"


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
