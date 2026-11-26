"""Exercise NVIDIA NIM's endpoint and credentials without inventing prices."""

from __future__ import annotations

import pytest

from sagent.model.exceptions import AuthError
from sagent.providers.nvidia.api import NVIDIA


@pytest.mark.asyncio
async def test_endpoint_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = NVIDIA(name="nvidia")
    assert await provider.endpoint() == "https://integrate.api.nvidia.com/v1"
    monkeypatch.setenv("NVIDIA_API_KEY", "secret")
    assert await provider.credentials() == {"authorization": "Bearer secret"}
    monkeypatch.delenv("NVIDIA_API_KEY")
    with pytest.raises(AuthError, match="NVIDIA_API_KEY"):
        await provider.credentials()
    explicit = NVIDIA(api_key="explicit", base_url="http://box/v1")
    assert await explicit.endpoint() == "http://box/v1"
    assert await explicit.credentials() == {"authorization": "Bearer explicit"}


def test_no_unpriced_model_is_advertised_as_free() -> None:
    provider = NVIDIA()
    assert provider.name == "nvidia"
    assert list(provider) == []
    assert len(provider.models()) == 0


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
