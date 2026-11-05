"""Tests for account-scoped Workers AI Chat."""

import pytest

from sagent.model.exceptions import AuthError
from sagent.model.types import Request
from sagent.providers.cloudflare.api import Cloudflare


@pytest.mark.asyncio
async def test_cloudflare_scopes_the_endpoint_to_the_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account123")
    provider = Cloudflare(api_key="token")
    assert (
        await provider.endpoint()
        == "https://api.cloudflare.com/client/v4/accounts/account123/ai/v1"
    )
    assert await provider.credentials() == {"authorization": "Bearer token"}
    model = provider["default"]
    assert (
        provider.encode(Request(max_response_tokens=8), model=model)["max_tokens"] == 8
    )


@pytest.mark.asyncio
async def test_cloudflare_override_and_missing_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        await Cloudflare(base_url="https://proxy.example/v1").endpoint()
        == "https://proxy.example/v1"
    )
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_BASE_URL", raising=False)
    with pytest.raises(AuthError, match="CLOUDFLARE_ACCOUNT_ID"):
        await Cloudflare().endpoint()


if __name__ == "__main__":
    from sagent.lib.testing.runner import test_main

    test_main(__file__)
