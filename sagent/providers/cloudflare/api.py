"""Cloudflare Workers AI through its account-scoped Chat endpoint."""

from types import ModuleType
from typing import ClassVar, override

import os

from sagent.model.exceptions import AuthError
from sagent.providers import cloudflare
from sagent.providers.openai.compat import OpenAICompatible


__all__ = ["Cloudflare"]


class Cloudflare(OpenAICompatible):
    """Serve Workers AI models using a Cloudflare API token."""

    section: ClassVar[str] = "cloudflare"
    aliases: ClassVar[tuple[str, ...]] = ("workers-ai",)
    package: ClassVar[ModuleType] = cloudflare
    key_variable: ClassVar[str] = "CLOUDFLARE_API_TOKEN"
    base_url_variable: ClassVar[str] = "CLOUDFLARE_BASE_URL"
    max_tokens_field: ClassVar[str] = "max_tokens"

    @override
    async def endpoint(self) -> str:
        if endpoint := await super().endpoint():
            return endpoint
        account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        if not account or not account.isalnum():
            raise AuthError("Set CLOUDFLARE_ACCOUNT_ID or pass base_url.")
        return f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"
