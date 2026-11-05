"""Bedrock's OpenAI Chat endpoint authenticated by a Bedrock API key.

Set ``AWS_REGION`` or pass ``base_url`` for a regional endpoint. This provider
supports Bedrock API keys, not AWS access-key/SigV4 credentials.
"""

from types import ModuleType
from typing import ClassVar, override

import os

from sagent.model.exceptions import AuthError
from sagent.providers import bedrock
from sagent.providers.openai.compat import OpenAICompatible


__all__ = ["Bedrock"]


class Bedrock(OpenAICompatible):
    """Serve Bedrock Chat models using ``AWS_BEARER_TOKEN_BEDROCK``."""

    section: ClassVar[str] = "bedrock"
    aliases: ClassVar[tuple[str, ...]] = ("aws-bedrock",)
    package: ClassVar[ModuleType] = bedrock
    key_variable: ClassVar[str] = "AWS_BEARER_TOKEN_BEDROCK"
    base_url_variable: ClassVar[str] = "BEDROCK_BASE_URL"
    max_tokens_field: ClassVar[str] = "max_completion_tokens"

    @override
    async def endpoint(self) -> str:
        if endpoint := await super().endpoint():
            return endpoint
        region = os.environ.get("AWS_REGION") or os.environ.get(
            "AWS_DEFAULT_REGION",
            "",
        )
        if not region:
            raise AuthError("Set AWS_REGION or pass base_url for Bedrock.")
        return f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1"
