"""OpenAI API-key provider using the Responses API."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Self

import os

from sagent.catalog import openai as openai_catalog
from sagent.providers.lib.perloop import PerLoop
from sagent.providers.openai.responses import _OpenAIResponsesModel
from sagent.types.capability import ModelCapability
from sagent.types.providers import ModelRole, resolve


if TYPE_CHECKING:
    import openai
else:
    from wrapt import lazy_import

    openai = lazy_import("openai")

__all__ = ["OpenAI"]


class OpenAI:
    """API-key authentication and loop-local OpenAI SDK ownership."""

    DEFAULT_MODEL: ClassVar[str] = "gpt-5.6-sol+1m"
    DEFAULT_UTILITY_MODEL: ClassVar[str] = "gpt-5.4-mini"
    ENV_VAR: ClassVar[str] = "OPENAI_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.openai.com/v1"
    CAPABILITIES: ClassVar[Mapping[str, ModelCapability]] = openai_catalog.models()
    TRANSPORT: ClassVar[ModelCapability] = openai_catalog.api()

    def __init__(self, *, api_key: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url or self.BASE_URL
        self._sdks: PerLoop[openai.AsyncOpenAI] = PerLoop(self._make_sdk)

    @property
    def ROLES(self) -> Mapping[ModelRole, str]:  # noqa: N802 -- provider protocol spelling.
        """Role names resolved by model construction."""
        return MappingProxyType(
            {"default": self.DEFAULT_MODEL, "utility": self.DEFAULT_UTILITY_MODEL}
        )

    @classmethod
    def from_key(cls, api_key: str, *, base_url: str | None = None) -> Self:
        """Construct a provider from an API key.

        Args:
          api_key: OpenAI API key.
          base_url: Optional Responses endpoint override.

        Returns:
          provider: Configured provider.

        """
        return cls(api_key=api_key, base_url=base_url)

    @classmethod
    def from_env(cls, *, base_url: str | None = None) -> Self:
        """Construct a provider from OPENAI_API_KEY.

        Args:
          base_url: Optional Responses endpoint override.

        Returns:
          provider: Configured provider.

        """
        key = os.environ.get(cls.ENV_VAR, "")
        if not key:
            raise RuntimeError(f"{cls.__name__} API key not configured.")
        return cls(api_key=key, base_url=base_url)

    def model(self, model_id: str | None = None) -> _OpenAIResponsesModel:
        """Resolve a catalog model or role to a Responses backend.

        Args:
          model_id: Model id, context-tagged id, or role.

        Returns:
          model: Configured Responses model.

        """
        capability, settings = resolve(
            model_id if model_id is not None else "default",
            models=self.CAPABILITIES,
            roles=self.ROLES,
            transport=self.TRANSPORT,
        )
        return _OpenAIResponsesModel(
            provider=self, capability=capability, settings=settings
        )

    def utility_model(self) -> _OpenAIResponsesModel:
        """Return the utility-role model.

        Returns:
          model: Utility Responses backend.

        """
        return self.model("utility")

    async def get_sdk(self) -> openai.AsyncOpenAI:
        """Get this event loop's SDK client.

        Returns:
          sdk: Reusable client owned by this provider.

        """
        return self._sdks.get()

    async def close_sdk(self) -> None:
        """Close and release this loop's SDK without creating one."""
        sdk = self._sdks.peek()
        self._sdks.clear()
        if sdk is not None:
            await sdk.close()

    def _make_sdk(self) -> openai.AsyncOpenAI:
        return openai.AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, max_retries=0
        )
