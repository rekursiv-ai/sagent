"""OpenAI API-key provider using the Responses API."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

import os

from sagent.providers.lib.perloop import PerLoop
from sagent.providers.openai.responses import _OpenAIResponsesModel
from sagent.types.providers import ModelCatalog

import sagent.catalog.openai


if TYPE_CHECKING:
    import openai as openai_sdk

else:
    from wrapt import lazy_import

    openai_sdk = lazy_import("openai")

__all__ = ["OpenAI"]


class OpenAI:
    """API-key authentication and loop-local OpenAI SDK ownership."""

    ENV_VAR: ClassVar[str] = "OPENAI_API_KEY"
    BASE_URL: ClassVar[str] = "https://api.openai.com/v1"
    catalog = ModelCatalog(
        rows=sagent.catalog.openai.models(),
        transport=sagent.catalog.openai.api(),
    )

    def __init__(self, *, api_key: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url or self.BASE_URL
        self._sdks: PerLoop[openai_sdk.AsyncOpenAI] = PerLoop(self._make_sdk)

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
        capability, settings = self.catalog.resolve(
            model_id if model_id is not None else "default",
        )
        return _OpenAIResponsesModel(
            provider=self,
            capability=capability,
            settings=settings,
        )

    async def get_sdk(self) -> openai_sdk.AsyncOpenAI:
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

    def _make_sdk(self) -> openai_sdk.AsyncOpenAI:
        return openai_sdk.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=0,
        )
