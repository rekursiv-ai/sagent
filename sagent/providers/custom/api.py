"""Any OpenAI-compatible endpoint, routed to custom base URLs and models."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

import os

from treekle import json_dumps, json_loads

from sagent.lib.files.owner_only import write_private
from sagent.lib.userdirs import config_dir
from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
from sagent.model.cost import PriceCatalog, PriceCondition, TokenPrice
from sagent.model.exceptions import AuthError, UnknownModelError
from sagent.providers import custom
from sagent.providers.openai.compat import OpenAICompatible, OpenAIModel
from sagent.providers.table import ModelTags


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import ModuleType


__all__ = [
    "Custom",
    "custom_config_path",
    "read_custom_config",
    "write_custom_config",
]


def custom_config_path() -> Path:
    """Return the path to the user's stored custom provider configuration."""
    return config_dir() / "rekursiv-ai" / "secrets" / "custom.json"


def read_custom_config() -> dict[str, str]:
    """Read the user's stored custom provider settings, or empty dict if none."""
    path = custom_config_path()
    if path.exists():
        try:
            return json_loads(path.read_text("utf-8"), dict[str, str])
        except (OSError, ValueError):
            return {}
    return {}


def write_custom_config(
    *,
    base_url: str,
    api_key: str | None = None,
    model: str | None = None,
) -> None:
    """Persist custom provider settings to the user's private secrets directory."""
    path = custom_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_custom_config()
    if base_url:
        existing["base_url"] = base_url.strip()
    if api_key is not None:
        existing["api_key"] = api_key.strip()
    if model is not None:
        if stripped_model := model.strip():
            existing["model"] = stripped_model
        else:
            existing.pop("model", None)
    write_private(path, json_dumps(existing))


class Custom(OpenAICompatible):
    """Serve any OpenAI-compatible API using custom environment variables or parameters.

    Endpoint resolution checks:
      1. Explicit ``base_url`` passed to constructor
      2. Stored ``base_url`` in ``~/.config/rekursiv-ai/secrets/custom.json``
      3. ``CUSTOM_BASE_URL`` environment variable

    API key resolution checks:
      1. Explicit ``api_key`` passed to constructor
      2. Stored ``api_key`` in ``~/.config/rekursiv-ai/secrets/custom.json``
      3. ``CUSTOM_API_KEY`` environment variable
      (Keyless operation is permitted if none is set, e.g. for local proxies).

    Model resolution:
      Pass any exact model name (e.g., ``gpt-6-luna``), or configure a model for
      Sagent's role aliases. No model name is invented when neither is provided.
    """

    section: ClassVar[str] = "custom"
    aliases: ClassVar[tuple[str, ...]] = ("custom-api", "generic", "proxy")
    package: ClassVar[ModuleType] = custom
    url: ClassVar[str] = ""
    key_variable: ClassVar[str] = "CUSTOM_API_KEY"
    base_url_variable: ClassVar[str] = "CUSTOM_BASE_URL"
    model_roles: ClassVar[frozenset[str]] = frozenset(
        {"default", "best", "everyday", "utility"}
    )
    # Let the upstream model choose its sampling default. Compatibility gateways
    # do not consistently accept a temperature parameter.
    temperature: ClassVar[float | None] = None
    # Gateways commonly return one of these upstream OpenAI-compatible keys.
    # Reading them is safe; replaying one requires an endpoint-specific contract.
    reasoning_fields: ClassVar[tuple[str, ...]] = (
        "reasoning_content",
        "reasoning",
        "reasoning_details",
    )

    def configure(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Persist settings supplied by a frontend for this provider."""
        write_custom_config(base_url=base_url, api_key=api_key, model=model)

    @override
    async def endpoint(self) -> str:
        """Return the target endpoint up to and including its version."""
        stored = read_custom_config().get("base_url", "")
        url = (
            self._base_url
            or stored
            or os.environ.get(self.base_url_variable, "")
            or self.url
        )
        if not url:
            raise AuthError(
                f"{self.section} provider requires {self.base_url_variable} or UI configuration to be set.",
            )
        return url

    @override
    async def credentials(self) -> dict[str, str]:
        """Return the authentication headers for the request."""
        stored = read_custom_config().get("api_key", "")
        key = (
            self._api_key
            or stored
            or os.environ.get(self.key_variable, "")
        )
        if key:
            return {"authorization": f"Bearer {key}"}
        return {}

    @override
    def __getitem__(self, model: str) -> OpenAIModel:
        tags = ModelTags.parse(model)
        if tags.model in self.model_roles:
            configured_model = self._configured_model()
            if not configured_model:
                raise UnknownModelError(
                    "Custom provider requires a model; configure one or select "
                    "custom/<model-id>.",
                )
            exact_name = configured_model
        else:
            exact_name = tags.model
        if not exact_name:
            raise UnknownModelError(
                "Custom provider requires a model; configure one or select "
                "custom/<model-id>.",
            )

        if isinstance(tags.context, int):
            window = tags.context
        else:
            env_window = os.environ.get("CUSTOM_CONTEXT_WINDOW", "")
            window = int(env_window) if env_window.isdigit() else 0

        prices = PriceCatalog(
            prices={PriceCondition(): TokenPrice()},
            window=window,
        )
        capability = ModelCapability(
            model_id=exact_name,
            wire_model_id=exact_name,
            limits=ModelLimits(
                max_request_tokens=window,
            ),
            prices=prices,
            default=ModelSettings(
                context_size=prices["auto", "fullcontext"],
                manage_context_server_side=False,
            ),
        )
        return OpenAIModel(
            provider=self,
            capability=capability,
            settings=capability.default,
        )

    @override
    def __iter__(self) -> Iterator[str]:
        configured_model = self._configured_model()
        return iter((configured_model,)) if configured_model else iter(())

    def _configured_model(self) -> str:
        """Return only a model explicitly supplied through config or environment."""
        model = read_custom_config().get("model", "") or os.environ.get(
            "CUSTOM_MODEL", ""
        )
        return model.strip()
