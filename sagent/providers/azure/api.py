"""Azure OpenAI v1 Responses with deployment-specific model tables.

Set ``AZURE_OPENAI_BASE_URL`` to ``https://RESOURCE.openai.azure.com/openai/v1``.
Pass ``models=ModelTable(...)`` with deployment names as ``wire_model_id`` and
rates for the deployment's region and billing agreement. No generic paid rate
is assumed. Counting estimates locally; Azure has no input_tokens endpoint.
"""

from types import ModuleType
from typing import ClassVar

from sagent.providers import azure
from sagent.providers.openai.compat import ResponsesCompatible


__all__ = ["Azure"]


class Azure(ResponsesCompatible):
    """Serve Azure deployments through the OpenAI v1 Responses API."""

    section: ClassVar[str] = "azure"
    aliases: ClassVar[tuple[str, ...]] = ("azure-openai",)
    package: ClassVar[ModuleType] = azure
    key_variable: ClassVar[str] = "AZURE_OPENAI_API_KEY"
    base_url_variable: ClassVar[str] = "AZURE_OPENAI_BASE_URL"
    key_header: ClassVar[str] = "api-key"
    key_prefix: ClassVar[str] = ""
