"""OpenAI models over the Responses API, billed to an API key.

Typical usage example:

  async with OpenAI() as openai:                # reads OPENAI_API_KEY
      sol = openai["sol"]
      await sol.count(request)                   # exact, from the server
      async for event in sol.stream(request):    # TextChunk..., then Response
          ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final, override

import functools
import os

from treekle import from_plain

from sagent.providers import openai
from sagent.providers.openai.compat import ResponsesCompatible
from sagent.providers.workload_identity import WorkloadIdentity


if TYPE_CHECKING:
    from types import ModuleType

    from sagent.model.types import Request
    from sagent.providers.openai.compat import OpenAIModel


__all__ = [
    "EXCHANGE_URL",
    "OpenAI",
    "exchange",
]

EXCHANGE_URL: Final = "https://auth.openai.com/oauth/token"


class OpenAI(ResponsesCompatible):
    """OpenAI's models, billed to the API key in ``OPENAI_API_KEY``.

    With no key, ``OPENAI_IDENTITY_TOKEN_COMMAND`` turns on workload identity,
    with ``OPENAI_IDENTITY_PROVIDER_ID`` and ``OPENAI_SERVICE_ACCOUNT_ID``. A key wins.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.
      api_key: The key; empty reads ``OPENAI_API_KEY``.
      base_url: The endpoint, up to and including ``/v1``; empty reads
        ``OPENAI_BASE_URL``, as OpenAI's SDK does, then uses OpenAI's.

    """

    section: ClassVar[str] = "openai"
    aliases: ClassVar[tuple[str, ...]] = ("openai-api", "oapi")
    package: ClassVar[ModuleType] = openai
    url: ClassVar[str] = "https://api.openai.com/v1"
    key_variable: ClassVar[str] = "OPENAI_API_KEY"
    base_url_variable: ClassVar[str] = "OPENAI_BASE_URL"

    _identity: WorkloadIdentity | None = None

    @override
    async def credentials(self) -> dict[str, str]:
        """Return the headers that authenticate a request."""
        command = os.environ.get("OPENAI_IDENTITY_TOKEN_COMMAND", "")
        if self._api_key or os.environ.get(self.key_variable, "") or not command:
            return await super().credentials()
        if self._identity is None:
            self._identity = WorkloadIdentity(
                resources=self._resources,
                url=EXCHANGE_URL,
                command=command,
                request=functools.partial(
                    exchange,
                    provider=os.environ.get("OPENAI_IDENTITY_PROVIDER_ID", ""),
                    service_account=os.environ.get("OPENAI_SERVICE_ACCOUNT_ID", ""),
                ),
            )
        return {"authorization": f"Bearer {await self._identity.token()}"}

    @override
    async def reauthenticate(self) -> bool:
        """Renew a workload identity token after a 401; ``False`` for an API key."""
        if self._identity is None:
            return False
        self._identity.forget()
        return True

    @override
    async def count(self, request: Request, *, model: OpenAIModel) -> int:
        """Return the tokens ``request`` would send ``model``, as the server counts them."""
        body = self.encode(request, model=model)
        # ``input_tokens`` rejects the generation knobs it has no use for.
        counted = {
            "model",
            "input",
            "instructions",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
        }
        reply = await self.reply(
            "/responses/input_tokens",
            body={key: value for key, value in body.items() if key in counted},
            headers=self.headers(request),
        )
        return from_plain(reply.get("input_tokens"), int)


def exchange(
    subject: str,
    *,
    provider: str,
    service_account: str,
) -> dict[str, object]:
    """Build OpenAI's token exchange request for an identity token.

    Args:
      subject: The identity token.
      provider: The identity provider's ID.
      service_account: The service account to act as.

    Returns:
      request: The JSON body of the exchange.

    """
    return {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
        "subject_token": subject,
        "identity_provider_id": provider,
        "service_account_id": service_account,
    }
