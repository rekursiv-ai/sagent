"""Any OpenAI-compatible endpoint: a base provider, and the model it serves.

A vendor subclasses ``OpenAICompatible``, names its catalog and endpoint, and
overrides ``encode`` for whatever its server reads differently. The model asks
its provider for every wire decision, so one model class serves every vendor.

Typical usage example:

  class Kimi(OpenAICompatible):
      section = "moonshot"
      package, document = "sagent.providers.moonshot", "moonshot.yaml"
      url = "https://api.moonshot.ai/v1"
      key_variable = "MOONSHOT_API_KEY"
      reasoning_field = "reasoning_content"

  async with Kimi() as kimi:                     # reads MOONSHOT_API_KEY
      async for event in kimi["default"].stream(request):
          ...
  box = Kimi(api_key="none", base_url="http://gpu-box:8000/v1")
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING, ClassVar, Protocol, override

import asyncio
import contextlib
import os

from treekle import json_dumps, json_loads

from sagent.model.exceptions import AuthError, DisconnectedError, RetryableError
from sagent.providers.openai import chat, token_count, wire
from sagent.providers.resources import SharedResources
from sagent.providers.table import ModelTable
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterator, Mapping

    import httpx2

    from sagent.model.capability import ModelCapability, ModelSettings
    from sagent.model.types import Request, Response, TextChunk
    from sagent.providers.types import ProviderSharedResources, UsageWindow
    from sagent.session.types import Attachment
else:
    from wrapt import lazy_import

    httpx2 = lazy_import("httpx2")


__all__ = [
    "OpenAICompatible",
    "OpenAIModel",
    "Reader",
    "models",
]


class Reader(Protocol):
    """Collects a streamed reply's events into a ``Response``."""

    def feed(self, event: Mapping[str, object]) -> TextChunk | None:
        """Take one event, and return the text it adds, if any."""
        ...

    def response(self) -> Response:
        """Return the finished reply."""
        ...


class OpenAICompatible(AsyncResource):
    """Models of one OpenAI-compatible endpoint, over Chat Completions.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.
      api_key: The key; empty reads ``key_variable``.
      base_url: The endpoint; empty uses ``url``, as for a self-hosted server
        that serves a vendor's models.

    """

    section: ClassVar[str] = "openai-compatible"
    """The catalog section whose models and settings this provider serves."""

    package: ClassVar[str] = ""
    """The package holding the catalog document."""

    document: ClassVar[str] = ""
    """The catalog document's file name."""

    url: ClassVar[str] = ""
    """The endpoint, up to and including its version (``.../v1``)."""

    key_variable: ClassVar[str] = ""
    """The environment variable holding the key; empty for an endpoint with none."""

    path: ClassVar[str] = "/chat/completions"
    """Where under ``url`` a reply is streamed from."""

    reasoning_field: ClassVar[str] = ""
    """The message field reasoning streams in and is replayed in; empty for none."""

    temperature: ClassVar[float | None] = 1.0
    """The sampling temperature every request sends; ``None`` sends none."""

    def __init__(
        self,
        *,
        name: str = "",
        resources: ProviderSharedResources | None = None,
        api_key: str = "",
        base_url: str = "",
    ) -> None:
        self._name = name or self.section
        self._owns_resources = resources is None
        self._resources = SharedResources() if resources is None else resources
        self._api_key = api_key
        self._base_url = base_url
        self._usage: tuple[UsageWindow, ...] = ()

    @property
    def name(self) -> str:
        """Return the name the catalog knows this provider by."""
        return self._name

    @property
    def resources(self) -> ProviderSharedResources:
        """Return what this provider shares with others."""
        return self._resources

    @property
    def usage(self) -> tuple[UsageWindow, ...]:
        """Return the account's rate-limit windows, as last reported."""
        return self._usage

    def __getitem__(self, model: str) -> OpenAIModel:
        capability, settings = self.models().resolve(model)
        return OpenAIModel(provider=self, capability=capability, settings=settings)

    def __iter__(self) -> Iterator[str]:
        return iter(self.models())

    @override
    async def aclose(self) -> None:
        if self._owns_resources:
            await self._resources.aclose()

    def models(self) -> ModelTable:
        """Return the models this provider's catalog section serves."""
        return models(self.package, document=self.document, section=self.section)

    async def endpoint(self) -> str:
        """Return the endpoint to reach, up to and including its version."""
        return self._base_url or self.url

    async def credentials(self) -> dict[str, str]:
        """Return the headers that authenticate a request.

        Returns:
          headers: The bearer key; none for an endpoint that takes no key.

        Raises:
          AuthError: The key variable is named but not set.

        """
        key = self._api_key or os.environ.get(self.key_variable, "")
        if key:
            return {"authorization": f"Bearer {key}"}
        if self.key_variable:
            raise AuthError(f"{self.key_variable} is not set.")
        return {}

    async def reauthenticate(self) -> bool:
        """Renew credentials after a 401; ``False`` when there are none to renew."""
        return False

    def headers(self, request: Request) -> dict[str, str]:
        """Return the headers one request adds; none by default."""
        del request
        return {}

    def answered(
        self,
        sent: Mapping[str, str],
        received: Mapping[str, str],
    ) -> None:
        """Note what the server answered a request ``sent`` with; nothing by default."""
        del sent, received

    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        """Return the body that asks ``model`` the request.

        Args:
          request: What the model is asked.
          model: The model asked, with its settings.

        Returns:
          body: The request body.

        """
        return chat.encode(
            request,
            capability=model.capability,
            settings=model.settings,
            reasoning_field=self.reasoning_field,
            temperature=self.temperature,
        )

    def reader(self, *, model: OpenAIModel) -> Reader:
        """Return what collects ``model``'s streamed reply."""
        return chat.StreamReader(
            capability=model.capability,
            settings=model.settings,
            reasoning_field=self.reasoning_field,
        )

    async def count(self, request: Request, *, model: OpenAIModel) -> int:
        """Return an estimate of the tokens ``request`` would send ``model``."""
        return token_count.estimate(request, capability=model.capability)

    async def events(
        self,
        body: Mapping[str, object],
        *,
        headers: Mapping[str, str],
    ) -> AsyncIterator[dict[str, object]]:
        """Stream the reply to ``body``, retrying a short failure before the first event.

        Args:
          body: The request body.
          headers: Headers beyond the credentials.

        Yields:
          event: Each server-sent event, as JSON.

        Raises:
          RetryableError: A failure after the first event, or before it with an
            advised wait longer than 30 seconds, or after five attempts.

        """
        attempt = 0
        while True:
            started = False
            try:
                async with self._reply(
                    self.path,
                    body=body,
                    headers={**headers, "accept": "text/event-stream"},
                ) as response:
                    async for event in wire.events(response.aiter_lines()):
                        started = True
                        yield event
                return
            except RetryableError as failure:
                # Text the caller already showed cannot be taken back, and a
                # long wait is the caller's to sit through or not.
                if started or attempt == 4 or failure.retry_after_sec > 30.0:
                    raise
                await asyncio.sleep(max(failure.retry_after_sec, 0.5 * (1 << attempt)))
                attempt += 1

    async def reply(
        self,
        path: str,
        *,
        body: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict[str, object]:
        """POST ``body`` to ``path`` and return the JSON reply."""
        async with self._reply(path, body=body, headers=headers) as response:
            return json_loads(await response.aread(), dict[str, object])

    @contextlib.asynccontextmanager
    async def _reply(
        self,
        path: str,
        *,
        body: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> AsyncGenerator[httpx2.Response]:
        """Open the reply to ``body``, renewing the login once on a 401."""
        renewable = True
        try:
            while True:
                async with self.resources.http.stream(
                    "POST",
                    f"{(await self.endpoint()).rstrip('/')}{path}",
                    content=json_dumps(body),
                    headers={
                        **await self.credentials(),
                        **headers,
                        "content-type": "application/json",
                    },
                ) as response:
                    self._usage = wire.rate_limits(response.headers) or self._usage
                    self.answered(headers, response.headers)
                    if response.status_code == 401 and renewable:
                        renewable = False
                        if await self.reauthenticate():
                            continue
                    if response.status_code >= 400:
                        text = (await response.aread()).decode(errors="replace")
                        raise wire.to_exception(
                            response.status_code,
                            body=_error_body(text),
                            headers=response.headers,
                        )
                    yield response
                    return
        except httpx2.TransportError as exc:
            raise DisconnectedError(str(exc)) from exc


class OpenAIModel(AsyncResource):
    """One model, served by ``provider`` with ``settings``."""

    def __init__(
        self,
        *,
        provider: OpenAICompatible,
        capability: ModelCapability,
        settings: ModelSettings,
    ) -> None:
        self._provider = provider
        self._capability = capability
        self._settings = settings

    @property
    def provider(self) -> OpenAICompatible:
        """Return the provider serving this model."""
        return self._provider

    @property
    def capability(self) -> ModelCapability:
        """Return every setting this model offers."""
        return self._capability

    @property
    def settings(self) -> ModelSettings:
        """Return the settings chosen."""
        return self._settings

    @settings.setter
    def settings(self, settings: ModelSettings) -> None:
        """Choose ``settings``; raise unless the capability offers each choice."""
        self._capability.check(settings)
        self._settings = settings

    async def count(self, request: Request) -> int:
        """Return how many tokens ``request`` would send, exact where the server says."""
        return await self.provider.count(request, model=self)

    def attachment_tokens(self, attachment: Attachment) -> int:
        """Return about how many tokens ``attachment`` adds, by OpenAI's formula."""
        return token_count.attachment_tokens(attachment, capability=self.capability)

    async def stream(self, request: Request) -> AsyncIterator[TextChunk | Response]:
        """Yield the reply as it is written, then the finished ``Response``.

        Args:
          request: What the model is asked.

        Yields:
          event: Each ``TextChunk`` as the model writes it, then one ``Response``.

        """
        reader = self.provider.reader(model=self)
        body = self.provider.encode(request, model=self)
        headers = self.provider.headers(request)
        async for event in self.provider.events(body, headers=headers):
            chunk = reader.feed(event)
            if chunk is not None:
                yield chunk
        yield reader.response()


@cache
def models(package: str, *, document: str, section: str) -> ModelTable:
    """Return the models one section of a vendor's catalog document serves."""
    return ModelTable.from_yaml(files(package) / document, provider=section)


# A body cut short or not JSON at all is still an error of its status, so it is
# classified by its text rather than failing to parse.
def _error_body(text: str) -> object:
    try:
        return json_loads(text)
    except ValueError:
        return text
