"""Any OpenAI-compatible endpoint: a base provider, and the model it serves.

A vendor subclasses ``OpenAICompatible``, names its catalog and endpoint, and
overrides ``encode`` for whatever its server reads differently. The model asks
its provider for every wire decision, so one model class serves every vendor.

Typical usage example:

  class Kimi(OpenAICompatible):
      package, section = moonshot, "moonshot"
      url = "https://api.moonshot.ai/v1"
      key_variable = "MOONSHOT_API_KEY"
      reasoning_field = "reasoning_content"

  async with Kimi() as kimi:                     # reads MOONSHOT_API_KEY
      async for event in kimi["default"].stream(request):
          ...
  box = Kimi(api_key="none", base_url="http://gpu-box:8000/v1")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, override

import contextlib
import os

from treekle import json_dumps, json_loads

from sagent.model.exceptions import AuthError, DisconnectedError, UnknownModelError
from sagent.providers.openai import chat, responses, token_count, wire
from sagent.providers.resources import SharedResources
from sagent.providers.retry import account_key, observed, retried
from sagent.providers.table import HasModels, ModelTable
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterator, Mapping

    import httpx2

    from sagent.model.capability import (
        ModelCapability,
        ModelSettings,
    )
    from sagent.model.types import Request, Response, TextChunk
    from sagent.providers.types import ProviderResources, UsageWindow
    from sagent.session.types import Attachment
else:
    from wrapt import lazy_import

    httpx2 = lazy_import("httpx2")


__all__ = [
    "OpenAICompatible",
    "OpenAIModel",
    "Reader",
    "ResponsesCompatible",
]


class Reader(Protocol):
    """Collects a streamed reply's events into a ``Response``."""

    def feed(self, event: Mapping[str, object]) -> tuple[TextChunk, ...]:
        """Take one event and return its text and reasoning chunks."""
        ...

    def response(self) -> Response:
        """Return the finished reply."""
        ...


class OpenAICompatible(HasModels, AsyncResource):
    """Models of one OpenAI-compatible endpoint, over Chat Completions.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.
      api_key: The key; empty reads ``key_variable``.
      base_url: The endpoint; empty reads ``base_url_variable``, then ``url``.
      models: A caller's priced model catalog; ``None`` uses the shipped catalog.

    """

    url: ClassVar[str] = ""
    """The endpoint, up to and including its version (``.../v1``)."""

    key_variable: ClassVar[str] = ""
    """The environment variable holding the key; empty for an endpoint with none."""

    path: ClassVar[str] = "/chat/completions"
    """Where under ``url`` a reply is streamed from."""

    base_url_variable: ClassVar[str] = ""
    """The environment variable overriding the endpoint, if any."""

    key_header: ClassVar[str] = "authorization"
    key_prefix: ClassVar[str] = "Bearer "

    max_tokens_field: ClassVar[str] = ""
    """The token cap's spelling; empty uses the model's OpenAI spelling."""

    reasoning_details_field: ClassVar[str] = ""
    """The field holding opaque reasoning blocks that must be replayed."""

    reasoning_content_blocks: ClassVar[bool] = False
    """Whether thinking blocks are replayed inside assistant content."""

    reasoning_field: ClassVar[str] = ""
    """The message field reasoning streams in and is replayed in; empty for none."""

    temperature: ClassVar[float | None] = 1.0
    """The sampling temperature every request sends; ``None`` sends none."""

    def __init__(
        self,
        *,
        name: str = "",
        resources: ProviderResources | None = None,
        api_key: str = "",
        base_url: str = "",
        models: ModelTable | None = None,
    ) -> None:
        self._models = models
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
    def resources(self) -> ProviderResources:
        """Return what this provider shares with others."""
        return self._resources

    @property
    def usage(self) -> tuple[UsageWindow, ...]:
        """Return the account's rate-limit windows, as last reported."""
        return self._usage

    @property
    def gate_key(self) -> str:
        """Return the key every call to this account is paced by."""
        endpoint = (
            self._base_url or os.environ.get(self.base_url_variable, "") or self.url
        )
        key = self._api_key or os.environ.get(self.key_variable) or ""
        return account_key(self.section, credential=f"{endpoint}\0{key}")

    def __getitem__(self, model: str) -> OpenAIModel:
        table = self.models() if self._models is None else self._models
        if not table:
            raise UnknownModelError(
                f"{self.name} requires models=ModelTable(...) with deployment IDs, limits, and prices.",
            )
        capability, settings = table.resolve(model)
        return OpenAIModel(provider=self, capability=capability, settings=settings)

    def __iter__(self) -> Iterator[str]:
        return iter(self.models() if self._models is None else self._models)

    @override
    async def aclose(self) -> None:
        if self._owns_resources:
            await self._resources.aclose()

    async def endpoint(self) -> str:
        """Return the endpoint to reach, up to and including its version."""
        return self._base_url or os.environ.get(self.base_url_variable, "") or self.url

    async def credentials(self) -> dict[str, str]:
        """Return the headers that authenticate a request.

        Returns:
          headers: The bearer key; none for an endpoint that takes no key.

        Raises:
          AuthError: The key variable is named but not set.

        """
        key = self._api_key or os.environ.get(self.key_variable) or ""
        if key:
            return {self.key_header: f"{self.key_prefix}{key}"}
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
        body = chat.encode(
            request,
            capability=model.capability,
            settings=model.settings,
            reasoning_field=self.reasoning_field,
            temperature=self.temperature,
            reasoning_details_field=self.reasoning_details_field,
            reasoning_content_blocks=self.reasoning_content_blocks,
        )
        if self.max_tokens_field:
            for field in ("max_tokens", "max_completion_tokens"):
                if field in body:
                    body[self.max_tokens_field] = body.pop(field)
                    break
        return body

    def reader(self, *, model: OpenAIModel) -> Reader:
        """Return what collects ``model``'s streamed reply.

        Args:
          model: The model whose reply is collected.

        Returns:
          reader: The collector for this endpoint's wire format.

        """
        return chat.StreamReader(
            capability=model.capability,
            settings=model.settings,
            reasoning_field=self.reasoning_field,
            reasoning_details_field=self.reasoning_details_field,
            reasoning_content_blocks=self.reasoning_content_blocks,
        )

    async def count(self, request: Request, *, model: OpenAIModel) -> int:
        """Return an estimate of the tokens ``request`` would send ``model``."""
        return token_count.estimate(request, capability=model.capability)

    async def events(
        self,
        body: Mapping[str, object],
        *,
        headers: Mapping[str, str],
        retry: bool = True,
    ) -> AsyncIterator[dict[str, object]]:
        """Stream the reply to ``body``, retrying a short failure before the first event.

        Args:
          body: The request body.
          headers: Headers beyond the credentials.
          retry: Whether to retry failures before the first event.

        Yields:
          event: Each server-sent event, as JSON.

        Raises:
          RetryableError: A failure after the first event, or before it with an
            advised wait longer than 30 seconds, or after five attempts.

        """

        async def attempt() -> AsyncIterator[dict[str, object]]:
            async with (
                self._reply(
                    self.path,
                    body=body,
                    headers={
                        **headers,
                        "accept": "text/event-stream",  # pragma: no mutate -- HTTP ignores case.
                    },
                ) as response
            ):
                async for event in wire.events(response.aiter_lines()):
                    yield event

        gate, key = self.resources.gate, self.gate_key
        async for event in retried(attempt, gate=gate, key=key) if retry else attempt():
            yield event

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
                    await observed(self.resources.gate, self.gate_key, self._usage)
                    yield response
                    return
        except httpx2.TransportError as exc:
            raise DisconnectedError(str(exc)) from exc


class ResponsesCompatible(OpenAICompatible):
    """Models served over the stateless OpenAI Responses wire protocol."""

    path: ClassVar[str] = "/responses"

    @override
    def encode(self, request: Request, *, model: OpenAIModel) -> dict[str, object]:
        return responses.encode(
            request,
            capability=model.capability,
            settings=model.settings,
            temperature=self.temperature,
        )

    @override
    def reader(self, *, model: OpenAIModel) -> Reader:
        return responses.StreamReader(
            capability=model.capability,
            settings=model.settings,
        )


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
        async for event in self.provider.events(
            body,
            headers=headers,
            retry=request.retry,
        ):
            for chunk in reader.feed(event):
                yield chunk
        yield reader.response()


# A body cut short or not JSON at all is still an error of its status, so it is
# classified by its text rather than failing to parse.
def _error_body(text: str) -> object:
    try:
        return json_loads(text)
    except ValueError:
        return text
