"""Gemini over the Generative Language API, billed to an API key.

Typical usage example:

  async with Google() as google:                 # reads GOOGLE_API_KEY
      pro = google["gemini-pro"]
      await pro.count(request)                     # exact, from the server
      async for event in pro.stream(request):      # TextChunk..., then Response
          ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

import contextlib
import os

from treekle import from_plain, json_dumps, json_loads

from sagent.model.exceptions import AuthError, DisconnectedError
from sagent.providers import google
from sagent.providers.google import wire
from sagent.providers.resources import SharedResources
from sagent.providers.retry import account_key, retried
from sagent.providers.table import HasModels
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterator, Mapping
    from types import ModuleType

    import httpx2

    from sagent.model.capability import ModelCapability, ModelSettings
    from sagent.model.types import Request, Response, TextChunk
    from sagent.providers.table import ModelTable
    from sagent.providers.types import ProviderResources, UsageWindow
    from sagent.session.types import Attachment
else:
    from wrapt import lazy_import

    httpx2 = lazy_import("httpx2")


__all__ = [
    "Google",
    "GoogleModel",
]


class Google(HasModels, AsyncResource):
    """Gemini models, billed to the API key in ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.
      api_key: The API key; empty reads Google's environment variables.
      base_url: The endpoint; empty uses Google's Generative Language API.
      models: A caller's priced model catalog; ``None`` uses the shipped catalog.

    """

    section: ClassVar[str] = "google"
    aliases: ClassVar[tuple[str, ...]] = ("google-api", "gapi")
    package: ClassVar[ModuleType] = google

    def __init__(
        self,
        *,
        name: str = "",
        resources: ProviderResources | None = None,
        api_key: str = "",
        base_url: str = "",
        models: ModelTable | None = None,
    ) -> None:
        self._name = name or self.section
        self._owns_resources = resources is None
        self._resources = SharedResources() if resources is None else resources
        self._api_key = api_key
        self._base_url = base_url
        self._models = models

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
        """Return ``()``: Gemini reports no rate-limit windows."""
        return ()

    def gate_key(self, model: str) -> str:
        """Return the key every call to ``model`` on this account is paced by.

        Args:
          model: The model called, as the catalog names it.

        Returns:
          key: As ``account_key`` makes it, for this API key and endpoint.

        """
        return account_key(
            self.section,
            model=model,
            credential=f"{self._base_url}\0{self._key()}",
        )

    def __getitem__(self, model: str) -> GoogleModel:
        table = self.models() if self._models is None else self._models
        capability, settings = table.resolve(model)
        return GoogleModel(provider=self, capability=capability, settings=settings)

    def __iter__(self) -> Iterator[str]:
        return iter(self.models() if self._models is None else self._models)

    @override
    async def aclose(self) -> None:
        if self._owns_resources:
            await self._resources.aclose()

    async def credentials(self) -> dict[str, str]:
        """Return the headers that authenticate a request.

        Returns:
          headers: The Google API key header.

        """
        key = self._key()
        if not key:
            raise AuthError("Neither GOOGLE_API_KEY nor GEMINI_API_KEY is set.")
        return {"x-goog-api-key": key}

    def _key(self) -> str:
        """Return the API key given, else the environment's; ``""`` for none."""
        # Google's own SDK reads both, the first winning.
        return (
            self._api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        )

    async def reauthenticate(self, *, rejected: Mapping[str, str]) -> bool:
        """Renew credentials after a 401; ``False`` when there are none to renew.

        Args:
          rejected: The headers the refused request carried.

        Returns:
          renewed: Whether a retry may now succeed.

        """
        del rejected
        return False

    def endpoint(self, method: str, *, model: str) -> str:
        """Return the URL of ``method`` (``countTokens``) for ``model``."""
        base = self._base_url or "https://generativelanguage.googleapis.com/v1beta"
        return f"{base.rstrip('/')}/models/{model}:{method}"

    async def envelope(
        self,
        body: Mapping[str, object],
        *,
        model: str,
        request: Request,
    ) -> Mapping[str, object]:
        """Return what is posted to ask ``model`` ``request``, as ``body``: itself."""
        del model, request
        return body

    def unwrap(self, event: dict[str, object]) -> dict[str, object]:
        """Return the ``GenerateContentResponse`` a streamed event holds: itself."""
        return event

    async def count(
        self,
        body: Mapping[str, object],
        *,
        capability: ModelCapability,
    ) -> int:
        """Return the tokens ``body`` would send, as the server counts them.

        Args:
          body: A ``generateContent`` request body.
          capability: The model it would be sent to.

        Returns:
          tokens: Input tokens, system prompt and tools included.

        """
        model = capability.wire_model_id or capability.model_id
        reply = await self.reply(
            self.endpoint("countTokens", model=model),
            body={"generateContentRequest": {"model": f"models/{model}", **body}},
        )
        return from_plain(reply.get("totalTokens"), int)

    async def events(
        self,
        body: Mapping[str, object],
        *,
        model: str,
        request: Request,
    ) -> AsyncIterator[dict[str, object]]:
        """Stream the reply to ``body``, retrying a short failure before the first event.

        Args:
          body: The ``generateContent`` request body.
          model: The vendor's id of the model asked.
          request: What ``body`` encodes.

        Yields:
          event: Each server-sent ``GenerateContentResponse``, as JSON.

        Raises:
          RetryableError: A failure after the first event, or before it with an
            advised wait longer than 30 seconds, or after five attempts.

        """
        url = f"{self.endpoint('streamGenerateContent', model=model)}?alt=sse"
        posted = await self.envelope(body, model=model, request=request)

        async def attempt() -> AsyncIterator[dict[str, object]]:
            async with self._reply(url, body=posted) as response:
                async for event in wire.events(response.aiter_lines()):
                    yield self.unwrap(event)

        async for event in retried(
            attempt,
            gate=self.resources.gate,
            key=self.gate_key(model),
        ):
            yield event

    async def reply(self, url: str, *, body: Mapping[str, object]) -> dict[str, object]:
        """POST ``body`` to ``url`` and return the JSON reply."""
        async with self._reply(url, body=body) as response:
            return json_loads(await response.aread(), dict[str, object])

    @contextlib.asynccontextmanager
    async def _reply(
        self,
        url: str,
        *,
        body: Mapping[str, object],
    ) -> AsyncGenerator[httpx2.Response]:
        """Open the reply to ``body``, renewing the login once on a 401."""
        renewable = True
        try:
            while True:
                headers = {
                    **await self.credentials(),
                    "content-type": "application/json",
                }
                async with self.resources.http.stream(
                    "POST",
                    url,
                    content=json_dumps(body),
                    headers=headers,
                ) as response:
                    if response.status_code == 401 and renewable:
                        renewable = False
                        if await self.reauthenticate(rejected=headers):
                            continue
                    if response.status_code >= 400:
                        raise wire.to_exception(
                            response.status_code,
                            body=_error_body(await response.aread()),
                            headers=response.headers,
                        )
                    yield response
                    return
        except httpx2.TransportError as exc:
            raise DisconnectedError(str(exc)) from exc


class GoogleModel(AsyncResource):
    """One Gemini model, served by ``provider`` with ``settings``."""

    def __init__(
        self,
        *,
        provider: Google,
        capability: ModelCapability,
        settings: ModelSettings,
    ) -> None:
        self._provider = provider
        self._capability = capability
        self._settings = settings

    @property
    def provider(self) -> Google:
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
        """Return the tokens ``request`` would send, exact where the provider counts.

        Args:
          request: What the model would be asked.

        Returns:
          tokens: Input tokens, system prompt and tools included.

        """
        return await self.provider.count(
            self.encode(request),
            capability=self.capability,
        )

    def encode(self, request: Request) -> dict[str, object]:
        """Return the ``generateContent`` body that asks this model ``request``."""
        return wire.encode(request, capability=self.capability, settings=self.settings)

    def attachment_tokens(self, attachment: Attachment) -> int:
        """Return about how many tokens ``attachment`` adds, by Gemini's formula."""
        return wire.attachment_tokens(attachment, limits=self.capability.limits)

    async def stream(self, request: Request) -> AsyncIterator[TextChunk | Response]:
        """Yield the reply as it is written, then the finished ``Response``.

        Args:
          request: What the model is asked.

        Yields:
          event: Each ``TextChunk`` as the model writes it, then one ``Response``.

        """
        reader = wire.StreamReader(capability=self.capability)
        body = self.encode(request)
        model = self.capability.wire_model_id or self.capability.model_id
        async for event in self.provider.events(body, model=model, request=request):
            for chunk in reader.feed(event):
                yield chunk
        yield reader.response()


# Not every error body is JSON, and one that is not still says what failed.
def _error_body(raw: bytes) -> object:
    """Return an error reply's JSON, or its text when it is not JSON."""
    try:
        return json_loads(raw)
    except ValueError:
        return raw.decode(errors="replace")
