"""Gemini over the Generative Language API, billed to an API key.

Typical usage example:

  async with Google() as google:                 # reads GOOGLE_API_KEY
      pro = google["gemini-pro"]
      await pro.count(request)                     # exact, from the server
      async for event in pro.stream(request):      # TextChunk..., then Response
          ...
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING, ClassVar, override

import asyncio
import contextlib
import os

from treekle import from_plain, json_dumps, json_loads

from sagent.model.exceptions import AuthError, DisconnectedError, RetryableError
from sagent.providers.google import wire
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
    "Google",
    "GoogleModel",
    "models",
]


class Google(AsyncResource):
    """Gemini models, billed to the API key in ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.

    """

    section: ClassVar[str] = "google"
    """The catalog section whose models and settings this provider serves."""

    def __init__(
        self,
        *,
        name: str = "",
        resources: ProviderSharedResources | None = None,
    ) -> None:
        self._name = name or self.section
        self._owns_resources = resources is None
        self._resources = SharedResources() if resources is None else resources

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
        """Return ``()``: Gemini reports no rate-limit windows."""
        return ()

    def __getitem__(self, model: str) -> GoogleModel:
        capability, settings = models(self.section).resolve(model)
        return GoogleModel(provider=self, capability=capability, settings=settings)

    def __iter__(self) -> Iterator[str]:
        return iter(models(self.section))

    @override
    async def aclose(self) -> None:
        if self._owns_resources:
            await self._resources.aclose()

    async def credentials(self) -> dict[str, str]:
        """Return the headers that authenticate a request."""
        # Google's own SDK reads both, the first winning.
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
        if not key:
            raise AuthError("Neither GOOGLE_API_KEY nor GEMINI_API_KEY is set.")
        return {"x-goog-api-key": key}

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
        base = "https://generativelanguage.googleapis.com/v1beta"
        return f"{base}/models/{model}:{method}"

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
        attempt = 0
        while True:
            streamed: dict[str, object] | None = None
            try:
                async with self._reply(url, body=posted) as response:
                    async for event in wire.events(response.aiter_lines()):
                        streamed = event
                        yield self.unwrap(event)
                return
            except RetryableError as failure:
                # Text the caller already showed cannot be taken back, and a
                # long wait is the caller's to sit through or not.
                if (
                    streamed is not None
                    or attempt == 4
                    or failure.retry_after_sec > 30.0
                ):
                    raise
                await asyncio.sleep(max(failure.retry_after_sec, 0.5 * (1 << attempt)))
                attempt += 1

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
            wire.encode(request, capability=self.capability, settings=self.settings),
            capability=self.capability,
        )

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
        body = wire.encode(request, capability=self.capability, settings=self.settings)
        model = self.capability.wire_model_id or self.capability.model_id
        async for event in self.provider.events(body, model=model, request=request):
            for chunk in reader.feed(event):
                yield chunk
        yield reader.response()


@cache
def models(section: str) -> ModelTable:
    """Return the models one section of the Google catalog serves."""
    return ModelTable.from_yaml(
        files("sagent.providers.google") / "google.yaml",
        provider=section,
    )


# Not every error body is JSON, and one that is not still says what failed.
def _error_body(raw: bytes) -> object:
    """Return an error reply's JSON, or its text when it is not JSON."""
    try:
        return json_loads(raw)
    except ValueError:
        return raw.decode(errors="replace")
