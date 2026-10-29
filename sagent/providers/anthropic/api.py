"""Claude over the Messages API, billed to an API key.

Typical usage example:

  async with Anthropic() as anthropic:          # reads ANTHROPIC_API_KEY
      opus = anthropic["opus"]
      await opus.count(request)                  # exact, from the server
      async for event in opus.stream(request):   # TextChunk..., then Response
          ...
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING, ClassVar, override

import asyncio
import contextlib
import os
import time

from treekle import from_plain, json_dumps, json_loads

from sagent.model.exceptions import AuthError, DisconnectedError, RetryableError
from sagent.providers.anthropic import wire
from sagent.providers.resources import SharedResources
from sagent.providers.table import ModelTable
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Iterator,
        Mapping,
        Sequence,
    )

    import httpx2

    from sagent.model.capability import ModelCapability, ModelSettings
    from sagent.model.types import Request, Response, TextChunk
    from sagent.providers.types import ProviderSharedResources, UsageWindow
    from sagent.session.types import Attachment
else:
    from wrapt import lazy_import

    httpx2 = lazy_import("httpx2")


__all__ = [
    "Anthropic",
    "AnthropicModel",
    "models",
]


class Anthropic(AsyncResource):
    """Claude models, billed to the API key in ``ANTHROPIC_API_KEY``.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes and
        closes its own.

    """

    section: ClassVar[str] = "anthropic"
    """The catalog section whose models and settings this provider serves."""

    betas: ClassVar[tuple[str, ...]] = ()
    """Beta features every request names, whatever it asks for."""

    def __init__(
        self,
        *,
        name: str = "",
        resources: ProviderSharedResources | None = None,
    ) -> None:
        self._name = name or self.section
        self._owns_resources = resources is None
        self._resources = SharedResources() if resources is None else resources
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
        """Return the account's rate-limit windows; ``()`` for an API key."""
        return self._usage

    def __getitem__(self, model: str) -> AnthropicModel:
        capability, settings = models(self.section).resolve(model)
        return AnthropicModel(provider=self, capability=capability, settings=settings)

    def __iter__(self) -> Iterator[str]:
        return iter(models(self.section))

    @override
    async def aclose(self) -> None:
        if self._owns_resources:
            await self._resources.aclose()

    async def credentials(self) -> dict[str, str]:
        """Return the headers that authenticate a request."""
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise AuthError("ANTHROPIC_API_KEY is not set.")
        return {"x-api-key": key}

    async def reauthenticate(self) -> bool:
        """Renew credentials after a 401; ``False`` when there are none to renew."""
        return False

    def system(
        self,
        request: Request,
        *,
        settings: ModelSettings,
    ) -> list[dict[str, object]]:
        """Return the system blocks a request opens with."""
        return wire.system_blocks(request.system, settings=settings)

    def headers(self, request: Request) -> dict[str, str]:
        """Return the headers one request adds; none by default."""
        del request
        return {}

    async def events(
        self,
        body: Mapping[str, object],
        *,
        betas: Sequence[str],
        headers: Mapping[str, str],
    ) -> AsyncIterator[dict[str, object]]:
        """Stream the reply to ``body``, retrying a short failure before the first event.

        Args:
          body: The ``/v1/messages`` request body.
          betas: Beta features this request needs, beyond ``self.betas``.
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
                    "/v1/messages",
                    body=body,
                    betas=betas,
                    headers=headers,
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
        betas: Sequence[str],
        headers: Mapping[str, str],
    ) -> dict[str, object]:
        """POST ``body`` to ``path`` and return the JSON reply.

        Args:
          path: Where under the API host to post.
          body: The request body.
          betas: Beta features this request needs, beyond ``self.betas``.
          headers: Headers beyond the credentials.

        Returns:
          reply: The JSON the server answered with.

        """
        async with self._reply(
            path,
            body=body,
            betas=betas,
            headers=headers,
        ) as response:
            return json_loads(await response.aread(), dict[str, object])

    @contextlib.asynccontextmanager
    async def _reply(
        self,
        path: str,
        *,
        body: Mapping[str, object],
        betas: Sequence[str],
        headers: Mapping[str, str],
    ) -> AsyncGenerator[httpx2.Response]:
        """Open the reply to ``body``, renewing the login once on a 401."""
        renewable = True
        try:
            while True:
                async with self.resources.http.stream(
                    "POST",
                    f"https://api.anthropic.com{path}",
                    content=json_dumps(body),
                    headers={
                        **await self.credentials(),
                        **headers,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                        "anthropic-beta": ",".join(
                            dict.fromkeys((*self.betas, *betas)),
                        ),
                    },
                ) as response:
                    self._usage = wire.rate_limits(response.headers) or self._usage
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


class AnthropicModel(AsyncResource):
    """One Claude model, served by ``provider`` with ``settings``."""

    def __init__(
        self,
        *,
        provider: Anthropic,
        capability: ModelCapability,
        settings: ModelSettings,
    ) -> None:
        self._provider = provider
        self._capability = capability
        self._settings = settings
        self._answered_at = 0.0

    @property
    def provider(self) -> Anthropic:
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
        """Return the tokens ``request`` would send, as the server counts them.

        Args:
          request: What the model would be asked.

        Returns:
          tokens: Input tokens, system prompt and tools included.

        """
        body, betas = self._encode(request)
        # ``count_tokens`` rejects the generation knobs it has no use for.
        for knob in ("max_tokens", "stream", "service_tier", "speed", "output_config"):
            body.pop(knob, None)
        reply = await self.provider.reply(
            "/v1/messages/count_tokens",
            body=body,
            betas=betas,
            headers=self.provider.headers(request),
        )
        return from_plain(reply.get("input_tokens"), int)

    def attachment_tokens(self, attachment: Attachment) -> int:
        """Return about how many tokens ``attachment`` adds, by Anthropic's formula."""
        return wire.attachment_tokens(attachment, limits=self.capability.limits)

    async def stream(self, request: Request) -> AsyncIterator[TextChunk | Response]:
        """Yield the reply as it is written, then the finished ``Response``.

        Args:
          request: What the model is asked.

        Yields:
          event: Each ``TextChunk`` as the model writes it, then one ``Response``.

        """
        reader = wire.StreamReader(capability=self.capability, settings=self.settings)
        body, betas = self._encode(request)
        async for event in self.provider.events(
            body,
            betas=betas,
            headers=self.provider.headers(request),
        ):
            chunk = reader.feed(event)
            if chunk is not None:
                yield chunk
        self._answered_at = time.time()
        yield reader.response()

    def _encode(self, request: Request) -> tuple[dict[str, object], list[str]]:
        idle_sec = time.time() - self._answered_at
        return wire.encode(
            request,
            capability=self.capability,
            settings=self.settings,
            system=self.provider.system(request, settings=self.settings),
            cache_expired=self._answered_at > 0.0
            and idle_sec > self.settings.cache_ttl_sec,
        )


@cache
def models(section: str) -> ModelTable:
    """Return the models one section of the Anthropic catalog serves."""
    return ModelTable.from_yaml(
        files("sagent.providers.anthropic") / "anthropic.yaml",
        provider=section,
    )


def _error_body(text: str) -> object:
    """Return an error reply's JSON, or its text when it is not JSON."""
    # A proxy's error page, or a reply cut off mid-body, may start like JSON.
    try:
        return json_loads(text)
    except ValueError:
        return text
