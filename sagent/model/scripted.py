"""A model that answers from a script: for tests, and for runs without a network.

Each model call takes the next reply: a ``Response``, an exception to raise,
or a function of the request. Every request is kept, so a test can read what
the agent sent.

Typical usage example:

  provider = ScriptedProvider(replies=[reply(calls=[call("Read", file_path="a.py")]), reply("done")])
  async with Agent(model=provider["scripted"], tools=[Read()]) as agent:
      assert await agent.send("read a.py") == "done"
  assert len(provider.requests) == 2
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, ClassVar, override

from sagent.model.capability import ModelCapability, ModelLimits, ModelSettings
from sagent.model.cost import PriceCatalog, TokenCost, TokenCount
from sagent.model.types import Request, Response, TextChunk
from sagent.providers.resources import SharedResources
from sagent.session.types import AssistantMessage, ToolCall
from sagent.types import AsyncResource


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence

    from treekle import PlainTree

    from sagent.providers.table import ModelTable
    from sagent.providers.types import ProviderSharedResources, UsageWindow
    from sagent.session.types import Attachment


__all__ = [
    "Reply",
    "ScriptedModel",
    "ScriptedProvider",
    "call",
    "reply",
]


type Reply = Response | BaseException | Callable[[Request], Response]
"""What one model call does: answer, raise, or answer from the request."""


class ScriptedProvider(AsyncResource):
    """Models that share one script of replies, taken in order.

    Args:
      name: The name the catalog knows this provider by; the section by default.
      resources: What this shares with other providers; ``None`` makes its own.
      replies: The script; a call past its end raises ``RuntimeError``.
      capability: What every model offers; a small window by default.
      models: The models served, looked up and windowed as a vendor's are;
        ``None`` serves ``capability`` under any name.

    """

    section: ClassVar[str] = "scripted"
    """The catalog section this provider would serve."""

    def __init__(
        self,
        *,
        name: str = "",
        resources: ProviderSharedResources | None = None,
        replies: Sequence[Reply] = (),
        capability: ModelCapability | None = None,
        models: ModelTable | None = None,
    ) -> None:
        self._name = name or self.section
        self.models = models
        self._owns_resources = resources is None
        self._resources = SharedResources() if resources is None else resources
        self.replies: deque[Reply] = deque(replies)
        self.requests: list[Request] = []
        prices = PriceCatalog(window=200_000)
        self.capability = (
            capability
            if capability is not None
            else ModelCapability(
                model_id="scripted",
                limits=ModelLimits(
                    max_request_tokens=200_000,
                    max_response_tokens=8_000,
                ),
                prices=prices,
                default=ModelSettings(context_size=prices["auto", "fullcontext"]),
            )
        )

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
        """Return ``()``: a script has no rate limits."""
        return ()

    def __getitem__(self, model: str) -> ScriptedModel:
        if self.models is None:
            return ScriptedModel(
                provider=self,
                capability=self.capability,
                settings=self.capability.default,
            )
        capability, settings = self.models.resolve(model)
        return ScriptedModel(provider=self, capability=capability, settings=settings)

    def __iter__(self) -> Iterator[str]:
        return iter(self.models or (self.capability.model_id,))

    @override
    async def aclose(self) -> None:
        if self._owns_resources:
            await self._resources.aclose()

    def next_reply(self, request: Request) -> Response:
        """Record ``request`` and return the script's next reply to it.

        Args:
          request: What the model was asked.

        Returns:
          response: The reply.

        Raises:
          RuntimeError: The script has run out.
          BaseException: The exception the script holds next.

        """
        self.requests.append(request)
        if not self.replies:
            raise RuntimeError(f"The script ran out at request {len(self.requests)}.")
        step = self.replies.popleft()
        if isinstance(step, BaseException):
            raise step
        return step if isinstance(step, Response) else step(request)


class ScriptedModel(AsyncResource):
    """One model answering from its provider's script."""

    def __init__(
        self,
        *,
        provider: ScriptedProvider,
        capability: ModelCapability,
        settings: ModelSettings,
    ) -> None:
        self._provider = provider
        self._capability = capability
        self._settings = settings

    @property
    def provider(self) -> ScriptedProvider:
        """Return the provider holding the script."""
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
        """Return ``request``'s size in tokens, estimated from its characters.

        Args:
          request: What the model would be asked.

        Returns:
          tokens: The characters of its system blocks and records, scaled.

        """
        chars = sum(len(block) for block in request.system) + sum(
            len(repr(record)) for record in request.records
        )
        return int(chars / self.capability.approx_chars_per_token)

    def attachment_tokens(self, attachment: Attachment) -> int:
        """Return ``attachment``'s bytes, scaled as characters are."""
        return int(len(attachment.data) / self.capability.approx_chars_per_token)

    async def stream(self, request: Request) -> AsyncIterator[TextChunk | Response]:
        """Yield each text record of the reply as one chunk, then the reply.

        Args:
          request: What the model is asked.

        Yields:
          event: A ``TextChunk`` per record that says anything, then the ``Response``.

        """
        response = self.provider.next_reply(request)
        for i, r in enumerate(response.records):
            if isinstance(r, AssistantMessage) and r.content not in (None, ""):
                yield TextChunk(text=r.content, record=i)
        yield response


def reply(
    text: str = "",
    *,
    calls: Sequence[ToolCall] = (),
    tokens: TokenCount | None = None,
    spend: TokenCost | None = None,
    stop_reason: str | None = None,
) -> Response:
    """Return a model turn that says ``text`` and makes ``calls``.

    Args:
      text: What it says; ``""`` for nothing.
      calls: The tool calls it makes.
      tokens: What it was billed; nothing by default.
      spend: What it cost; nothing by default.
      stop_reason: Why it stopped; ``tool_use`` with calls, else ``end_turn``.

    Returns:
      response: The turn.

    """
    records: list[AssistantMessage | ToolCall] = []
    if text:
        records.append(AssistantMessage(content=text))
    records.extend(calls)
    return Response(
        records=tuple(records),
        tokens=tokens if tokens is not None else TokenCount(),
        spend=spend if spend is not None else TokenCost(),
        stop_reason=stop_reason
        if stop_reason is not None
        else "tool_use"
        if calls
        else "end_turn",
    )


def call(name: str, call_id: str = "", **arguments: PlainTree) -> ToolCall:
    """Return a call of the tool ``name`` with ``arguments``.

    Args:
      name: The tool.
      call_id: Its id; ``call_<name>`` when omitted.
      **arguments: What the model passes.

    Returns:
      call: The tool call.

    """
    return ToolCall(call_id=call_id or f"call_{name}", name=name, arguments=arguments)
