"""What a model is asked, and how it answers.

The ``Model`` that answers lives in ``providers.types``, beside the provider
serving it.

Typical usage example:

  request = Request(system=("Be terse.",), records=tuple(context), tools=tools)
  async for event in model.stream(request):
      if isinstance(event, TextChunk):
          print(event.text, end="")
      else:
          session.append(*event.records)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import uuid

from sagent.model.cost import TokenCost, TokenCount
from sagent.session.types import (
    AssistantMessage,
    SessionRecord,
    Thinking,
    ToolCall,
)
from sagent.tools.types import Tool


if TYPE_CHECKING:
    from sagent.model.capability import ModelLimits


__all__ = [
    "Request",
    "Response",
    "TextChunk",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Request:
    """Everything one model call reads."""

    system: tuple[str, ...] = ()
    """System prompt blocks, the most stable first, so their cache lasts."""

    records: Sequence[SessionRecord] = ()
    tools: Sequence[Tool] = ()

    allowed: frozenset[str] | None = None
    """The tools the model may call now; ``None`` allows every one offered."""

    max_response_tokens: int = 0
    """The most the reply may hold, as the window leaves room; ``0`` for the model's."""

    purpose: Literal["main", "compaction"] = "main"
    """Why the call is made: the agent's own turn, or a summary of its context."""

    compaction: Literal["", "auto", "manual"] = ""
    """What started the compaction this call makes, or, on the first main call
    after one, what started that; ``""`` for neither."""

    session: str = ""
    """The conversation: its root agent's id, the same across compactions and resumes."""

    agent: str = ""
    """The id of the agent asking; a root's is ``session``."""

    parent: str = ""
    """The id of the agent that spawned the one asking; ``""`` for a root."""

    parent_turn: int = 0
    """The parent's ``turn`` when it spawned the one asking."""

    name: str = ""
    """Where the agent asking stands: ``/root``, a child ``/root/<label>``."""

    window: int = 0
    """How many compactions the asking agent's context has had."""

    turn: int = 0
    """How many messages the agent has been sent; a turn is the calls between two."""

    turn_started_ms: int = 0
    """When the turn began, in milliseconds since the epoch; ``0`` before any."""

    step: int = 0
    """How many calls the agent made earlier in this turn."""

    root_prompt: str = ""
    """The ``prompt_id`` of the root's turn that spawned this agent's line; ``""``
    for a root."""

    @property
    def prompt_id(self) -> str:
        """Return the id of the turn this call serves, the same on every resume."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.agent}/{self.turn}"))

    @property
    def call_id(self) -> str:
        """Return the id of this call, its retries alike."""
        return str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{self.agent}/{self.turn}/{self.step}"),
        )

    def response_tokens(self, limits: ModelLimits) -> int:
        """Return the most the reply may hold: the model's limit, or this, if less."""
        cap, limit = self.max_response_tokens, limits.max_response_tokens
        return cap if cap > 0 and (cap < limit or limit == 0) else limit


@dataclass(frozen=True, slots=True, kw_only=True)
class TextChunk:
    """A few words of the reply, sent as the model writes them."""

    text: str
    thinking: bool = False
    """Whether the words are the model's reasoning, not its reply."""

    record: int = 0
    """The index in ``Response.records`` of the record the words join."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Response:
    """The finished reply."""

    records: tuple[AssistantMessage | Thinking | ToolCall, ...]
    tokens: TokenCount
    spend: TokenCost
    stop_reason: str
    """Why the model stopped: ``end_turn``, ``tool_use``, ``max_tokens``, or ``refusal``; another vendor reason as spelled."""
