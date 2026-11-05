"""The summary compactor: starts the session over from a summary of it.

The runtime calls :meth:`SummaryCompactor.compact` when it handles a
``Compact``: on ``run``, or on its own before a model call the compactor says
is due. The summary keeps a recent tail verbatim when asked, restores the skills
the window loaded, and lists the jobs still running. It brings back no file the
model read: the model reads again, in one turn, what it still needs, as
``Write`` and ``Edit`` make it.

Two styles: ``summary`` condenses what the model sees, asked as its next
request so the prompt cache serves it; ``session-audit`` reads the whole
session from its journal -- every message the user sent, every reply, a line
per tool call -- and audits it, as the session-audit skill does. :func:`halve`
says how much of an audit too long for one request is summarized first.

Typical usage example:

  async with Agent(model=opus, compactor=SummaryCompactor) as agent:
      await agent.session.compactor.run("keep the API design")

  Agent(model=opus, compactor=partial(SummaryCompactor, keep_recent=6))
"""

from __future__ import annotations

from bisect import bisect_right
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from importlib.resources import files
from itertools import accumulate
from typing import TYPE_CHECKING, Literal, cast
from xml.sax.saxutils import escape

import asyncio
import logging
import math
import re
import time

from treekle import PlainTree, from_plain, json_dumps, mutable

from sagent.agent.scrunch import safe_boundaries
from sagent.model.capability import ThinkingEffort, literal_members
from sagent.model.exceptions import ContextOverflowError, RetryableError
from sagent.model.types import Request, Response
from sagent.runtime import types
from sagent.runtime.tape import (
    ContextSplice,
    calls,
    coalesce_roles,
    full_tape_mask,
    unpaired_call_ids,
)
from sagent.session.types import (
    AgentToAgentMessage,
    AssistantMessage,
    Attachment,
    SystemMessage,
    ToolCall,
    ToolResult,
    TurnContext,
    UserMessage,
)
from sagent.tools.skill import Skill


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence

    from sagent.agent.types import Agent
    from sagent.providers.types import Model
    from sagent.runtime.tape import TapeRecord, TapeRef
    from sagent.runtime.types import Job, ModelContextEvent
    from sagent.session.types import SessionRecord


__all__ = [
    "Style",
    "SummaryCompactor",
    "halve",
    "summary_tokens",
    "text",
]


type Style = Literal[
    "summary",
    "session-audit",
]


logger = logging.getLogger(__name__)


def summary_tokens(
    body: float,
    *,
    max_share: float = 0.10,
    min_share: float = 0.05,
    power: float = 0.25,
    max_context: float = 1e6,
) -> float:
    """Return the room a summary of a ``body``-token context is assumed to need.

    The calculation is::

        S(B) = B · m·(C / clip(B, B₀, C))^p,  B₀ = C·(m / M)^(1/p)

    Clipping caps the share at ``M`` below the knee ``B₀``, floors it at ``m``
    past ``C``, and keeps the base positive. With ``p ≤ 1``, ``S`` never falls
    as ``B`` grows, as :meth:`SummaryCompactor.trigger_tokens` needs.

    Args:
      body: ``B``, the tokens summarized.
      max_share: ``M``, the share's cap.
      min_share: ``m``, the share at ``C`` and its floor past it.
      power: ``p``, how fast the share falls between ``B₀`` and ``C``.
      max_context: ``C``, the context size where the share reaches ``m``.

    Returns:
      tokens: ``S(B)``.

    Raises:
      ValueError: ``0 < m ≤ M`` or ``0 < p ≤ 1`` fails, NaN included.

    """
    if math.isnan(min_share) or min_share <= 0.0:
        raise ValueError(f"min_share must be > 0, got {min_share}.")
    if math.isnan(max_share) or max_share < min_share:
        raise ValueError(f"max_share must be >= min_share, got {max_share}.")
    if math.isnan(power) or power <= 0.0 or power > 1.0:
        raise ValueError(f"power must be in (0, 1], got {power}.")
    # Typeshed types ``float ** float`` as Any; both bases are positive above.
    knee = max_context * cast(float, (min_share / max_share) ** (1 / power))
    clip = min(max_context, max(knee, body))
    return body * min_share * cast(float, (max_context / clip) ** power)


class SummaryCompactor:
    """Starts its agent's session over from a summary of it.

    Args:
      agent: Whose session it compacts.
      prompt: The summary prompt; ``None`` uses ``assets/compactor.md``.
      partial_prompt: The prompt when a tail is kept; ``None`` uses
        ``assets/compactor_partial.md``.
      summary_tokens: ``S(B)``, the room held for the summary of a ``B``-token
        body; see :func:`summary_tokens`. ``B + S(B)`` must never fall as
        ``B`` grows: :meth:`trigger_tokens` bisects it.
      max_attempts: Attempts per selected prefix; an overflow shortens its suffix.
      keep_recent: Entries kept verbatim beside the summary.
      direction: ``"from"`` keeps the tail after the summary; ``"up_to"``
        keeps the head before it.
      proactive: Whether the continuation tells the model to carry on alone.
      verify_summary: Whether a second call checks the summary and fills gaps.
      model: What writes the summary; ``None`` uses the agent's model.
      effort: Thinking effort for the summary; ``None`` keeps the model's.
      style: ``summary`` condenses the context; ``session-audit`` audits the
        whole session from its journal. Assign ``compactor.style`` to switch.

    Raises:
      ValueError: ``max_attempts`` is below 1.

    """

    def __init__(
        self,
        agent: Agent,
        *,
        prompt: str | None = None,
        partial_prompt: str | None = None,
        summary_tokens: Callable[[float], float] = summary_tokens,
        max_attempts: int = 3,
        keep_recent: int = 0,
        direction: Literal["from", "up_to"] = "from",
        proactive: bool = False,
        verify_summary: bool = False,
        model: Model | None = None,
        effort: ThinkingEffort | None = None,
        style: Style = "summary",
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}.")
        self._agent = agent
        self._prompt = prompt
        self._partial_prompt = partial_prompt
        self._summary_tokens = summary_tokens
        self._max_attempts = max_attempts
        self._keep_recent = keep_recent
        self._direction: Literal["from", "up_to"] = direction
        self._proactive = proactive
        self._verify_summary = verify_summary
        self._model = model
        self._effort: ThinkingEffort | None = effort
        self.style: Style = style

    def should_run(self) -> bool:
        """Return whether the next request nears its token or its byte ceiling.

        Its tokens are the last reply's billed request, less the system prompt,
        plus an estimate of what was added since, against
        :meth:`trigger_tokens`. Its bytes are its text, plus its attachments as
        base64 sends them, against ``byte_share`` of the vendor's
        ``max_request_bytes``: images fill that long before they fill a window.

        Returns:
          should: Whether to compact before the next model call.

        """
        return self._held() >= self.trigger_tokens() or self._wire_bytes_due()

    def _wire_bytes_due(self, *, byte_share: float = 0.8) -> bool:
        """Return whether the next request holds ``byte_share`` of the vendor's ceiling."""
        ceiling = self._agent.model.capability.limits.max_request_bytes
        if ceiling <= 0:
            return False
        runtime = self._agent.runtime
        entries = runtime.context().messages
        size = sum(map(len, runtime.system)) + sum(len(text(e)) for e in entries)
        size += sum(
            len(attachment.data) * 4 / 3
            for entry in entries
            if isinstance(entry, (UserMessage, AgentToAgentMessage, ToolResult))
            for attachment in entry.attachments
        )
        return size >= byte_share * ceiling

    def reply_room(self) -> int:
        """Return the most the next reply may hold, as the window leaves room.

        A vendor that stops a reply at the window needs no cap; one that may
        reject a request gets the remainder after a 5% estimation reserve.

        Returns:
          tokens: The reserved window less the request, at least one; ``0`` uncapped.

        """
        return _room(self._agent.model, self._held() + self._system_tokens())

    def _held(self) -> int:
        """Return the next request's size past its system prompt: billed, plus since."""
        runtime = self._agent.runtime
        if runtime.token_accounting.has_measurements:
            request = runtime.request(max_response_tokens=0)
            return max(
                0,
                self._request_size(self._agent.model, request) - self._system_tokens(),
            )
        resolved = runtime.context()
        spliced = {r.ref for r in runtime.tape if isinstance(r, ContextSplice)}
        return _request_tokens(
            self._agent.model,
            resolved.messages,
            system=self._system_tokens(),
            carried=frozenset(
                i for i, origin in enumerate(resolved.origins) if origin in spliced
            ),
            unbilled=self._tool_tokens(),
        )

    def trigger_tokens(self) -> float:
        """Return the request size, past the system prompt, at which compaction is due.

        The largest body whose summary call still fits: the body, the system
        prompt, and the summary prompt within the input limit, and with the
        summary's room, :func:`summary_tokens`, within the window input and
        output share::

            B* = max{B ∈ ℕ : B + s + p ≤ R,  B + s + p + S(B) ≤ C}

        Returns:
          tokens: ``B*``; infinite for a model with no window.

        """
        return self._room(self._system_tokens() + self._prompt_tokens())

    # Infinite for a model with no window.
    def _room(self, fixed: float) -> float:
        """Return the largest body a summary call holding ``fixed`` more tokens fits."""
        model = self._agent.model
        limits = model.capability.limits
        if limits.max_request_tokens <= 0:
            return math.inf
        size = model.settings.context_size.tokens
        request = min(limits.max_request_tokens, size) - fixed
        # A shared window's room past the input limit holds output alone, whatever the size.
        headroom = limits.context_tokens - limits.max_request_tokens
        context = size + headroom - fixed
        # B + S(B) rises with B, so the bodies that fit are a prefix: count them.
        return float(
            bisect_right(
                range(1, math.floor(min(request, context)) + 1),
                context,
                key=lambda b: b + self._summary_tokens(b),
            ),
        )

    def _system_tokens(self) -> int:
        """Return the system prompt's size in tokens."""
        system = [SystemMessage(content=s) for s in self._agent.runtime.system]
        return _tokens(self._agent.model, system)

    def _tool_tokens(self) -> int:
        """Return the size in tokens of the tools every request offers."""
        chars = sum(
            len(f"{tool.name}{tool.description}{json_dumps(mutable(tool.schema))}")
            for tool in self._agent.runtime.tools.values()
        )
        return int(chars / self._agent.model.capability.approx_chars_per_token)

    def _prompt_tokens(self) -> int:
        """Return the summary prompt's size in tokens."""
        body = self._body(to_keep=self._keep_recent > 0, direction=self._direction)
        return _tokens(self._agent.model, [UserMessage(content=_summary_prompt(body))])

    async def run(self, instructions: str = "") -> None:
        """Compact the session at the runtime's next safe point, and wait for it.

        Args:
          instructions: What the summary must keep.

        Raises:
          Exception: What the compaction raised, as ``CompactFailed`` carried it.

        """
        done = asyncio.Event()
        failed: list[BaseException] = []

        def observe(event: types.RuntimeEvent) -> None:
            if isinstance(event, types.CompactComplete):
                done.set()
            elif isinstance(event, types.CompactFailed):
                failed.append(event.exception)
                done.set()

        runtime = self._agent.runtime
        runtime.observers.append(observe)
        try:
            runtime.push(types.Compact(args=instructions))
            await done.wait()
        finally:
            runtime.observers.remove(observe)
        if failed:
            raise failed[0]

    async def compact(
        self,
        tape: Sequence[TapeRecord],
        context: Sequence[ModelContextEvent],
        model: Model,
        mint_ref: Callable[[], TapeRef],
        custom_instructions: str | None = None,
    ) -> ContextSplice:
        """Return the splice that starts the session over from a summary.

        Args:
          tape: The runtime's tape; the splice masks all of it.
          context: What the model sees now.
          model: The runtime's model, used when the compactor has none.
          mint_ref: Mints the splice's ref.
          custom_instructions: What the summary must keep.

        Returns:
          splice: The summary, the kept tail, and the jobs still running.

        """
        summarizer = self._model if self._model is not None else model
        instructions = custom_instructions or ""
        with _at_effort(summarizer, self._effort):
            if self.style == "session-audit":
                payload, strategy, reason, kept = await self._audit(
                    summarizer,
                    list(context),
                    instructions,
                )
            else:
                payload, strategy, reason, kept = await self._summarize(
                    summarizer,
                    list(context),
                    instructions,
                    keep=self._tail_len(context),
                )
        return ContextSplice(
            ref=mint_ref(),
            mask=full_tape_mask(tape),
            insert_after=None,
            payload=payload,
            strategy=strategy,
            token_before=_request_tokens(summarizer, context),
            token_after=_tokens(summarizer, payload),
            fallback_reason=reason,
            preserved_tail_count=kept,
            paired_externally=unpaired_call_ids(payload),
        )

    def _tail_len(self, history: Sequence[ModelContextEvent]) -> int:
        """Return the entries kept: ``keep_recent``, or the user's unanswered turn."""
        if self._direction == "from":
            return max(self._keep_recent, _trailing_user_tail_len(history))
        return self._keep_recent

    async def _summarize(
        self,
        model: Model,
        history: list[ModelContextEvent],
        instructions: str,
        *,
        keep: int,
    ) -> tuple[tuple[ModelContextEvent, ...], str, str, int]:
        """Return the fitted replacement, its strategy, any truncation note, and kept count."""
        direction = self._direction
        to_keep: list[ModelContextEvent] = []
        pending: list[ModelContextEvent] = []
        to_summarize = history
        if direction == "up_to":
            unanswered = _trailing_user_tail_len(history)
            if unanswered:
                to_summarize, pending = _kept_tail(history, unanswered)
        if keep > 0:
            if direction == "from":
                to_summarize, to_keep = _kept_tail(history, keep)
            else:
                to_keep, to_summarize = _kept_head(to_summarize, keep)
            if not to_keep:
                logger.warning(
                    "compaction keep_recent=%d kept no tail: an unanswered tool"
                    " call spans the boundary; summarizing everything.",
                    keep,
                )
        body = self._body(to_keep=bool(to_keep), direction=direction)
        if guidance := instructions.strip():
            body = _append_user_guidance(body, guidance)
        if not to_summarize:
            payload = self._enrich(
                coalesce_roles([*to_keep, *pending]),
                history=history,
            )
            if not self._fits_payload(payload, [*to_keep, *pending]):
                raise ContextOverflowError("no nonempty prefix fits a summary call")
            return (
                payload,
                "summary_fallback",
                "nothing to summarize",
                len(to_keep) + len(pending),
            )

        payload, reason = await self._prefixes(
            model,
            to_summarize,
            instruction=_summary_prompt(body),
            render=partial(
                self._render_summary,
                direction=direction,
                to_keep=to_keep,
                pending=pending,
                history=history,
            ),
            fixed=to_keep if direction == "up_to" else (),
            pending=to_keep if direction == "from" else pending,
            force=True,
        )
        kept = len(to_keep) + len(pending)
        logger.info("compacted %d entries, kept %d", len(to_summarize), kept)
        return payload, "summary", reason, kept

    def _render_summary(
        self,
        summaries: list[UserMessage],
        remaining: list[ModelContextEvent],
        *,
        direction: Literal["from", "up_to"],
        to_keep: list[ModelContextEvent],
        pending: list[ModelContextEvent],
        history: list[ModelContextEvent],
    ) -> tuple[ModelContextEvent, ...]:
        summary = "\n\n".join(s.content or "" for s in summaries)
        payload = self._resumed(
            summary,
            direction=direction,
            to_keep=[*remaining, *to_keep] if direction == "from" else to_keep,
        )
        if direction == "up_to":
            payload = coalesce_roles([*payload, *remaining, *pending])
        return self._enrich(payload, history=history)

    def _resumed(
        self,
        summary: str,
        *,
        direction: Literal["from", "up_to"],
        to_keep: list[ModelContextEvent],
    ) -> tuple[ModelContextEvent, ...]:
        """Return the new window: ``summary``, told how to resume, beside the kept entries."""
        continuation = _continuation(
            summary,
            recent_preserved=bool(to_keep),
            proactive=self._proactive,
            direction=direction,
        )
        return _beside(continuation, direction=direction, to_keep=to_keep)

    # The journal holds every window, so the audit reads what earlier compactions
    # summarized away; their summaries, and each window's restated start, are left out.
    async def _audit(
        self,
        model: Model,
        history: list[ModelContextEvent],
        instructions: str,
    ) -> tuple[tuple[ModelContextEvent, ...], str, str, int]:
        """Return the new window from an audit of the whole session, and how it went."""
        # A long session's journal takes seconds to read: a worker thread reads it.
        lines = await asyncio.to_thread(_extract, self._agent.session.history())
        _, to_keep = _kept_tail(history, _trailing_user_tail_len(history))
        body = _asset("compactor_session_audit")
        if guidance := instructions.strip():
            body = _append_user_guidance(body, guidance)
        response = await self._audited(model, lines, body)
        summary = _summary(response) if response is not None else None
        if summary is None:
            raise ContextOverflowError("the session audit failed")
        payload, reason = await self._prefixes(
            model,
            [UserMessage(content=summary)],
            instruction=_summary_prompt(self._body(to_keep=False, direction="from")),
            render=partial(self._render_audit, to_keep=to_keep, history=history),
            pending=to_keep,
        )
        logger.info("audited %d session lines, kept %d", len(lines), len(to_keep))
        return payload, "session_audit", reason, len(to_keep)

    def _render_audit(
        self,
        summaries: list[UserMessage],
        remaining: list[ModelContextEvent],
        *,
        to_keep: list[ModelContextEvent],
        history: list[ModelContextEvent],
    ) -> tuple[ModelContextEvent, ...]:
        content = "\n\n".join(text(entry) for entry in [*summaries, *remaining])
        payload = self._resumed(content, direction="from", to_keep=to_keep)
        return self._enrich(payload, history=history)

    # An audit too long for one request has its older half summarized first, as a
    # compaction would, until the rest fits; each round must shrink it, or it fails.
    async def _audited(
        self,
        model: Model,
        lines: list[str],
        body: str,
    ) -> Response | None:
        """Ask for the audit of ``lines``; ``None`` when it cannot be had."""
        per = model.capability.approx_chars_per_token
        budget = self._room((len(_asset("compactor_system")) + len(body)) / per)
        try:
            while k := halve([len(line) / per for line in lines], budget=budget):
                earlier = await self._digest(model, lines[:k])
                shorter = (
                    [f"[earlier lines, summarized] {earlier}", *lines[k:]]
                    if earlier is not None
                    else lines
                )
                if sum(map(len, shorter)) >= sum(map(len, lines)):
                    logger.warning("the session audit could not be cut to fit")
                    return None
                lines = shorter
            joined = "\n".join(lines)
            return await self._ask(
                model,
                [UserMessage(content=f"{body}\n\n<session>\n{joined}\n</session>")],
            )
        except (ContextOverflowError, RetryableError) as exc:
            logger.warning("the session audit request failed: %s", exc)
            return None

    async def _digest(self, model: Model, lines: list[str]) -> str | None:
        """Return a summary of ``lines``, as a compaction writes one; ``None`` if none."""
        joined = "\n".join(lines)
        prompt = _summary_prompt(_asset("compactor"))
        response = await self._ask(
            model,
            [UserMessage(content=f"{joined}\n\n{prompt}")],
        )
        return _summary(response)

    async def _prefixes(
        self,
        model: Model,
        history: list[ModelContextEvent],
        *,
        instruction: str,
        render: Callable[
            [list[UserMessage], list[ModelContextEvent]],
            tuple[ModelContextEvent, ...],
        ],
        fixed: Sequence[ModelContextEvent] = (),
        pending: Sequence[ModelContextEvent] = (),
        force: bool = False,
    ) -> tuple[tuple[ModelContextEvent, ...], str]:
        """Consume fitting prefixes; revisit summaries only after each pass ends."""
        summaries: list[UserMessage] = []
        remaining: list[ModelContextEvent] = list(coalesce_roles(history))
        pass_size = self._size(model, remaining)
        reason = ""
        while force or not self._fits_payload(render(summaries, remaining), pending):
            if not remaining:
                size = self._size(model, summaries)
                if not summaries or size >= pass_size:
                    raise ContextOverflowError(
                        "the compaction pass did not shrink the context",
                    )
                remaining = list(summaries)
                summaries = []
                pass_size = size
            prefix = [*fixed, *summaries]
            response, summary, taken = await self._prefix_summary(
                model,
                remaining,
                instruction=instruction,
                retained=prefix,
            )
            finished = _finished(response)
            if self._verify_summary:
                try:
                    summary, finished = await self._verify(
                        model,
                        [*prefix, *remaining[:taken]],
                        _reply(response),
                        summary=summary,
                        finished=finished,
                        retained=prefix,
                    )
                except Exception as exc:  # noqa: BLE001 -- The usable first summary stands if verification fails.
                    logger.warning(
                        "summary verification failed; using the original: %s",
                        exc,
                    )
            summaries.append(UserMessage(content=summary))
            remaining = remaining[taken:]
            if not finished:
                reason = "summary cut off"
            force = False
        return render(summaries, remaining), reason

    async def _prefix_summary(
        self,
        model: Model,
        remaining: list[ModelContextEvent],
        *,
        instruction: str,
        retained: Sequence[ModelContextEvent],
    ) -> tuple[Response, str, int]:
        """Send the largest fitting safe prefix, returning only a validated summary."""
        safe = safe_boundaries(remaining)
        stops = [i for i in range(1, len(remaining) + 1) if safe[i]]
        for attempt in range(self._max_attempts):
            while True:
                if not stops:
                    raise ContextOverflowError("no nonempty prefix fits a summary call")
                taken = stops[-1]
                request = self._prefix_request(
                    model,
                    [*retained, *remaining[:taken]],
                    instruction,
                    retained=retained,
                )
                if self._fits_request(model, request):
                    break
                stops.pop()
            try:
                response = await self._stream(model, request)
            except ContextOverflowError:
                stops.pop()
                continue
            except RetryableError as exc:
                logger.warning(
                    "summary call failed (%s); waiting %ss",
                    exc,
                    exc.retry_after_sec,
                )
                if attempt + 1 < self._max_attempts:
                    await asyncio.sleep(exc.retry_after_sec)
                continue
            summary = _summary(response)
            if summary is not None:
                return response, summary, taken
        raise ContextOverflowError(
            f"summary failed after {self._max_attempts} attempts",
        )

    def _prefix_request(
        self,
        model: Model,
        entries: Sequence[ModelContextEvent],
        instruction: str,
        *,
        retained: Sequence[ModelContextEvent] = (),
    ) -> Request:
        """Keep the original frame and entries; distinguish retained context from input."""
        if retained:
            count = sum(
                len(entry.records) if isinstance(entry, Response) else 1
                for entry in retained
            )
            instruction += (
                f"\n\nThe first {count} entries are already retained."
                " Use them only as context; summarize only the newly selected"
                " entries after them. Do not repeat the retained summaries."
            )
        request = replace(
            self._agent.runtime.request(max_response_tokens=0),
            records=[
                record
                for entry in entries
                for record in (
                    entry.records if isinstance(entry, Response) else (entry,)
                )
            ]
            + [UserMessage(content=instruction)],
            purpose="compaction",
        )
        return replace(
            request,
            max_response_tokens=_room(model, self._request_size(model, request)),
        )

    def _request_size(self, model: Model, request: Request) -> int:
        """Measure unchanged blocks when available, otherwise estimate their size."""
        return self._agent.runtime.token_accounting.estimate_request(
            request,
            model=model,
        )

    def _size(self, model: Model, entries: Sequence[ModelContextEvent]) -> int:
        """Return a block's recorded size, or the best estimate available."""
        return self._agent.runtime.token_accounting.estimate_block(entries, model=model)

    def _fits_request(self, model: Model, request: Request) -> bool:
        """Reserve room for both a summary request and its predicted answer."""
        limits = model.capability.limits
        if limits.max_request_tokens <= 0:
            return True
        held = self._request_size(model, request)
        size = model.settings.context_size.tokens
        return (
            held <= min(size, limits.max_request_tokens)
            and held + self._summary_tokens(held)
            <= size + limits.context_tokens - limits.max_request_tokens
        )

    def _fits_payload(
        self,
        payload: Sequence[ModelContextEvent],
        pending: Sequence[ModelContextEvent],
    ) -> bool:
        """Leave room for the pending input or the next compaction, whichever is larger."""
        model = self._agent.model
        limits = model.capability.limits
        if limits.max_request_tokens <= 0:
            return True
        request = replace(
            self._agent.runtime.request(max_response_tokens=0),
            records=[
                record
                for entry in payload
                for record in (
                    entry.records if isinstance(entry, Response) else (entry,)
                )
            ],
        )
        held = self._request_size(model, request)
        body = max(0, held - self._size(model, pending))
        prompt = self._prompt_tokens()
        size = model.settings.context_size.tokens
        return (
            max(held, body + prompt) <= min(size, limits.max_request_tokens)
            and max(held, body + prompt + self._summary_tokens(body))
            <= size + limits.context_tokens - limits.max_request_tokens
        )

    def _body(self, *, to_keep: bool, direction: Literal["from", "up_to"]) -> str:
        """Return the summary prompt for a full or a partial compaction."""
        if not to_keep:
            return self._prompt if self._prompt is not None else _asset("compactor")
        body = (
            self._partial_prompt
            if self._partial_prompt is not None
            else _asset("compactor_partial")
        )
        if direction == "up_to":
            body = _filled("compactor_partial_up_to", body=body.rstrip())
        return body

    # Re-runs the model on the same entries plus its summary, asking it to fill gaps.
    # A replacement is taken only when it still meets the contract the original met:
    # verification may not turn a success into a failure.
    async def _verify(
        self,
        model: Model,
        entries: list[ModelContextEvent],
        raw: str,
        *,
        summary: str,
        finished: bool,
        retained: Sequence[ModelContextEvent] = (),
    ) -> tuple[str, bool]:
        """Ask the model to check the complete selected prefix without dropping input."""
        response = await self._stream(
            model,
            self._prefix_request(
                model,
                entries,
                _filled("compactor_verify", summary=raw),
                retained=retained,
            ),
        )
        improved = _reply(response).strip()
        if not improved or improved.upper() == "IDENTICAL":
            return summary, finished
        improved_finished = _finished(response)
        formatted = _summary(response)
        if formatted is None:
            logger.warning("verifier returned no usable summary; keeping the original")
            return summary, finished
        if finished and not improved_finished:
            logger.warning("verifier's summary was cut off; keeping the original")
            return summary, finished
        return formatted, improved_finished

    async def _ask(
        self,
        model: Model,
        records: Sequence[ModelContextEvent],
    ) -> Response:
        """Ask ``model`` with no tools, its reply held to the room left; return it."""
        system = _asset("compactor_system").strip()
        request = Request(
            system=(system,),
            records=[
                r
                for entry in records
                for r in (entry.records if isinstance(entry, Response) else (entry,))
            ],
            max_response_tokens=_room(
                model,
                _tokens(model, [SystemMessage(content=system), *records]),
            ),
            purpose="compaction",
        )
        return await self._stream(model, request)

    # Every call a compaction makes is the agent's, whichever model answers it.
    async def _stream(self, model: Model, request: Request) -> Response:
        """Ask ``model`` ``request``; return its reply, its cost spent by the agent."""
        runtime = self._agent.runtime
        request = runtime.sending(request)
        snapshot = runtime.token_accounting.capture(request, model=model)
        reply = None
        async for event in model.stream(request):
            if isinstance(event, Response):
                reply = event
        if reply is None:
            raise RuntimeError("The model ended its stream without a reply.")
        measurement = runtime.token_accounting.observe(snapshot, tokens=reply.tokens)
        self._agent.charge(
            reply.spend.total,
            by=runtime.tool_context.label,
            measurement=measurement,
        )
        return replace(reply, measurement=measurement)

    def _enrich(
        self,
        payload: tuple[ModelContextEvent, ...],
        *,
        history: Sequence[ModelContextEvent],
    ) -> tuple[ModelContextEvent, ...]:
        """Return ``payload`` with the skills in use restored and the running jobs named."""
        entries = list(payload)
        skills, restored = self._skills_note(history)
        if skills:
            _append_to_first_user(entries, skills, skills=restored)
        jobs = _jobs_note(self._agent.runtime.jobs)
        if jobs:
            _append_to_first_user(entries, jobs)
        return coalesce_roles(entries)

    # A summary keeps what a skill said to do, not its steps; the steps come back
    # whole, newest first, while they fit a quarter of what the window may hold.
    # Returns ``("", [])`` for none.
    def _skills_note(
        self,
        history: Sequence[ModelContextEvent],
        *,
        share: float = 0.25,
    ) -> tuple[str, list[str]]:
        """Return the skills ``history`` loaded, their bodies read again, and their names."""
        tool = self._agent.tools.get("Skill")
        names = _skills_loaded(history)
        if not isinstance(tool, Skill) or not names:
            return "", []
        found = {
            s.name: s for s in tool.discover(self._agent.runtime.tool_context.origin)
        }
        room = share * self.trigger_tokens()
        blocks: list[str] = []
        restored: list[str] = []
        skipped: list[str] = []
        for name in names:
            info = found.get(name)
            if info is None:
                continue
            block = info.block()
            size = _tokens(self._agent.model, [UserMessage(content=block)])
            if size > room:
                skipped.append(name)
                continue
            room -= size
            blocks.append(block)
            restored.append(name)
        lines = (
            ["Skills loaded before compaction, restored:", *blocks] if blocks else []
        )
        if skipped:
            lines.append(
                "Skills loaded before compaction but too long to restore;"
                f" re-load with Skill: {', '.join(skipped)}.",
            )
        return "\n\n".join(lines), restored


def halve(sizes: Sequence[float], *, budget: float) -> int:
    """Return how many leading lines of an audit to summarize first; none when it fits.

    ::

        sᵢ                              ∈ ℝ₊    # tokens of extract line i, i ∈ 1..m
        B                               ∈ ℝ     # tokens the audit request may hold
        E  = Σᵢ sᵢ                      ∈ ℝ₊    # tokens of the extract
        k  = 0                          if E ≤ B or m = 0
           = min{j : Σ_{i≤j} sᵢ ≥ E/2}  else    # the older half, by tokens

    Args:
      sizes: ``s``, one per extract line, oldest first.
      budget: ``B``.

    Returns:
      k: ``k``; ``0`` when the extract fits.

    """
    total = sum(sizes)
    if total <= budget or not sizes:
        return 0
    return next(i + 1 for i, run in enumerate(accumulate(sizes)) if run >= total / 2)


def text(entry: ModelContextEvent) -> str:
    """Return the characters of one history entry the model reads.

    Args:
      entry: The entry.

    Returns:
      text: Its words, a call's arguments as JSON, a result's text.

    """
    if isinstance(entry, (ToolResult, TurnContext)):
        return entry.text()
    if isinstance(entry, Response):
        return "".join(
            json_dumps(mutable(r.arguments))
            if isinstance(r, ToolCall)
            else r.content or ""
            for r in entry.records
        )
    return entry.content or ""


def _trailing_user_tail_len(history: Sequence[ModelContextEvent]) -> int:
    """Count the user messages, and the notes among them, ending ``history``."""
    count = 0
    for entry in reversed(history):
        if not isinstance(
            entry,
            (UserMessage, AgentToAgentMessage, SystemMessage, TurnContext),
        ):
            break
        count += 1
    return count


# A cut that would leave a call apart from its result moves left until none does.
# A cut moved back to the start keeps nothing, unless ``keep`` asked for it all.
def _kept_tail(
    history: list[ModelContextEvent],
    keep: int,
) -> tuple[list[ModelContextEvent], list[ModelContextEvent]]:
    """Split ``history`` into what to summarize and its last ``keep`` entries."""
    cut = _safe_cut(history, max(0, len(history) - keep))
    if cut == 0 and keep < len(history):
        return history, []
    return history[:cut], history[cut:]


def _kept_head(
    history: list[ModelContextEvent],
    keep: int,
) -> tuple[list[ModelContextEvent], list[ModelContextEvent]]:
    """Split ``history`` into its first ``keep`` entries and what to summarize."""
    cut = _safe_cut(history, min(keep, len(history)))
    return history[:cut], history[cut:]


def _safe_cut(history: list[ModelContextEvent], cut: int) -> int:
    """Return ``cut`` moved left until no call before it lacks its result."""
    # The boundary before the first entry is always safe, so the walk stops there.
    safe = safe_boundaries(history)
    while not safe[cut]:
        cut -= 1
    return cut


def _markers(content: str, attachments: Sequence[Attachment]) -> str:
    """Return ``content`` followed by one marker per attachment."""
    marks = [_marker(a.mime_descriptor) for a in attachments]
    return " ".join([content, *marks] if content else marks)


def _marker(mime: str) -> str:
    """Return the marker for an attachment of ``mime``."""
    if mime.startswith("application/pdf"):
        return "[document]"
    if mime.startswith("image/"):
        return "[image]"
    return "[attachment]"


def _beside(
    content: str,
    *,
    direction: Literal["from", "up_to"],
    to_keep: list[ModelContextEvent],
) -> tuple[ModelContextEvent, ...]:
    """Return ``content`` as a user message before a kept tail, or after a kept head."""
    message = UserMessage(content=content)
    return coalesce_roles(
        (message, *to_keep) if direction == "from" else (*to_keep, message),
    )


def _append_user_guidance(body: str, guidance: str) -> str:
    """Return ``body`` with the user's emphasis fenced as data."""
    return _filled(
        "compactor_user_guidance",
        body=body.rstrip(),
        guidance=escape(guidance),
    )


def _finished(response: Response) -> bool:
    """Return whether ``response`` ended on its own, not at its length limit."""
    return response.stop_reason != "max_tokens"


def _summary(response: Response) -> str | None:
    """Return a valid text-only summary, formatted; ``None`` if unusable."""
    if response.stop_reason == "refusal" or calls(response.records):
        return None
    return _format_summary(_reply(response), finished=_finished(response))


# A summary its limit cut off is kept, ending in a note that says so.
def _format_summary(raw: str, *, finished: bool) -> str | None:
    """Return the ``<summary>`` content, the ``<analysis>`` stripped."""
    match = _summary_match(raw)
    if match is None:
        return None
    body = match.group(1).strip()
    if not finished and not match.group(2):
        body += f"\n\n{_asset('compactor_cut_off')}"
    return re.sub(r"\n\n+", "\n\n", f"Summary:\n{body}").strip()


def _summary_match(raw: str) -> re.Match[str] | None:
    """Return the ``<summary>`` block of ``raw`` past its ``<analysis>``; ``None`` if empty."""
    text = re.sub(r"<analysis>[\s\S]*?</analysis>", "", raw)
    match = re.search(r"<summary>([\s\S]*?)(</summary>|\Z)", text)
    if match is None or not match.group(1).strip():
        logger.warning("compactor output has no usable <summary> (%d chars)", len(text))
        return None
    return match


@contextmanager
def _at_effort(model: Model, effort: ThinkingEffort | None) -> Generator[None]:
    """Run the body with ``model`` lowered to ``effort``, then put its own back."""
    before = model.settings
    ranked = cast(tuple[ThinkingEffort, ...], literal_members(ThinkingEffort))
    lowered = False
    if effort is not None and ranked.index(effort) < ranked.index(
        before.thinking_effort,
    ):
        try:
            model.settings = replace(before, thinking_effort=effort)
        except ValueError:
            logger.info(
                "the model does not offer effort %r; keeping %r",
                effort,
                before.thinking_effort,
            )
        else:
            lowered = True
    try:
        yield
    finally:
        if lowered:
            model.settings = replace(
                model.settings,
                thinking_effort=before.thinking_effort,
            )


# A load naming no skill restores nothing, so it is not counted at all.
def _skills_loaded(history: Sequence[ModelContextEvent]) -> list[str]:
    """Return the skills ``history`` loaded or restored, newest first, once each."""
    loads: dict[str, str] = {}
    seen: list[str] = []
    for entry in history:
        if isinstance(entry, Response):
            loads.update(
                (r.call_id, name)
                for r in calls(entry.records)
                if r.name == "Skill"
                and isinstance(name := r.arguments.get("skill"), str)
            )
        elif isinstance(entry, ToolResult) and entry.call_id in loads:
            if entry.is_error is not True and entry.stub is None:
                seen.append(loads[entry.call_id])
        elif isinstance(entry, (UserMessage, AgentToAgentMessage)):
            # Restored blocks are written newest first. The names ride with the
            # message, not its text, which a user could have pasted.
            restored = from_plain(entry.extra.get("skills"), list[str], default=[])
            seen.extend(reversed(restored))
    return list(dict.fromkeys(reversed(seen)))


def _jobs_note(jobs: Mapping[str, Job]) -> str:
    """Return the jobs still running, as a note; ``""`` for none."""
    now = time.time()
    lines = [
        f"- [{call_id}] {job.call.name}: {job.state} ({int(now - job.started)}s since started)"
        for call_id, job in jobs.items()
    ]
    if not lines:
        return ""
    return "Active background tasks (re-surfaced post-compaction):\n" + "\n".join(lines)


# ``skills`` names the skills ``text`` restores, kept on the message's ``extra``.
def _append_to_first_user(
    entries: list[ModelContextEvent],
    text: str,
    *,
    skills: Sequence[str] = (),
) -> None:
    """Append ``text`` to the first user message, or start the entries with one."""
    extra: dict[str, PlainTree] = {"skills": list(skills)} if skills else {}
    for i, entry in enumerate(entries):
        if isinstance(entry, (UserMessage, AgentToAgentMessage)):
            joined = f"{entry.content}\n\n{text}" if entry.content is not None else text
            entries[i] = replace(entry, content=joined, extra={**entry.extra, **extra})
            return
    entries.insert(0, UserMessage(content=text, extra=extra))


# Each line is what someone said or did: the user verbatim, the agent's replies, and a
# call's name and arguments. Results, reasoning, notes, and each window's start, with
# the summary and history it restates, say nothing new.
def _extract(records: Iterable[SessionRecord]) -> list[str]:
    """Return the session as numbered lines, one per message or tool call."""
    lines: list[str] = []
    for record in records:
        if isinstance(record, UserMessage):
            who, said = "USER: ", _markers(record.content or "", record.attachments)
        elif isinstance(record, AgentToAgentMessage):
            who, said = f"{record.sender} to {record.recipient}: ", record.content or ""
        elif isinstance(record, AssistantMessage):
            who, said = "AGENT: ", record.content or ""
        elif isinstance(record, ToolCall):
            args = json_dumps(mutable(record.arguments))[:200]
            who, said = "TOOL ", f"{record.name} {args}"
        else:
            continue
        if said:
            lines.append(f"L{len(lines) + 1} {who}{said}")
    return lines


def _asset(name: str) -> str:
    """Return the text of ``assets/<name>.md``."""
    asset = files(__package__) / "assets" / f"{name}.md"
    return asset.read_bytes().decode()


def _filled(name: str, **slots: str) -> str:
    """Return ``assets/<name>.md`` with each ``{{slot}}`` filled, and stripped."""
    # One pass: a value that quotes ``{{slot}}`` stays literal.
    return re.sub(r"\{\{(\w+)\}\}", lambda m: slots[m.group(1)], _asset(name)).strip()


def _summary_prompt(body: str) -> str:
    """Return the summary request's last message: ``body`` between the no-tools rules."""
    return _asset("compactor_no_tools") + body + _asset("compactor_no_tools_reminder")


def _reply(response: Response) -> str:
    """Return what the model said in ``response``."""
    return "".join(
        r.content or "" for r in response.records if isinstance(r, AssistantMessage)
    )


def _continuation(
    summary: str,
    *,
    recent_preserved: bool,
    proactive: bool,
    direction: Literal["from", "up_to"],
) -> str:
    """Return the message the new window starts with, from its asset template."""
    recent = (
        f"\n\n{_asset(f'compactor_recent_{direction}').strip()}"
        if recent_preserved
        else ""
    )
    resume = (
        f"\n\n{_asset('compactor_resume_proactive')}"
        if proactive
        else f"\n{_asset('compactor_resume')}"
    )
    return _filled(
        "compactor_continuation",
        summary=summary,
        recent=recent,
        resume=resume,
    )


# A vendor that stops a reply at the window needs no cap; one that may reject a request
# whose reply could pass it, or is not known not to, gets what is left, at least one.
def _room(model: Model, request: int) -> int:
    """Return a reply cap with 5% of the window held for estimation error; ``0`` uncapped."""
    limits = model.capability.limits
    if limits.truncates_at_context is True or limits.max_request_tokens <= 0:
        return 0
    return max(1, limits.context_tokens * 95 // 100 - request)


def _tokens(model: Model, entries: Sequence[ModelContextEvent]) -> int:
    """Return ``entries``' size in tokens: characters scaled, and each attachment's."""
    chars = sum(len(text(entry)) for entry in entries)
    return int(chars / model.capability.approx_chars_per_token) + sum(
        model.attachment_tokens(attachment)
        for entry in entries
        if isinstance(entry, (UserMessage, AgentToAgentMessage, ToolResult))
        for attachment in entry.attachments
    )


# A billed count holds the system prompt and the tools; an estimate of the context
# holds neither, so ``unbilled`` adds the tools to it. A reply a splice carried over
# was billed for the context before the splice, which a compaction summarized away,
# so it says nothing of the next request; nor does a resumed reply, billed nothing.
def _request_tokens(
    model: Model,
    context: Sequence[ModelContextEvent],
    *,
    system: int = 0,
    carried: frozenset[int] = frozenset(),
    unbilled: int = 0,
) -> int:
    """Return the next request's size past its system prompt of ``system`` tokens."""
    for i in range(len(context) - 1, -1, -1):
        entry = context[i]
        if isinstance(entry, Response) and entry.tokens.total > 0 and i not in carried:
            asked = max(0, entry.tokens.total - system)
            return asked + _tokens(model, context[i + 1 :])
    return _tokens(model, context) + unbilled
