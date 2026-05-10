"""Compaction handlers: ``BudgetWatcher``, ``CompactHandler``, ``RecompactHandler``.

Compaction has three triggers, all routed to a shared ``_run_compaction``
helper that mutates ``agent.history`` exactly once:

- ``BudgetWatcher`` (spawned, fires on ``text/x-model-call``): pre-flight
  check; runs microcompact and (if needed) full compaction. Spawned
  rather than inline so a long compaction does not freeze the dispatch
  loop -- ``/abort`` and ``/break`` stay responsive while compaction is
  running. Acquires ``agent.model_lock`` first so ``ModelCallHandler``
  (also spawned on the same descriptor) waits behind it; CPython
  asyncio's FIFO task scheduling combined with the lock's FIFO wait
  queue gives ``BudgetWatcher`` deterministic priority since it
  registers first in :func:`core_handlers`.
- ``CompactHandler`` (inline, fires on ``text/x-compact-request``):
  user-or-tool-initiated compaction.
- ``RecompactHandler`` (inline, fires on ``text/x-recompact-request``):
  reload the most recent ``pre_compact_N.jsonl`` from disk and re-run
  compaction with custom instructions (or a different model, or whatever
  you want to retry against the un-compacted transcript).

Overflow recovery (running compaction when the provider rejects a
request as over-budget) is folded into ``ModelCallHandler`` itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import logging

from sagent.agent.compaction import post_compact_enrich
from sagent.agent.handlers.base import (
    InlineHandler,
    SpawnedHandler,
)
from sagent.agent.session_io import load_message
from sagent.custom_types import Message, TokenCount
from sagent.lib.compaction import write_pre_compact_transcript
from sagent.lib.descriptors import is_binary, is_multipart
from sagent.sessions import parse_jsonl


if TYPE_CHECKING:
    from pathlib import Path

    from sagent.agent.agent import Agent
    from sagent.custom_types import Model


logger = logging.getLogger(__name__)


MAX_COMPACT_FAILURES = 3


class BudgetWatcher(SpawnedHandler):
    """Pre-flight budget check before each ``text/x-model-call``.

    Runs ``compactor.maintain`` (microcompact stale file reads) and,
    if the post-microcompact estimate still exceeds budget, runs full
    compaction. Acquires ``agent.model_lock`` so ``ModelCallHandler``
    (also spawned on the same descriptor) waits until BudgetWatcher
    releases -- CPython asyncio's FIFO task scheduling and the lock's
    FIFO wait queue give us deterministic ordering as long as
    BudgetWatcher is registered first in :func:`core_handlers`.

    Spawned (not inline) so a long compaction does not freeze the
    dispatch loop -- ``/abort`` cancels ``agent.tasks``, which
    includes this handler's task; the cancel unwinds through
    :func:`run_compaction`'s ``finally`` so ``compacting`` resets
    cleanly.
    """

    descriptors: tuple[str, ...] = ("text/x-model-call",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        """Microcompact then maybe-compact based on token estimate."""
        del msg
        if agent.compactor is None:
            return
        async with agent.model_lock:
            agent.compactor.maintain(
                agent.history,
                agent.tools_map,
                read_cache=agent.tool_state.read_cache,
                last_response_time=agent.cost_tracker.last_response_time,
            )
            system = agent.system_prompt()
            input_tokens = max(
                agent.cost_tracker.last_request.input_tokens,
                estimate_total_tokens(system, agent.history, agent.model),
            )
            compacted = await _maybe_compact(agent, input_tokens)
            if not compacted and input_tokens > (
                agent.max_request_tokens
                - agent.max_response_tokens
                - agent.budget.buffer_tokens
            ):
                await _force_compact(agent)


class CompactHandler(InlineHandler):
    """Run compaction in response to ``text/x-compact-request``."""

    descriptors: tuple[str, ...] = ("text/x-compact-request",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        """Run compaction with the message content as custom instructions."""
        if agent.compactor is None:
            return
        instructions = str(msg.content) or None
        await run_compaction(
            agent,
            custom_instructions=instructions,
            count_toward_budget=False,
        )


class RecompactHandler(InlineHandler):
    """Reload the most recent pre-compact transcript and re-run compaction."""

    descriptors: tuple[str, ...] = ("text/x-recompact-request",)

    @override
    async def handle(self, agent: Agent, msg: Message) -> None:
        """Reload from disk, re-run compaction, install or roll back."""
        if agent.compactor is None or agent.session_dir is None:
            return
        # ``run_compaction`` writes ``pre_compact_{n}.jsonl`` then increments
        # ``compact_count``; the latest readable file is at ``count - 1``.
        # When count == 0 no compaction has run yet -- nothing to recompact.
        count = agent.compaction_state.compact_count
        if count == 0:
            return
        transcript = agent.session_dir / f"pre_compact_{count - 1}.jsonl"
        if not transcript.exists():
            return
        try:
            raw_text = transcript.read_text(encoding="utf-8")
            loaded = [load_message(rec) for rec in parse_jsonl(raw_text)]
        except (OSError, KeyError, AssertionError, TypeError) as e:
            logger.warning("Uncompact transcript load failed: %s", e)
            return
        if not loaded:
            return
        saved = list(agent.history)
        agent.history.clear()
        agent.history.extend(loaded)
        instructions = str(msg.content) or None
        success = await run_compaction(
            agent,
            custom_instructions=instructions,
            write_transcript=False,
            count_toward_budget=False,
        )
        if not success:
            agent.history.clear()
            agent.history.extend(saved)


async def _maybe_compact(agent: Agent, input_tokens: int) -> bool:
    """Threshold-based compaction. Returns True if compaction ran."""
    if agent.compactor is None:
        return False
    if agent.compaction_state.compacting:
        return False
    if agent.compaction_state.compact_failures >= MAX_COMPACT_FAILURES:
        return False
    should = await agent.compactor.should_compact(
        input_tokens=input_tokens,
        max_request_tokens=agent.max_request_tokens,
        max_response_tokens=agent.max_response_tokens,
    )
    if not should:
        return False
    return await run_compaction(agent)


async def _force_compact(agent: Agent) -> None:
    """Run compaction unconditionally; raise if compaction is disabled."""
    if agent.compactor is None:
        return
    if agent.compaction_state.compacting:
        return
    if agent.compaction_state.compact_failures >= MAX_COMPACT_FAILURES:
        msg = (
            f"Compaction disabled after {agent.compaction_state.compact_failures} "
            f"failures; cannot reduce conversation below context window."
        )
        logger.warning(msg)
        raise RuntimeError(msg)
    _ = await run_compaction(agent)


async def run_compaction(
    agent: Agent,
    *,
    keep_recent: int | None = None,
    custom_instructions: str | None = None,
    write_transcript: bool = True,
    count_toward_budget: bool = True,
) -> bool:
    """Run one compaction round; mutate ``agent.history`` on success.

    Args:
      agent: Agent whose history gets compacted.
      keep_recent: Override compactor's default ``keep_recent``.
      custom_instructions: Extra guidance for the compactor.
      write_transcript: True for autonomous compactions (write
          ``pre_compact_N.jsonl``); False for Recompact.
      count_toward_budget: True for autonomous compactions (a failure
          increments ``compact_failures``); False for manual triggers.

    Returns:
      success: True on success, False if the compactor raised.

    """
    assert agent.compactor is not None
    agent.compaction_state.compacting = True
    try:
        transcript_path: Path | None = None
        if agent.session_dir is not None:
            transcript_path = (
                agent.session_dir
                / f"pre_compact_{agent.compaction_state.compact_count}.jsonl"
            )
            if write_transcript:
                write_pre_compact_transcript(transcript_path, agent.history)
        prior_pointers = list(agent.compaction_state.summary_pointers)
        result = await agent.compactor.compact(
            messages=agent.history,
            model=agent.model,
            transcript_path=transcript_path,
            keep_recent=keep_recent,
            custom_instructions=custom_instructions,
            summary_pointers=prior_pointers or None,
        )
        agent.history.clear()
        agent.history.extend(result)
    except Exception as e:  # noqa: BLE001 -- compactor / model raises varied errors
        if count_toward_budget:
            agent.compaction_state.compact_failures += 1
        logger.warning(
            "Compaction failed (%d/%d): %s: %s",
            agent.compaction_state.compact_failures,
            MAX_COMPACT_FAILURES,
            type(e).__name__,
            e,
        )
        return False
    finally:
        agent.compaction_state.compacting = False

    system = agent.system_prompt()
    used = estimate_total_tokens(system, agent.history, agent.model)
    headroom = agent.max_response_tokens + agent.budget.buffer_tokens
    await post_compact_enrich(
        result=result,
        messages=agent.history,
        state=agent.compaction_state,
        session_dir=agent.session_dir,
        tool_state=agent.tool_state,
        budget=agent.budget,
        tools=agent.tools_map,
        background_tasks=agent.background_tasks,
        estimate_tokens=agent.max_request_tokens - used,
        headroom=headroom,
    )
    agent.compaction_state.compact_count += 1
    agent.cost_tracker.last_request = TokenCount()
    # History was wholesale-replaced; persist via clear barrier so the
    # on-disk live view matches memory (prior bytes preserved).
    agent.save_session(clear=True)
    return True


def estimate_total_tokens(
    system: str,
    messages: list[Message],
    model: Model,
) -> int:
    """Estimate total tokens for a (system, messages) pair."""
    total = model.estimate_text_token_count(system)
    for msg in messages:
        total += _estimate_message_tokens(msg, model)
    return total


def _estimate_message_tokens(msg: Message, model: Model) -> int:
    """Recursively estimate tokens for one message; handles multipart + binary."""
    if is_multipart(msg.descriptor):
        return sum(
            _estimate_message_tokens(p, model)
            for p in cast("tuple[Message, ...]", msg.content)
        )
    if is_binary(msg.descriptor):
        return model.estimate_image_token_count(cast("bytes", msg.content))
    return model.estimate_text_token_count(str(msg.content))
