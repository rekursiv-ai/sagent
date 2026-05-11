"""Cohort: bundles the tool tasks a single model response dispatches.

See ``docs/private/execution_model.md`` for the role this plays. A
Cohort owns N pre-spawned tool tasks (one per ``tool_use`` block in
the model response).

Members are added in one of two modes:

- **Foreground** (default): the cohort waits for the task to complete;
  the task's result joins the emission bundle.
- **Background** (``bg=True``): the cohort does not wait; the
  emission bundle contains a ``[Running in background: <tool>]``
  placeholder for this member. The caller owns the task's lifecycle
  (registers its own completion callback). Used by the agent for
  ``background:true`` directive tool calls so they round-trip through
  the same cohort bundle as fg calls without separate stitching.

Lifecycle entrypoints:

- **Natural emission.** Every foreground member's task completes;
  the cohort emits one consolidated ``tool_result`` bundle via the
  ``on_emit`` callback. Bg members contribute their placeholder.
- **``force_close``.** A user message arrived before the cohort
  finished. Completed fg members keep their real results; unfinished
  fg members get ``[Running in background: <tool>]`` placeholders and
  their tasks are handed to ``on_promote_to_bg`` so the agent can move
  them into ``self.background``. Bg members emit their placeholder
  (no double-promotion). Emission happens immediately.

Cohort does **not** create tool tasks. Callers spawn tasks externally
and pass them in via ``add_member``. This keeps Cohort independent of
tool dispatch details (file-op sequencing, bg task setup, etc.).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import asyncio
import dataclasses

from sagent.custom_types import (
    Message,
    MultipartMessage,
    TextMessage,
)
from sagent.lib.descriptors import MultipartDescriptor


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CohortMember:
    """One tool task tracked by a cohort.

    Attributes:
      tool_use_id: Queue-id of the originating ``tool_use`` block.
      tool_name: Human-readable tool name for placeholder text.
      task: The asyncio task running the tool. Must produce a
        ``multipart/x-tool-result`` Message on success.

    """

    tool_use_id: str
    tool_name: str
    task: asyncio.Task[Message]


OnEmit = Callable[[list[Message]], None]
OnPromoteToBg = Callable[[CohortMember], None]


class Cohort:
    """One round's tool batch; emits one consolidated tool_result message."""

    def __init__(
        self,
        *,
        on_emit: OnEmit,
        on_promote_to_bg: OnPromoteToBg,
    ) -> None:
        self._on_emit = on_emit
        self._on_promote_to_bg = on_promote_to_bg
        self._members: list[CohortMember] = []
        self._remaining: int = 0
        self._emitted: bool = False
        self._bg_member_ids: set[str] = set()

    @property
    def members(self) -> tuple[CohortMember, ...]:
        """Members added to this cohort (read-only view)."""
        return tuple(self._members)

    @property
    def emitted(self) -> bool:
        """True once this cohort has fired ``on_emit`` (terminal state)."""
        return self._emitted

    def add_member(self, member: CohortMember, *, bg: bool = False) -> None:
        """Track ``member``; register a done-callback for natural emission.

        When ``bg=True`` the cohort does not wait for the member, and the
        emission bundle contains a ``[Running in background: <tool>]``
        placeholder for it. The caller owns the task's lifecycle.
        """
        if self._emitted:
            raise RuntimeError("Cohort already emitted; cannot add member")
        self._members.append(member)
        if bg:
            self._bg_member_ids.add(member.tool_use_id)
            return
        self._remaining += 1
        member.task.add_done_callback(self._on_member_done)

    def force_close(self) -> None:
        """Emit immediately; promote unfinished fg members to background.

        Idempotent: a second ``force_close`` (or one after a natural
        emission) is a no-op. Bg members emit their placeholder; no
        double-promotion since the caller already owns their lifecycle.
        """
        if self._emitted:
            return
        results: list[Message] = []
        for m in self._members:
            if m.tool_use_id in self._bg_member_ids:
                results.append(self._build_running_in_bg_result(m))
            elif m.task.done():
                results.append(self._build_settled_result(m))
            else:
                results.append(self._build_running_in_bg_result(m))
                self._on_promote_to_bg(m)
        self._emitted = True
        self._on_emit(results)

    def _on_member_done(self, task: asyncio.Task[Message]) -> None:
        del task
        self._remaining -= 1
        if self._remaining == 0 and not self._emitted:
            self._emit_natural()

    def _emit_natural(self) -> None:
        results = [self._build_settled_result(m) for m in self._members]
        self._emitted = True
        self._on_emit(results)

    def _build_settled_result(self, m: CohortMember) -> Message:
        if m.tool_use_id in self._bg_member_ids:
            return self._build_running_in_bg_result(m)
        if m.task.cancelled():
            return _make_tool_result(
                m.tool_use_id,
                "[Cancelled by user]",
                is_error=True,
            )
        exc = m.task.exception()
        if exc is not None:
            return _make_tool_result(
                m.tool_use_id,
                f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
        return m.task.result()

    def _build_running_in_bg_result(self, m: CohortMember) -> Message:
        return _make_tool_result(
            m.tool_use_id,
            f"[Running in background: {m.tool_name}]",
            is_error=False,
        )


def _make_tool_result(
    use_id: str,
    content: str,
    *,
    is_error: bool,
) -> Message:
    return MultipartMessage(
        (
            TextMessage(use_id, "text/x-queue-id"),
            TextMessage(content, "text/x-error" if is_error else "text/plain"),
        ),
        cast(MultipartDescriptor, "multipart/x-tool-result"),
    )
