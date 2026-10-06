"""State and lifecycle shared by CLI-subprocess providers.

``AnthropicCLI`` and ``GoogleCLI`` each drive a long-lived vendor CLI
subprocess (``claude`` / ``gemini``). They track the same facts about it:
how much of sagent's tape the subprocess has seen, which system prompt
it was spawned with, and how many turns and tokens it has served. They
apply the same rules to those facts: when to respawn, what counts as
input, and what an input-less turn returns. Those facts and rules live
here once, in :class:`CLISubprocessModel`, so a lifecycle fix to one
provider lands in both.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hashlib
import shutil
import tempfile

from sagent.types.model import ModelResponse
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    UserMessage,
)
from sagent.types.tape import coalesce_roles


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from sagent.providers.lib.subproc import Subproc
    from sagent.types.runtime import ModelContextEvent


# A subprocess is respawned after this many turns regardless of context size,
# bounding per-process state (KV cache, MCP handshake drift) that accretes
# across a long-lived ``--print`` session.
TURN_RESPAWN_THRESHOLD = 100  # house-ignore[globals] -- Respawn cadence threshold dial.

# ...or once the last request's input footprint crosses this fraction of the
# model's context window, so the next turn starts from a fresh process before
# the window fills.
CONTEXT_FRACTION_RESPAWN_THRESHOLD = (
    0.5  # house-ignore[globals] -- Context-fraction respawn threshold dial.
)


def respawn_for_cadence(
    *,
    turn_count: int,
    last_input_tokens: int,
    max_request_tokens: int,
) -> bool:
    """Return whether the subprocess should respawn for turn/context cadence.

    Args:
      turn_count: Turns served by the current subprocess.
      last_input_tokens: Cache-inclusive input footprint of the last request.
      max_request_tokens: The model's context window.

    Returns:
      respawn: True when the turn cap is hit or the context fraction is crossed.

    """
    if turn_count >= TURN_RESPAWN_THRESHOLD:
        return True
    return last_input_tokens > max_request_tokens * CONTEXT_FRACTION_RESPAWN_THRESHOLD


def hash_system(system: str | None) -> str:
    """Hash a system prompt for cheap equality checks across turns."""
    return hashlib.sha256((system or "").encode()).hexdigest()


def merge_user_input(
    entries: Sequence[ModelContextEvent],
) -> UserMessage | AgentSendMessage | None:
    """Fold the user-side entries of ``entries`` into one subprocess turn.

    The CLI answers each line it reads with a full model turn, so writing
    N pending entries one at a time costs N model turns, and a mid-sequence
    failure leaves the earlier ones answered and the later ones stranded.
    Assistant turns and tool results never cross stdin: the subprocess
    produced them itself, through its own tool loop.

    Args:
      entries: Tape entries the subprocess has not seen yet.

    Returns:
      entry: The merged user-side input, or ``None`` when there is none.

    """
    user_side = [e for e in entries if isinstance(e, (UserMessage, AgentSendMessage))]
    if not user_side:
        return None
    merged = coalesce_roles(user_side)[0]
    assert isinstance(merged, (UserMessage, AgentSendMessage))
    return merged


def empty_turn_response() -> ModelResponse:
    """Answer a turn that has no input without touching the subprocess.

    Writing nothing and then reading would block until the transport's idle
    timeout, because the CLI only answers a line it has been sent.

    Returns:
      response: An empty, finished assistant turn with zero usage.

    """
    return ModelResponse(
        message=AssistantMessage(text="", tool_calls=()),
        stop_reason="model_finished",
    )


def populated_tmpdir(prefix: str, populate: Callable[[Path], None]) -> Path:
    """Create a temp dir and fill it, deleting it if filling fails.

    The dir holds copied OAuth credentials; until a ``Subproc`` owns it, a
    failure here is the only point that can still delete it.

    Args:
      prefix: ``mkdtemp`` prefix.
      populate: Writes the dir's contents.

    Returns:
      tmpdir: The populated directory, owned by the caller.

    """
    tmpdir = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        populate(tmpdir)
    except BaseException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    return tmpdir


class CLISubprocessModel:
    """Delta and cadence state for a model driving a vendor CLI subprocess.

    Attributes:
      _last_sent_index: Count of tape entries the active subprocess has seen.
      _sent_history_head: First tape entry at the time of the first send. A
        different head means the tape was rewritten (e.g. compaction).
      _system_hash: Hash of the system prompt the active subprocess runs.
      _turn_count: Turns served by the active subprocess.
      _last_input_tokens: Context footprint of the last request.
      _warming_proc: Subprocess mid-spawn, closed if the warm-up is cancelled.

    """

    def __init__(self) -> None:
        self._last_sent_index = 0
        self._sent_history_head: ModelContextEvent | None = None
        self._system_hash = ""
        self._turn_count = 0
        self._last_input_tokens = 0
        self._warming_proc: Subproc | None = None

    def _respawn_due(
        self,
        history: Sequence[ModelContextEvent],
        *,
        system: str | None,
        max_request_tokens: int,
    ) -> bool:
        """Whether the active subprocess can no longer serve ``history``."""
        if not history or self._last_sent_index > len(history):
            return True
        if self._sent_history_head is not None and (
            history[0] is not self._sent_history_head
        ):
            return True
        if hash_system(system) != self._system_hash:
            return True
        return respawn_for_cadence(
            turn_count=self._turn_count,
            last_input_tokens=self._last_input_tokens,
            max_request_tokens=max_request_tokens,
        )

    def _unsent_input(
        self,
        history: Sequence[ModelContextEvent],
    ) -> UserMessage | AgentSendMessage | None:
        """Merge the user-side entries the active subprocess has not seen."""
        if self._last_sent_index == 0 and history:
            self._sent_history_head = history[0]
        return merge_user_input(history[self._last_sent_index :])

    def _reset_active_state(self) -> None:
        """Reset per-subprocess counters after a respawn boundary."""
        self._turn_count = 0
        self._last_input_tokens = 0
        self._reset_delta_state()

    def _reset_delta_state(self) -> None:
        """Forget what the (replaced) subprocess has seen."""
        self._last_sent_index = 0
        self._sent_history_head = None

    async def _close_warming_proc(self) -> None:
        """Close the subprocess currently being warmed, if any."""
        proc = self._warming_proc
        self._warming_proc = None
        if proc is not None:
            await proc.close()
