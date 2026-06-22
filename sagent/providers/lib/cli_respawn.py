"""Shared respawn cadence for CLI-subprocess providers.

Both ``AnthropicCLI`` and ``GoogleCLI`` drive a long-lived ``claude`` /
``gemini`` subprocess and periodically respawn it to bound accumulated
per-process state and context growth. The cadence thresholds and the
turn-count / context-fraction decision are identical across the two, so they
live here once rather than drifting as parallel per-provider copies.
"""

from __future__ import annotations


# A subprocess is respawned after this many turns regardless of context size,
# bounding per-process state (KV cache, MCP handshake drift) that accretes
# across a long-lived ``--print`` session.
TURN_RESPAWN_THRESHOLD = 100

# ...or once the last request's input footprint crosses this fraction of the
# model's context window, so the next turn starts from a fresh process before
# the window fills.
CONTEXT_FRACTION_RESPAWN_THRESHOLD = 0.5


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
