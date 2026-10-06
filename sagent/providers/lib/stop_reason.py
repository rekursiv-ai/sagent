"""Normalize provider-specific termination reasons to one vocabulary.

Each LLM backend reports response termination differently:

- Anthropic: ``end_turn`` / ``tool_use`` / ``pause_turn`` / ``stop_sequence``
  / ``max_tokens`` / ``refusal``.
- OpenAI (and OpenAI-compat: Kimi, Qwen, MiniMax, local llama.cpp):
  ``stop`` / ``length`` / ``tool_calls`` / ``function_call`` / ``content_filter``.
- Google Gemini: ``STOP`` / ``MAX_TOKENS`` / ``SAFETY`` / ``RECITATION`` /
  ``LANGUAGE`` / ``BLOCKLIST`` / ``PROHIBITED_CONTENT`` / ``SPII`` /
  ``IMAGE_SAFETY`` / ``MALFORMED_FUNCTION_CALL`` / ``OTHER`` /
  ``FINISH_REASON_UNSPECIFIED``.
- ACP (GoogleCLI's ``session/prompt``): ``end_turn`` / ``max_tokens`` /
  ``max_turn_requests`` / ``refusal`` / ``cancelled``.

The agent works in its own canonical vocabulary - providers translate at the
adapter boundary so the agent's ``stop_reason`` guard
(:data:`BENIGN_STOP_REASONS`) is uniform across backends.

Without this, two failure modes happened in the wild:

- OpenAI ``"stop"`` (its end_turn) didn't match our canonical
  ``"model_finished"`` and was raised as a non-benign termination.
- Google's ``MAX_TOKENS`` was thrown away by the adapter (which
  reported only ``"model_tool_use"`` / ``"model_finished"``), so a truncated
  Gemini tool call would be silently dispatched.
"""

from __future__ import annotations

from typing import Final, Literal


ProviderKind = Literal["anthropic", "openai", "google", "acp"]

# Normal terminations: the message is complete and safe to consume.
BENIGN_STOP_REASONS: frozenset[str] = frozenset(
    {
        "model_finished",  # Natural completion.
        "model_tool_use",  # Model emitted tool call(s)
        "stop_sequence",  # Caller-requested stop sequence matched.
        "model_continuing",  # Server-tool loop paused; content is valid.
    },
)

# The output hit a length ceiling. Prose is a usable partial answer; a tool
# call may carry cut-off arguments and must not be dispatched.
TRUNCATED_STOP_REASONS: frozenset[str] = frozenset(
    {"max_tokens", "model_context_window_exceeded"},
)

# The model produced no usable answer.
FAILED_STOP_REASONS: frozenset[str] = frozenset({"model_refusal", "model_cancelled"})

# The vendor ended the turn without a usable verdict: it could not parse the
# model's tool call, or it reported an unspecified/"other" stop (Gemini
# ``OTHER`` / ``FINISH_REASON_UNSPECIFIED``). Consuming such a turn as success
# keeps whatever fragment arrived; a resend usually succeeds.
RETRYABLE_STOP_REASONS: frozenset[str] = frozenset(
    {"model_malformed_tool_call", "model_unknown"},
)

_ANTHROPIC_MAP: Final[dict[str, str]] = {
    "end_turn": "model_finished",
    "pause_turn": "model_continuing",
    "tool_use": "model_tool_use",
    "refusal": "model_refusal",
    "stop_sequence": "stop_sequence",
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "model_context_window_exceeded",
}

# Agent Client Protocol ``session/prompt`` result ``stopReason``.
_ACP_MAP: Final[dict[str, str]] = {
    "end_turn": "model_finished",
    "max_tokens": "max_tokens",
    "max_turn_requests": "model_continuing",
    "refusal": "model_refusal",
    "cancelled": "model_cancelled",
}

# OpenAI ``finish_reason`` → canonical vocabulary.
_OPENAI_MAP: Final[dict[str, str]] = {
    "stop": "model_finished",
    "length": "max_tokens",
    "tool_calls": "model_tool_use",
    "function_call": "model_tool_use",  # Legacy.
    "content_filter": "model_refusal",
}

# Google ``finishReason`` (Gemini) → canonical.
_GOOGLE_MAP: Final[dict[str, str]] = {
    "STOP": "model_finished",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "model_refusal",
    "RECITATION": "model_refusal",
    "LANGUAGE": "model_refusal",
    "BLOCKLIST": "model_refusal",
    "PROHIBITED_CONTENT": "model_refusal",
    "SPII": "model_refusal",
    "IMAGE_SAFETY": "model_refusal",
    "MALFORMED_FUNCTION_CALL": "model_malformed_tool_call",
    "OTHER": "model_unknown",
    "FINISH_REASON_UNSPECIFIED": "model_unknown",
}


def normalize_stop_reason(
    raw: str | None,
    *,
    kind: ProviderKind,
    has_tool_use: bool,
) -> str:
    """Translate a provider's termination string to canonical vocab.

    ``has_tool_use`` upgrades a plain "stop" → "model_tool_use" when the
    response carried tool call(s) -- handles backends that don't
    distinguish (OpenAI sometimes reports ``"stop"`` even with tool
    calls when streaming; Gemini's ``STOP`` doesn't differentiate).

    Args:
      raw: Provider-native finish reason string, or ``None``.
      kind: Which provider vocabulary to translate from.
      has_tool_use: Whether the response contained tool call(s).

    Returns:
      reason: Canonical stop reason string.

    """
    if kind == "anthropic":
        translated = _ANTHROPIC_MAP.get(raw or "end_turn", raw or "model_finished")
    elif kind == "openai":
        translated = _OPENAI_MAP.get(raw or "", raw or "model_finished")
    elif kind == "acp":
        translated = _ACP_MAP.get(raw or "", raw or "model_finished")
    else:
        translated = _GOOGLE_MAP.get(raw or "", raw or "model_finished")
    # Upgrade model_finished → model_tool_use when the response carried
    # tool calls but the provider didn't flag it.
    if has_tool_use and translated == "model_finished":
        return "model_tool_use"
    return translated
