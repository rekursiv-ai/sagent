"""Normalize provider-specific termination reasons to one vocabulary.

Each LLM backend reports response termination differently:

- Anthropic: ``end_turn`` / ``tool_use`` / ``pause_turn`` / ``stop_sequence``
  / ``max_tokens`` / ``refusal`` / ``model_context_window_exceeded``.
- OpenAI (and OpenAI-compat: Kimi, Qwen, MiniMax, local llama.cpp):
  ``stop`` / ``length`` / ``tool_calls`` / ``function_call`` / ``content_filter``.
- Google Gemini: ``STOP`` / ``MAX_TOKENS`` / ``SAFETY`` / ``RECITATION`` /
  ``LANGUAGE`` / ``BLOCKLIST`` / ``PROHIBITED_CONTENT`` / ``SPII`` /
  ``IMAGE_SAFETY`` / ``MALFORMED_FUNCTION_CALL`` / ``OTHER`` /
  ``FINISH_REASON_UNSPECIFIED``.

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

from typing import Literal


ProviderKind = Literal["anthropic", "openai", "google"]

# Stop reasons we consider a normal response termination. Anything else
# means the model stopped for a non-happy-path reason (token cap,
# refusal, context overflow) and any tool calls in the response may
# carry truncated input JSON.
BENIGN_STOP_REASONS: frozenset[str] = frozenset(
    {
        "model_finished",  # natural completion
        "model_tool_use",  # model emitted tool call(s)
        "model_continuing",  # long-running request, client should continue
    },
)

_ANTHROPIC_MAP: dict[str, str] = {
    "end_turn": "model_finished",
    "pause_turn": "model_continuing",
    "tool_use": "model_tool_use",
    "refusal": "model_refusal",
    "stop_sequence": "stop_sequence",
    "max_tokens": "max_tokens",
}

# OpenAI ``finish_reason`` → canonical vocabulary.
_OPENAI_MAP: dict[str, str] = {
    "stop": "model_finished",
    "length": "max_tokens",
    "tool_calls": "model_tool_use",
    "function_call": "model_tool_use",  # legacy
    "content_filter": "model_refusal",
}

# Google ``finishReason`` (Gemini) → canonical.
_GOOGLE_MAP: dict[str, str] = {
    "STOP": "model_finished",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "model_refusal",
    "RECITATION": "model_refusal",
    "LANGUAGE": "model_refusal",
    "BLOCKLIST": "model_refusal",
    "PROHIBITED_CONTENT": "model_refusal",
    "SPII": "model_refusal",
    "IMAGE_SAFETY": "model_refusal",
    "MALFORMED_FUNCTION_CALL": "model_refusal",
    "OTHER": "model_finished",
    "FINISH_REASON_UNSPECIFIED": "model_finished",
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
        return _ANTHROPIC_MAP.get(raw or "end_turn", raw or "model_finished")
    if kind == "openai":
        translated = _OPENAI_MAP.get(raw or "", raw or "model_finished")
    else:
        translated = _GOOGLE_MAP.get(raw or "", raw or "model_finished")
    # Upgrade model_finished → model_tool_use when the response carried
    # tool calls but the provider didn't flag it.
    if has_tool_use and translated == "model_finished":
        return "model_tool_use"
    return translated
