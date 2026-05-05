"""Domain exceptions for the sagent framework."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.lib.message import response_text, response_tool_calls


if TYPE_CHECKING:
    from sagent.custom_types import ModelResponse


class PromptTooLongError(Exception):
    """Raised by providers when the prompt exceeds model limits.

    Args:
      message: Error message.
      actual_tokens: Actual token count that exceeded the limit.
      limit_tokens: Maximum allowed token count.

    """

    def __init__(
        self,
        message: str = "prompt too long",
        *,
        actual_tokens: int | None = None,
        limit_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.actual_tokens = actual_tokens
        self.limit_tokens = limit_tokens

    @property
    def token_gap(self) -> int | None:
        """Return the number of tokens over the limit, or None if unknown."""
        if self.actual_tokens is not None and self.limit_tokens is not None:
            gap = self.actual_tokens - self.limit_tokens
            return gap if gap > 0 else None
        return None


class StreamInterruptedError(Exception):
    """Stream finished with ``stop_reason=tool_use`` but delivered no tool blocks.

    The Anthropic API flags ``stop_reason`` as not always set correctly
    when streaming -- the SDK may reconstruct a final message that
    records the intended terminal reason while the ``tool_use`` block
    itself was dropped mid-stream. Retrying the same request usually
    recovers the tool call; if it doesn't, the carried ``response`` lets
    the agent fall back to returning whatever partial text/thinking was
    delivered instead of looping into an API 400.

    Args:
      response: The partial model response.

    """

    def __init__(self, response: ModelResponse) -> None:
        super().__init__(
            "Stream indicated tool_use but delivered no tool blocks",
        )
        self.response = response


class ModelTerminationError(Exception):
    """Model stopped with an unrecognized non-benign ``stop_reason``.

    Safety net for stop_reasons we don't have an explicit handler for
    (e.g. a new provider value, a vocabulary drift). Recognized
    non-benign reasons (``max_tokens``, ``model_context_window_exceeded``,
    ``refusal``) are handled in the agent loop without raising  --
    ``max_tokens`` triggers the recovery flow, ``refusal`` surfaces
    as a user-visible message.

    Args:
      response: The model response with the unrecognized stop reason.

    """

    def __init__(self, response: ModelResponse) -> None:
        tool_count = len(response_tool_calls(response.content))
        text_len = len(response_text(response.content))
        super().__init__(
            f"Model stopped with unrecognized stop_reason="
            f"{response.stop_reason!r} (tool_calls={tool_count}, "
            f"text_len={text_len}).",
        )
        self.response = response
        self.stop_reason = response.stop_reason
