"""Agent-level model settings."""

from __future__ import annotations

from dataclasses import dataclass

from sagent.types.capability import ModelLimits


__all__ = ["AgentSettings", "default_buffer_tokens"]


def default_buffer_tokens(max_request_tokens: int) -> int:
    """Return proportional compaction headroom for a given input window."""
    return min(max(max_request_tokens // 15, 8_000), max(max_request_tokens // 2, 0))


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSettings:
    """What one Agent chose, within what its model's capability offers."""

    max_request_tokens: int
    """Maximum input tokens the agent will send."""

    max_response_tokens: int
    """Maximum output tokens reserved for the response."""

    buffer_tokens: int = 0
    """Headroom a ``Compactor`` deducts in ``largest_context``, in tokens."""

    max_attempts: int = 5
    """Retry attempts inside one send before the error surfaces."""

    max_tool_call_rounds: int | None = None
    """Cap on tool-call rounds per turn; ``None`` for no cap."""

    max_budget_usd: float | None = None
    """Hard spend cap for this agent's own tree; ``None`` for no cap."""

    def __post_init__(self) -> None:
        """Validate window parameters."""
        if self.max_request_tokens <= 0:
            raise ValueError(
                f"max_request_tokens must be > 0, got {self.max_request_tokens}",
            )
        if self.max_response_tokens <= 0:
            raise ValueError(
                f"max_response_tokens must be > 0, got {self.max_response_tokens}",
            )
        if self.buffer_tokens < 0 or self.buffer_tokens >= self.max_request_tokens:
            raise ValueError(
                f"buffer_tokens ({self.buffer_tokens}) must be in"
                f" [0, max_request_tokens={self.max_request_tokens})",
            )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.max_tool_call_rounds is not None and self.max_tool_call_rounds < 0:
            raise ValueError(
                "max_tool_call_rounds must be >= 0 or None, got"
                f" {self.max_tool_call_rounds}",
            )
        if self.max_budget_usd is not None and self.max_budget_usd < 0:
            raise ValueError(
                f"max_budget_usd must be >= 0 or None, got {self.max_budget_usd}",
            )

    @classmethod
    def from_limits(cls, limits: ModelLimits) -> AgentSettings:
        """Derive proportional defaults from the selected context's limits."""
        return cls(
            max_request_tokens=limits.max_request_tokens,
            max_response_tokens=limits.max_response_tokens,
            buffer_tokens=default_buffer_tokens(limits.max_request_tokens),
        )
