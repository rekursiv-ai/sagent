from typing import Any

class RetryState:
    next_attempt_requested: bool
    current_attempt: int
    custom_values: dict[str, Any] | None
    def __init__(
        self, *, current_attempt: int = ..., custom_values: dict[str, Any] | None = ...
    ) -> None: ...
    def increment_current_attempt(self) -> int: ...
