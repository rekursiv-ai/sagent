from collections.abc import Callable
from typing import Any

from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    AHMetadata,
    AHOperation,
    Choice,
    Feedback,
)
from torch._inductor.ir import ChoiceCaller

class LocalFeedback:
    def __init__(self, feedback_fn: Callable[[Choice], Feedback]) -> None: ...
    def __call__(self, choice: Choice) -> Feedback: ...

class InconsistentMetadata(Exception): ...

class AutoHeuristic:
    collected_feedback: dict[Choice, Feedback]
    def __init__(
        self,
        fallback: Callable[[], Choice],
        choices: list[Choice],
        feedback: LocalFeedback | None,
        context: AHContext,
        name: str,
        augment_context: list[AHOperation] | None = ...,
        precondition: Callable[[AHMetadata, AHContext], bool] | None = ...,
    ) -> None: ...
    def satisfies_precondition(self) -> bool: ...
    def get_choice(self) -> Choice: ...
    def get_top_k_choices(
        self, top_k: int, always_included: list[str] | None = ...
    ) -> list[Choice] | None: ...
    def get_collected_feedback(self, choice: Choice) -> Any: ...
    @staticmethod
    def get_device_identifier() -> str: ...
    def get_default_log_path(self) -> str: ...
    def serialize_metadata(self) -> str: ...
    def save_data(self, choice: Choice, feedback_val: Feedback) -> None: ...

class AutoHeuristicSelectAlgorithm(AutoHeuristic):
    def __init__(
        self,
        fallback: Callable[[], ChoiceCaller | None],
        choices: list[ChoiceCaller],
        input_nodes: list[Any],
        context: AHContext,
        name: str,
        augment_context: list[AHOperation] | None = ...,
        precondition: Callable[[AHMetadata, AHContext], bool] | None = ...,
    ) -> None: ...
    def register_global_feedback(
        self, input_nodes: list[Any], choices: list[ChoiceCaller]
    ) -> None: ...
    def get_choice_caller(self) -> ChoiceCaller | None: ...
    def get_top_k_choices_caller(
        self, top_k: int, always_included: list[str] | None = ...
    ) -> list[ChoiceCaller] | None: ...
