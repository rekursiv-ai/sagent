from typing import Any
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from torch.export import ExportedProgram
from torch.fx import GraphModule

import torch

NUMERIC_DEBUG_HANDLE_KEY = ...
CUSTOM_KEY = ...
log = ...

def generate_numeric_debug_handle(ep: ExportedProgram) -> None: ...

class OutputLogger(torch.nn.Module):
    _is_impure = ...
    def __init__(
        self,
        debug_handle: int,
        node_name: str | None = ...,
        nn_module_stack: object | None = ...,
    ) -> None: ...
    def forward(self, x: object) -> object: ...
    def __call__(self, *args: Any, **kwargs: Any) -> object: ...
    def __extra_repr__(self) -> str: ...

def prepare_for_propagation_comparison(model: GraphModule) -> GraphModule: ...

@dataclass(frozen=True)
class QuantizationComparisonResult:
    actual: torch.Tensor
    ref: torch.Tensor
    @property
    def mse_loss(self) -> object: ...
    @property
    def sqnr(self) -> object: ...
    def loss(
        self, loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> object: ...
    def __post_init__(self) -> None: ...

@dataclass(frozen=True)
class NodeAccuracySummary:
    handle: int
    actual_node_name: str
    actual_module_stack: str
    ref_node_name: str
    ref_module_stack: str
    results: Sequence[QuantizationComparisonResult]

def extract_results_from_loggers(
    model: GraphModule,
) -> dict[int, tuple[str | None, object, list[object]]]: ...
def compare_results(
    ref_results: dict[int, tuple[str | None, object, list[torch.Tensor]]],
    actual_results: dict[int, tuple[str | None, object, list[torch.Tensor]]],
) -> dict[int, NodeAccuracySummary]: ...
