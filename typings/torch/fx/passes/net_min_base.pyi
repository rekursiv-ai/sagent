from collections.abc import Callable
from dataclasses import dataclass

from torch.fx._compatibility import compatibility

import torch
import torch.fx

from .tools_common import Names, NodeList, NodeSet, TensorOrTensors, Tensors

__all__ = [
    "FxNetMinimizerBadModuleError",
    "FxNetMinimizerResultMismatchError",
    "FxNetMinimizerRunFuncError",
]
_LOGGER = ...

@compatibility(is_backward_compatible=False)
class FxNetMinimizerBadModuleError(Exception): ...

@compatibility(is_backward_compatible=False)
class FxNetMinimizerRunFuncError(Exception): ...

@compatibility(is_backward_compatible=False)
class FxNetMinimizerResultMismatchError(Exception): ...

@dataclass
class _MinimizerSettingBase:
    accumulate_error: bool = ...
    traverse_method: str = ...
    find_all: bool = ...
    return_intermediate: bool = ...
    all_outputs: bool = ...

class _MinimizerBase:
    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Tensors,
        compare_fn: Callable[
            [TensorOrTensors, TensorOrTensors, Names], tuple[float, bool]
        ],
        settings: _MinimizerSettingBase,
        module_exporter: Callable[[Tensors, torch.fx.GraphModule, str], None]
        | None = ...,
        exclusion_fn: Callable[[NodeList, int, int], None] | None = ...,
    ) -> None: ...
    def run_shape_prop(self) -> None: ...
    def run_a(
        self, mod: torch.fx.GraphModule, inputs: Tensors, report_idx: int = ...
    ) -> TensorOrTensors: ...
    def run_b(
        self, mod: torch.fx.GraphModule, inputs: Tensors, report_idx: int = ...
    ) -> TensorOrTensors: ...
    def run_nodes(self, start: str | None = ..., end: str | None = ...) -> None: ...
    def print_report(self, report: list[str]) -> None: ...
    def print_reports(self) -> None: ...
    def minimize(
        self,
        start: str | None = ...,
        end: str | None = ...,
        skip_nodes: list | None = ...,
        find_last_node: bool | None = ...,
    ) -> NodeSet: ...
