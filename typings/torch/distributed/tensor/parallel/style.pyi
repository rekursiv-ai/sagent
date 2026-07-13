from abc import ABC, abstractmethod

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

__all__ = [
    "ColwiseParallel",
    "ParallelStyle",
    "PrepareModuleInput",
    "PrepareModuleInputOutput",
    "PrepareModuleOutput",
    "RowwiseParallel",
    "SequenceParallel",
]

class ParallelStyle(ABC):
    src_data_rank: int | None
    @abstractmethod
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...

class ColwiseParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Placement | None = ...,
        output_layouts: Placement | None = ...,
        use_local_output: bool = ...,
    ) -> None: ...
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...

class RowwiseParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Placement | None = ...,
        output_layouts: Placement | None = ...,
        use_local_output: bool = ...,
    ) -> None: ...
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...

class SequenceParallel(ParallelStyle):
    def __init__(
        self,
        *,
        sequence_dim: int = ...,
        use_local_output: bool = ...,
    ) -> None: ...
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...

class PrepareModuleInput(ParallelStyle):
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...
    def __init__(
        self,
        *,
        input_layouts: Placement | tuple[Placement | None, ...] | None = ...,
        desired_input_layouts: Placement | tuple[Placement | None, ...] | None = ...,
        input_kwarg_layouts: dict[str, Placement] | None = ...,
        desired_input_kwarg_layouts: dict[str, Placement] | None = ...,
        use_local_output: bool = ...,
    ) -> None: ...

class PrepareModuleOutput(ParallelStyle):
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...
    def __init__(
        self,
        *,
        output_layouts: Placement | tuple[Placement | None, ...],
        desired_output_layouts: Placement | tuple[Placement, ...],
        use_local_output: bool = ...,
    ) -> None: ...

class PrepareModuleInputOutput(ParallelStyle):
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...
    def __init__(
        self,
        *,
        input_layouts: Placement | tuple[Placement | None, ...] | None = ...,
        desired_input_layouts: Placement | tuple[Placement | None, ...] | None = ...,
        input_kwarg_layouts: dict[str, Placement] | None = ...,
        desired_input_kwarg_layouts: dict[str, Placement] | None = ...,
        use_local_input: bool = ...,
        output_layouts: Placement | tuple[Placement | None, ...],
        desired_output_layouts: Placement | tuple[Placement, ...],
        use_local_output: bool = ...,
    ) -> None: ...
