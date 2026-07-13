from torch import nn
from torch.distributed.tensor import DTensor, Placement

import torch

from ..utils.generic import GeneralInterface

logger = ...
_torch_distributed_available = ...

def initialize_tensor_parallelism(tp_plan, tp_size=...): ...

str_to_dtype = ...

def get_packed_weights(param, empty_param, device_mesh, rank, dim): ...
def repack_weights(
    packed_parameter: torch.Tensor,
    sharded_dim: int,
    world_size: int,
    num_blocks: int = ...,
) -> torch.Tensor: ...
def get_tensor_shard(param, empty_param, device_mesh, rank, dim):  # -> Tensor:
    ...
def distribute_module(
    module: nn.Module, device_mesh=..., input_fn=..., output_fn=...
) -> nn.Module: ...

class TensorParallelLayer:
    use_dtensor = ...
    def partition_tensor(
        self,
        param,
        empty_param,
        param_type,
        param_casting_dtype,
        to_contiguous,
        rank,
        device_mesh,
    ): ...
    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module: ...

class GatherParallel(TensorParallelLayer):
    def __init__(
        self,
        *,
        input_layouts: Placement | None = ...,
        output_layouts: Placement | None = ...,
        use_local_output: bool = ...,
    ) -> None: ...
    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module: ...

class IsolatedParallel(TensorParallelLayer):
    def partition_tensor(
        self,
        param,
        empty_param,
        param_type,
        param_casting_dtype,
        to_contiguous,
        rank,
        device_mesh,
    ): ...
    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module: ...

class ReplicateParallel(TensorParallelLayer):
    def __init__(self, *, use_dtensor=..., use_local_output=...) -> None: ...
    def partition_tensor(
        self,
        param,
        empty_param,
        param_type,
        param_casting_dtype,
        to_contiguous,
        rank,
        device_mesh,
    ):  # -> DTensor:
        ...

class ColwiseParallel(TensorParallelLayer):
    def __init__(
        self,
        *,
        input_layouts: Placement | None = ...,
        output_layouts: Placement | None = ...,
        use_local_output: bool = ...,
        use_dtensor=...,
    ) -> None: ...
    def partition_tensor(
        self,
        param,
        empty_param,
        param_type,
        param_casting_dtype,
        to_contiguous,
        rank,
        device_mesh,
    ):  # -> Parameter:
        ...

class PackedColwiseParallel(ColwiseParallel):
    def partition_tensor(
        self,
        param,
        empty_param,
        param_type,
        param_casting_dtype,
        to_contiguous,
        rank,
        device_mesh,
    ):  # -> Parameter:
        ...

class RowwiseParallel(TensorParallelLayer):
    def __init__(
        self,
        *,
        input_layouts: Placement | None = ...,
        output_layouts: Placement | None = ...,
        use_local_output: bool = ...,
        use_dtensor=...,
    ) -> None: ...
    def partition_tensor(
        self,
        param,
        empty_param,
        param_type,
        param_casting_dtype,
        to_contiguous,
        rank,
        device_mesh,
    ):  # -> Parameter:
        ...
    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module: ...

class PackedRowwiseParallel(RowwiseParallel):
    def partition_tensor(
        self,
        param,
        empty_param,
        param_type,
        param_casting_dtype,
        to_contiguous,
        rank,
        device_mesh,
    ):  # -> Parameter:
        ...

class SequenceParallel(TensorParallelLayer):
    def __init__(
        self, *, sequence_dim: int = ..., use_local_output: bool = ..., use_dtensor=...
    ) -> None: ...
    def partition_tensor(
        self,
        param,
        empty_param,
        param_type,
        param_casting_dtype,
        to_contiguous,
        rank,
        device_mesh,
    ):  # -> Parameter:
        ...

class GroupedGemmParallel(TensorParallelLayer):
    def __init__(self) -> None: ...
    def partition_tensor(
        self,
        param,
        empty_param,
        param_type,
        param_casting_dtype,
        to_contiguous,
        rank,
        device_mesh,
    ): ...

class RouterParallel(TensorParallelLayer):
    def __init__(self, *args, **kwargs) -> None: ...
    def partition_tensor(
        self,
        param,
        empty_param,
        param_type,
        param_casting_dtype,
        to_contiguous,
        rank,
        device_mesh,
    ): ...
    def prepare_module_tp(self, module: nn.Module, device_mesh) -> nn.Module: ...

class ParallelInterface(GeneralInterface):
    _global_mapping = ...

ALL_PARALLEL_STYLES: ParallelInterface = ...

def convert_local_tensor_to_dtensor(
    parameter: torch.Tensor, parameter_name: str, device_mesh, tp_plan: dict[str, str]
) -> DTensor: ...
def replace_state_dict_local_with_dtensor(
    state_dict: dict[str, torch.Tensor], tp_plan: dict[str, str], device_mesh
) -> dict[str, torch.Tensor]: ...
def add_tensor_parallel_hooks_to_module(
    model,
    module,
    tp_plan,
    layer_name,
    current_module_plan,
    device_mesh,
    parameter_name=...,
):  # -> None:
    ...
def shard_and_distribute_module(
    model,
    param,
    empty_param,
    parameter_name,
    param_casting_dtype,
    is_contiguous,
    rank,
    device_mesh,
):  # -> Parameter:
    ...
def verify_tp_plan(
    expected_keys: list[str], tp_plan: dict[str, str] | None
):  # -> None:
    ...
def distribute_model(model, distributed_config, device_mesh, tp_size): ...
