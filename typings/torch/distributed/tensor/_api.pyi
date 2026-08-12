from torch import Tensor
from collections.abc import Callable, Sequence
from typing import Any, Self

from torch import nn
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor.placement_types import Placement

import torch
import torch.distributed.tensor._dispatch as op_dispatch

__all__ = [
    "DTensor",
    "distribute_module",
    "distribute_tensor",
    "empty",
    "full",
    "ones",
    "rand",
    "randn",
    "zeros",
]
aten = ...

class _ToTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input: DTensor, grad_placements: Sequence[Placement] | None
    ) -> Tensor: ...
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[DTensor, None]: ...

class _FromTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        run_check: bool,
        shape: torch.Size | None = ...,
        stride: tuple[int, ...] | None = ...,
    ) -> DTensor: ...
    @staticmethod
    def backward(
        ctx, grad_output: DTensor
    ) -> tuple[Tensor, None, None, None, None, None]: ...

class DTensor(torch.Tensor):
    _local_tensor: torch.Tensor
    _spec: DTensorSpec
    __slots__ = ...
    _op_dispatcher: op_dispatch.OpDispatcher = ...
    @staticmethod
    @torch._disable_dynamo
    def __new__(
        cls, local_tensor: torch.Tensor, spec: DTensorSpec, *, requires_grad: bool
    ) -> Self: ...
    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental
    def __init__(self, *args, **kwargs) -> None: ...
    def __tensor_flatten__(self) -> tuple[list[str], tuple[DTensorSpec, bool]]: ...
    @staticmethod
    def __tensor_unflatten__(
        inner_tensors, flatten_spec, outer_size, outer_stride
    ) -> DTensor: ...
    def __coerce_tangent_metadata__(self) -> Self | DTensor: ...
    def __coerce_same_metadata_as_tangent__(
        self, flatten_spec, expected_type=...
    ) -> DTensor | None: ...
    @classmethod
    @torch._disable_dynamo
    def __torch_dispatch__(cls, func, types, args=..., kwargs=...) -> object: ...
    @staticmethod
    def from_local(
        local_tensor: torch.Tensor,
        device_mesh: DeviceMesh | None = ...,
        placements: Sequence[Placement] | None = ...,
        *,
        run_check: bool = ...,
        shape: torch.Size | None = ...,
        stride: tuple[int, ...] | None = ...,
    ) -> DTensor: ...
    def to_local(
        self, *, grad_placements: Sequence[Placement] | None = ...
    ) -> torch.Tensor: ...
    def redistribute(
        self,
        device_mesh: DeviceMesh | None = ...,
        placements: Sequence[Placement] | None = ...,
        *,
        async_op: bool = ...,
        forward_dtype: torch.dtype | None = ...,
        backward_dtype: torch.dtype | None = ...,
    ) -> DTensor: ...
    def full_tensor(
        self, *, grad_placements: Sequence[Placement] | None = ...
    ) -> torch.Tensor: ...
    @property
    def device_mesh(self) -> DeviceMesh: ...
    @property
    def placements(self) -> tuple[Placement, ...]: ...
    def __create_write_items__(self, fqn: str, object: Any) -> list[WriteItem]: ...
    def __create_chunk_list__(self) -> list[ChunkStorageMetadata]: ...
    def __get_tensor_shard__(self, index) -> Tensor: ...

def distribute_tensor(
    tensor: torch.Tensor,
    device_mesh: DeviceMesh | None = ...,
    placements: Sequence[Placement] | None = ...,
    *,
    src_data_rank: int | None = ...,
) -> DTensor: ...
def distribute_module(
    module: nn.Module,
    device_mesh: DeviceMesh | None = ...,
    partition_fn: Callable[[str, nn.Module, DeviceMesh], None] | None = ...,
    input_fn: Callable[[nn.Module, Any, DeviceMesh], None] | None = ...,
    output_fn: Callable[[nn.Module, Any, DeviceMesh], None] | None = ...,
) -> nn.Module: ...
def ones(
    *size,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    requires_grad: bool = ...,
    device_mesh: DeviceMesh | None = ...,
    placements: Sequence[Placement] | None = ...,
) -> DTensor: ...
def empty(
    *size,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    requires_grad: bool = ...,
    device_mesh: DeviceMesh | None = ...,
    placements: Sequence[Placement] | None = ...,
) -> DTensor: ...
def full(
    size,
    fill_value,
    *,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    requires_grad: bool = ...,
    device_mesh: DeviceMesh | None = ...,
    placements: Sequence[Placement] | None = ...,
) -> DTensor: ...
def rand(
    *size,
    requires_grad: bool = ...,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device_mesh: DeviceMesh | None = ...,
    placements: Sequence[Placement] | None = ...,
) -> DTensor: ...
def randn(
    *size,
    requires_grad: bool = ...,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device_mesh: DeviceMesh | None = ...,
    placements: Sequence[Placement] | None = ...,
) -> DTensor: ...
def zeros(
    *size,
    requires_grad: bool = ...,
    dtype: torch.dtype | None = ...,
    layout: torch.layout = ...,
    device_mesh: DeviceMesh | None = ...,
    placements: Sequence[Placement] | None = ...,
) -> DTensor: ...
