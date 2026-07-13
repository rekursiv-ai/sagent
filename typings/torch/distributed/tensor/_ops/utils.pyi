from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar
from typing_extensions import ParamSpec

from torch._prims_common import DimsSequenceType, DimsType
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    OutputSharding,
    PlacementList,
    RuntimeSchemaInfo,
    StrategyType,
)
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement

import torch

_T = TypeVar("_T")
_P = ParamSpec("_P")

def register_prop_rule(
    op: torch._ops.OpOverload | list[torch._ops.OpOverload],
    schema_info: RuntimeSchemaInfo | None = ...,
) -> Callable[
    [Callable[[OpSchema], OutputSharding]], Callable[[OpSchema], OutputSharding]
]: ...
def register_op_strategy(
    op, schema_info=...
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def replicate_op_strategy(op_schema: OpSchema) -> StrategyType: ...
def as_list(
    x: list[object] | object,
) -> list[object] | torch.fx.immutable_collections.immutable_list: ...
def normalize_dim(dim: int, ndim: int) -> int: ...
def normalize_dims(dims: DimsType, ndim: int) -> DimsSequenceType: ...
def prod(xs: Iterable[int]) -> int: ...
def is_tensor_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool: ...
def is_tensor_evenly_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool: ...
def is_tensor_dim_sharded(spec: DTensorSpec, dim: int) -> bool: ...
def is_tensor_partial(spec: DTensorSpec) -> bool: ...
def infer_broadcast_dims_map(
    common_shape: torch.Size, input_shape: torch.Size
) -> list[int]: ...
def map_placements_after_broadcast(
    placements: tuple[Placement, ...],
    shape: torch.Size,
    broadcast_dims_map: list[int],
    partial_to_replicate: bool = ...,
) -> tuple[Placement, ...]: ...
def generate_redistribute_costs(
    src_strategy: OpStrategy, dst_spec: DTensorSpec
) -> list[float]: ...
def expand_to_full_mesh_op_strategy(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    single_mesh_dim_strategies: list[PlacementList],
    *,
    input_index: int = ...,
    inplace_op: bool = ...,
    is_valid_strategy_cb: Callable[
        [list[DTensorSpec], tuple[DTensorSpec | None, ...]], bool
    ]
    | None = ...,
) -> OpStrategy: ...
