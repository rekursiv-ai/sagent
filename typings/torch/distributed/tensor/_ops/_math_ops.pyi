from dataclasses import dataclass
from enum import Enum

from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    RuntimeSchemaInfo,
    TupleStrategy,
)
from torch.distributed.tensor._ops.utils import register_op_strategy
from torch.distributed.tensor.placement_types import Partial, Placement

aten = ...

class Reduction(Enum):
    NONE = ...
    MEAN = ...
    SUM = ...

@dataclass(frozen=True)
class NormReduction:
    norm_type: int | float | str

type ReductionOpType = NormReduction | str

@dataclass(frozen=True)
class _NormPartial(Partial):
    norm_type: int | float | str = ...
    def __post_init__(self) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

def replicate_reduction_dims(
    placements: tuple[Placement, ...], reduction_dims: list[int]
) -> tuple[Placement, ...]: ...
def map_placements_after_reduction(
    placements: tuple[Placement, ...],
    reduction_dims: list[int],
    reduction_dims_map: list[int],
    reduction_op: ReductionOpType,
) -> tuple[Placement, ...]: ...
def get_placement_from_reduction_op(reduction_op: ReductionOpType) -> Placement: ...
def common_reduction_strategy(
    input_strategy: OpStrategy,
    reduce_dims: list[int],
    keep_dim: bool = ...,
    reduction_linear: bool = ...,
    reduction_op: ReductionOpType = ...,
) -> OpStrategy: ...

LINEAR_REDUCTION_OP_MAP = ...

@register_op_strategy(
    list(LINEAR_REDUCTION_OP_MAP.keys()), schema_info=RuntimeSchemaInfo(1)
)
def linear_reduction_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy(aten.cumsum.default, schema_info=RuntimeSchemaInfo(1))
def cumsum_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy(
    [aten.var.correction, aten.var.correction_out],
    schema_info=RuntimeSchemaInfo(1, ["keepdim"]),
)
def var_reduction_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy(
    [aten.linalg_vector_norm.default], schema_info=RuntimeSchemaInfo(1)
)
def vector_norm_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy(
    [aten._foreach_norm.Scalar], schema_info=RuntimeSchemaInfo(1, needs_pytree=True)
)
def foreach_norm_strategy(op_schema: OpSchema) -> TupleStrategy: ...
@register_op_strategy(
    [
        aten._linalg_svd.default,
        aten.linalg_qr.default,
        aten.diagonal_copy.default,
        aten.diag_embed.default,
        aten.diag.default,
        aten.diagonal.default,
        aten.tril.default,
        aten.triu.default,
        aten._linalg_eigh.default,
        aten.upsample_bicubic2d.default,
        aten.upsample_bilinear2d.default,
        aten.upsample_linear1d.default,
        aten.upsample_nearest2d.default,
        aten.upsample_trilinear3d.default,
    ],
    schema_info=RuntimeSchemaInfo(1),
)
def linalg_replicate_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy(
    [aten._log_softmax.default, aten._softmax.default, aten._safe_softmax.default],
    schema_info=RuntimeSchemaInfo(1),
)
def softmax_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy(
    [aten._log_softmax_backward_data.default, aten._softmax_backward_data.default],
    schema_info=RuntimeSchemaInfo(2),
)
def softmax_backward_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy(
    [aten.nll_loss_forward.default, aten.nll_loss2d_forward.default],
    schema_info=RuntimeSchemaInfo(3),
)
def nll_loss_forward_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy(
    [aten.nll_loss_backward.default, aten.nll_loss2d_backward.default],
    schema_info=RuntimeSchemaInfo(4),
)
def nll_loss_backward_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy(
    [aten.native_layer_norm.default], schema_info=RuntimeSchemaInfo(1)
)
def layer_norm_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy([aten._fused_rms_norm.default], schema_info=RuntimeSchemaInfo(1))
def fused_rms_norm_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy(
    [aten.native_layer_norm_backward.default], schema_info=RuntimeSchemaInfo(2)
)
def layer_norm_bwd_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy(
    [aten._fused_rms_norm_backward.default], schema_info=RuntimeSchemaInfo(2)
)
def fused_rms_norm_bwd_strategy(op_schema: OpSchema) -> OpStrategy: ...
def sort_strategy(op_schema: OpSchema, sort_dim: int) -> OpStrategy: ...
@register_op_strategy([aten.topk.default], schema_info=RuntimeSchemaInfo(2))
def topk_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy(aten.sort.default, schema_info=RuntimeSchemaInfo(1))
def sort_default_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy(
    aten.sort.stable,
    schema_info=RuntimeSchemaInfo(1, static_kwargkey=["dim", "descending", "stable"]),
)
def sort_stable_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy([aten.histc.default], schema_info=RuntimeSchemaInfo(2))
def histc_strategy(op_schema: OpSchema) -> OpStrategy: ...
