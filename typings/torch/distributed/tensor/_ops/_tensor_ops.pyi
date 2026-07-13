from collections.abc import Sequence

from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpStrategy,
    OutputSharding,
    RuntimeSchemaInfo,
    StrategyType,
)
from torch.distributed.tensor._ops.utils import register_op_strategy, register_prop_rule
from torch.distributed.tensor.placement_types import Placement

aten = ...

def propagate_single_input_strategy(op_schema: OpSchema) -> StrategyType: ...
@register_op_strategy([aten.equal.default, aten.is_same_size.default])
def equal_strategy(op_schema: OpSchema) -> StrategyType: ...
@register_op_strategy(
    [
        aten.empty_like.default,
        aten.ones_like.default,
        aten.rand_like.default,
        aten.randn_like.default,
        aten.zeros_like.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
)
@register_op_strategy(
    [aten.full_like.default], schema_info=RuntimeSchemaInfo(2, ["dtype"])
)
@register_op_strategy(
    [
        aten.randint_like.default,
        aten.randint_like.low_dtype,
        aten.randint_like.low_dtype_out,
    ],
    schema_info=RuntimeSchemaInfo(3, ["dtype"]),
)
def create_like_strategy(op_schema: OpSchema) -> StrategyType: ...
@register_op_strategy(
    [
        aten.new_empty.default,
        aten.new_full.default,
        aten.new_ones.default,
        aten.new_zeros.default,
        aten.new_empty_strided.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
)
def new_factory_strategy(op_schema: OpSchema) -> StrategyType: ...
@register_op_strategy(aten.bucketize.Tensor)
def gen_bucketize_strategy(op_schema: OpSchema) -> StrategyType: ...
@register_op_strategy(aten.select.int, schema_info=RuntimeSchemaInfo(1))
def select_int_strategy(op_schema: OpSchema) -> StrategyType: ...
@register_op_strategy(aten.select_backward.default, schema_info=RuntimeSchemaInfo(1))
def select_backward_strategy(op_schema: OpSchema) -> OpStrategy: ...
@register_op_strategy(aten.slice.Tensor, schema_info=RuntimeSchemaInfo(1))
def gen_slice_strategy(op_schema: OpSchema) -> StrategyType: ...
@register_op_strategy(aten.slice_backward.default, schema_info=RuntimeSchemaInfo(1))
def slice_backward_rules(op_schema: OpSchema) -> OpStrategy: ...
def unshard_tensor_dim(
    placements: Sequence[Placement], dim: int
) -> tuple[Placement, ...]: ...
def replicate_tensor_dim(
    placements: Sequence[Placement], dim: int
) -> tuple[Placement, ...]: ...
@register_op_strategy(aten.slice_scatter.default, schema_info=RuntimeSchemaInfo(2))
def gen_slice_scatter_strategy(op_schema: OpSchema) -> StrategyType: ...
@register_op_strategy(aten._local_scalar_dense.default)
def replica_only_strategy(op_schema: OpSchema) -> StrategyType: ...
@register_op_strategy(
    [aten.scatter_.value, aten.scatter.value, aten.scatter_.src, aten.scatter.src],
    schema_info=RuntimeSchemaInfo(1),
)
def scatter_strategy(op_schema: OpSchema) -> StrategyType: ...
@register_op_strategy(aten.scatter_add.default, schema_info=RuntimeSchemaInfo(1))
def scatter_add_strategy(op_schema: OpSchema) -> StrategyType: ...
@register_op_strategy(aten.gather.default, schema_info=RuntimeSchemaInfo(1))
def gather_strategy(op_schema: OpSchema) -> StrategyType: ...
def normalize_shard_for_stack(
    placements: Sequence[Placement], insert_dim: int = ...
) -> Sequence[Placement]: ...
@register_op_strategy(aten.stack.default, RuntimeSchemaInfo(1, needs_pytree=True))
def stack_strategy(op_schema: OpSchema) -> StrategyType: ...
@register_op_strategy(aten.cat.default, RuntimeSchemaInfo(1, needs_pytree=True))
def cat_strategy(op_schema: OpSchema) -> StrategyType: ...
@register_prop_rule(aten.index_select.default, schema_info=RuntimeSchemaInfo(1))
def prop_index_select(op_schema: OpSchema) -> OutputSharding: ...
@register_op_strategy(
    [aten.index_put.default, aten._index_put_impl_.default],
    schema_info=RuntimeSchemaInfo(needs_pytree=True),
)
def prop_index_put(op_schema: OpSchema) -> StrategyType: ...
@register_prop_rule(aten.index.Tensor, schema_info=RuntimeSchemaInfo(needs_pytree=True))
def prop_index(op_schema: OpSchema) -> OutputSharding: ...
@register_op_strategy(
    [
        aten.split.Tensor,
        aten.split_with_sizes.default,
        aten.split_with_sizes_copy.default,
    ],
    RuntimeSchemaInfo(1),
)
def split_strategy(op_schema: OpSchema) -> OpStrategy: ...
