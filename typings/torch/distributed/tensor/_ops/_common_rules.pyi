from torch.distributed.tensor._op_schema import OpSchema, OutputSharding

def einop_rule(
    equation: str,
    op_schema: OpSchema,
    *,
    linearity: bool = ...,
    enforce_sharding: dict[str, int] | None = ...,
) -> OutputSharding: ...
def pointwise_rule(op_schema: OpSchema, linearity: bool = ...) -> OutputSharding: ...
