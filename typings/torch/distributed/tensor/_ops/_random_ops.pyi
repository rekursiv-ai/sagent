from torch.distributed.tensor._op_schema import OpSchema, StrategyType
from torch.distributed.tensor._ops.utils import register_op_strategy

aten = ...

@register_op_strategy(
    [
        aten.normal_.default,
        aten.uniform_.default,
        aten.native_dropout.default,
        aten.bernoulli_.float,
        aten.bernoulli.default,
    ]
)
def random_op_strategy(op_schema: OpSchema) -> StrategyType: ...
