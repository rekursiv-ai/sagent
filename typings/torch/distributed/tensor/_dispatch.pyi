from torch.distributed.tensor._op_schema import OpInfo, OpSchema, OutputSpecType

import torch

aten = ...
logger = ...

def is_same_size_handler(
    op_call: torch._ops.OpOverload, args: tuple[object, ...], kwargs: dict[str, object]
) -> bool: ...
def found_inf_reduce_handler(
    op_call: torch._ops.OpOverload, args: tuple[object, ...], kwargs: dict[str, object]
) -> None: ...

class OpDispatcher:
    def __init__(self) -> None: ...
    def dispatch(
        self,
        op_call: torch._ops.OpOverload,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object: ...
    @staticmethod
    def redistribute_local_args(
        op_info: OpInfo,
        suggested_input_schema: OpSchema,
        use_val_from_redistribute_schema: bool,
    ) -> None: ...
    def unwrap_to_op_info(
        self,
        op_call: torch._ops.OpOverload,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> OpInfo: ...
    @staticmethod
    def wrap(res: object, spec: OutputSpecType) -> object: ...
