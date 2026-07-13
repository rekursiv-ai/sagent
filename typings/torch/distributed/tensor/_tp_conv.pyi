import torch

aten = ...

def tp_convolution(
    op_call: torch._ops.OpOverload,
    local_tensor_args: tuple[object, ...],
    local_tensor_kwargs: dict[str, object],
) -> object: ...
def tp_convolution_backward(
    op_call: torch._ops.OpOverload,
    local_tensor_args: tuple[object, ...],
    local_tensor_kwargs: dict[str, object],
) -> object: ...
def convolution_handler(
    op_call: torch._ops.OpOverload, args: tuple[object, ...], kwargs: dict[str, object]
) -> object: ...
def convolution_backward_handler(
    op_call: torch._ops.OpOverload, args: tuple[object, ...], kwargs: dict[str, object]
) -> object: ...
