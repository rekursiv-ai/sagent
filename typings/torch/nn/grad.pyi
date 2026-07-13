# Missing nn.grad module (C extension).
from torch import Tensor

def conv2d_input(
    input_size: list[int] | tuple[int, ...],
    weight: Tensor,
    grad_output: Tensor,
    stride: int | tuple[int, ...] = ...,
    padding: int | tuple[int, ...] = ...,
    dilation: int | tuple[int, ...] = ...,
    groups: int = ...,
) -> Tensor: ...
def conv2d_weight(
    input: Tensor,
    weight_size: list[int] | tuple[int, ...],
    grad_output: Tensor,
    stride: int | tuple[int, ...] = ...,
    padding: int | tuple[int, ...] = ...,
    dilation: int | tuple[int, ...] = ...,
    groups: int = ...,
) -> Tensor: ...
