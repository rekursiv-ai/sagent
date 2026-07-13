from typing_extensions import deprecated

from torch.autograd.function import Function

class Type(Function):
    @staticmethod
    @deprecated(
        "`torch.autograd._functions.Type` is deprecated as of PyTorch 2.1, please use `torch.tensor.to(dtype=dtype)` instead.",
        category=FutureWarning,
    )
    def forward(ctx, i, dest_type): ...
    @staticmethod
    def backward(ctx, grad_output) -> tuple[Any, None]: ...

class Resize(Function):
    @staticmethod
    def forward(ctx, tensor, sizes): ...
    @staticmethod
    def backward(ctx, grad_output) -> tuple[Any, None]: ...
