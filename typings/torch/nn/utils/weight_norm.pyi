from typing import Any, TypeVar
from typing_extensions import deprecated

from torch.nn.modules import Module

r"""Weight Normalization from https://arxiv.org/abs/1602.07868."""
__all__ = ["WeightNorm", "remove_weight_norm", "weight_norm"]

class WeightNorm:
    name: str
    dim: int
    def __init__(self, name: str, dim: int) -> None: ...
    def compute_weight(self, module: Module) -> Any: ...
    @staticmethod
    @deprecated(
        "`torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.",
        category=FutureWarning,
    )
    def apply(module, name: str, dim: int) -> WeightNorm: ...
    def remove(self, module: Module) -> None: ...
    def __call__(self, module: Module, inputs: Any) -> None: ...

T_module = TypeVar("T_module", bound=Module)

def weight_norm(module: T_module, name: str = ..., dim: int = ...) -> T_module: ...
def remove_weight_norm(module: T_module, name: str = ...) -> T_module: ...
