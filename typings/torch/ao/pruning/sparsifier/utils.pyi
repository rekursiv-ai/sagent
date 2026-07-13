from typing import Any

from torch import nn

__all__ = [
    "FakeSparsity",
    "fqn_to_module",
    "get_arg_info_from_tensor_fqn",
    "module_contains_param",
    "module_to_fqn",
    "swap_module",
]

def module_contains_param(
    module: nn.Module, parametrization: type[nn.Module]
) -> bool: ...
def swap_module(
    mod: nn.Module, mapping: dict[type[nn.Module], type[nn.Module]]
) -> nn.Module: ...
def module_to_fqn(
    model: nn.Module, module: nn.Module, prefix: str = ...
) -> str | None: ...
def fqn_to_module(model: nn.Module | None, path: str) -> nn.Module | None: ...
def get_arg_info_from_tensor_fqn(
    model: nn.Module, tensor_fqn: str
) -> dict[str, Any]: ...

class FakeSparsity(nn.Module):
    def __init__(self, mask) -> None: ...
    def forward(self, x): ...
    def state_dict(self, *args, **kwargs) -> dict[Any, Any]: ...
