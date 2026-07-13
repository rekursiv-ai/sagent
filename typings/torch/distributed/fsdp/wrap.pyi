from abc import ABC
from collections.abc import Callable, Generator, Iterable
from typing import Any

import contextlib
import types

from torch import nn

__all__ = [
    "CustomPolicy",
    "ModuleWrapPolicy",
    "always_wrap_policy",
    "enable_wrap",
    "lambda_auto_wrap_policy",
    "size_based_auto_wrap_policy",
    "transformer_auto_wrap_policy",
    "wrap",
]

def always_wrap_policy(*args, **kwargs) -> bool: ...

class _Policy(ABC): ...

class ModuleWrapPolicy(_Policy):
    def __init__(self, module_classes: Iterable[type[nn.Module]]) -> None: ...
    def __call__(self, module, recurse, *args, **kwargs) -> bool: ...

class CustomPolicy(_Policy):
    def __init__(
        self, lambda_fn: Callable[[nn.Module], bool | dict[str, Any]]
    ) -> None: ...

def lambda_auto_wrap_policy(
    module: nn.Module, recurse: bool, nonwrapped_numel: int, lambda_fn: Callable
) -> bool: ...
def transformer_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    transformer_layer_cls: set[type[nn.Module]],
) -> bool: ...
def size_based_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    min_num_params: int = ...,
    force_leaf_modules: set[type[nn.Module]] | None = ...,
    exclude_wrap_modules: set[type[nn.Module]] | None = ...,
) -> bool: ...
@contextlib.contextmanager
def enable_wrap(*, wrapper_cls: Any, **wrapper_kwargs: Any) -> Generator[None]: ...
def wrap(module: nn.Module, **wrap_overrides: Any) -> nn.Module: ...

class _ConfigAutoWrap:
    in_autowrap_context: bool = ...
    wrapper_cls: Callable | None = ...
    kwargs: dict[str, Any] = ...
    def __init__(self, **kwargs: dict[str, Any]) -> None: ...
    @staticmethod
    def enable_autowrap_context(kwargs: Any) -> None: ...
    @staticmethod
    def disable_autowrap_context() -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...
