from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import Any, Self, TypeVar, overload
from typing_extensions import ParamSpec

from torch.utils.hooks import RemovableHandle

import torch

"""Base optimizer."""
_T = TypeVar("_T")
_P = ParamSpec("_P")
type Args = tuple[Any, ...]
type Kwargs = dict[str, Any]
type StateDict = dict[str, Any]
type DeviceDict = dict[torch.device | None, torch.Tensor]
type DeviceDtypeDict = dict[tuple[torch.device, torch.dtype] | None, torch.Tensor]
type GlobalOptimizerPreHook = Callable[
    [Optimizer, Args, Kwargs], tuple[Args, Kwargs] | None
]
type GlobalOptimizerPostHook = Callable[[Optimizer, Args, Kwargs], None]
__all__ = [
    "Optimizer",
    "register_optimizer_step_post_hook",
    "register_optimizer_step_pre_hook",
]
_global_optimizer_pre_hooks: dict[int, GlobalOptimizerPreHook] = ...
_global_optimizer_post_hooks: dict[int, GlobalOptimizerPostHook] = ...
_foreach_supported_types = ...

class _RequiredParameter: ...

required = ...
_params_doc = ...
_foreach_doc = ...
_fused_doc = ...
_capturable_doc = ...
_differentiable_doc = ...
_maximize_doc = ...

def register_optimizer_step_pre_hook(
    hook: GlobalOptimizerPreHook,
) -> RemovableHandle: ...
def register_optimizer_step_post_hook(
    hook: GlobalOptimizerPostHook,
) -> RemovableHandle: ...

type ParamsT = (
    Iterable[torch.Tensor]
    | Iterable[dict[str, Any]]
    | Iterable[tuple[str, torch.Tensor]]
)
R = TypeVar("R")
T = TypeVar("T")

class Optimizer:
    type OptimizerPreHook = Callable[[Self, Args, Kwargs], tuple[Args, Kwargs] | None]
    type OptimizerPostHook = Callable[[Self, Args, Kwargs], None]
    _optimizer_step_pre_hooks: dict[int, OptimizerPreHook]
    _optimizer_step_post_hooks: dict[int, OptimizerPostHook]
    _optimizer_state_dict_pre_hooks: OrderedDict[int, Callable[[Optimizer], None]]
    _optimizer_state_dict_post_hooks: OrderedDict[
        int, Callable[[Optimizer, StateDict], StateDict | None]
    ]
    _optimizer_load_state_dict_pre_hooks: OrderedDict[
        int, Callable[[Optimizer, StateDict], StateDict | None]
    ]
    _optimizer_load_state_dict_post_hooks: OrderedDict[int, Callable[[Optimizer], None]]
    def __init__(self, params: ParamsT, defaults: dict[str, Any]) -> None: ...
    def _accelerator_graph_capture_health_check(self) -> None: ...
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    @staticmethod
    def profile_hook_step(func: Callable[_P, R]) -> Callable[_P, R]: ...
    def register_step_pre_hook(self, hook: OptimizerPreHook) -> RemovableHandle: ...
    def register_step_post_hook(self, hook: OptimizerPostHook) -> RemovableHandle: ...
    def register_state_dict_pre_hook(
        self, hook: Callable[[Optimizer], None], prepend: bool = ...
    ) -> RemovableHandle: ...
    def register_state_dict_post_hook(
        self,
        hook: Callable[[Optimizer, StateDict], StateDict | None],
        prepend: bool = ...,
    ) -> RemovableHandle: ...
    @torch._disable_dynamo
    def state_dict(self) -> StateDict: ...
    def register_load_state_dict_pre_hook(
        self,
        hook: Callable[[Optimizer, StateDict], StateDict | None],
        prepend: bool = ...,
    ) -> RemovableHandle: ...
    def register_load_state_dict_post_hook(
        self, hook: Callable[[Optimizer], None], prepend: bool = ...
    ) -> RemovableHandle: ...
    @torch._disable_dynamo
    def load_state_dict(self, state_dict: StateDict) -> None: ...
    @torch._disable_dynamo
    def zero_grad(self, set_to_none: bool = ...) -> None: ...
    @overload
    def step(self, closure: None = ...) -> None: ...
    @overload
    def step(
        self, closure: Callable[[], torch.Tensor | float]
    ) -> torch.Tensor | float: ...
    def step(
        self,
        closure: Callable[[], torch.Tensor | float] | None = ...,
    ) -> torch.Tensor | float | None: ...
    @torch._disable_dynamo
    def add_param_group(self, param_group: dict[str, Any]) -> None: ...

    # Deliberately `dict[str, Any]`, matching real torch. A PEP 728 open
    # TypedDict types `params`/`lr` and removes ~29 casts, but `list` is
    # invariant: it then rejects every repo optimizer that declares its own
    # `param_groups: list[dict[str, Any]]`, for 148 new errors.
    param_groups: list[dict[str, Any]]

    state: dict[Any, Any]

    defaults: dict[str, Any]
