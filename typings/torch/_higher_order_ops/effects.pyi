from enum import Enum
from typing import Any

from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

import torch

class _EffectType(Enum):
    ORDERED = ...

type OpType = torch._ops.HigherOrderOperator | torch._ops.OpOverload
SIDE_EFFECTS = ...

class WithEffects(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(
        self, token, op: OpType, *args: tuple[Any, ...], **kwargs: dict[str, Any]
    ) -> tuple[Any, ...]: ...

with_effects = ...

def has_aliasing(op: OpType) -> bool: ...
def has_effects(op, args, kwargs) -> bool: ...
def get_effect_key(op, args, kwargs) -> _EffectType | None: ...
def new_token_tensor() -> torch.Tensor: ...
@with_effects.py_impl(DispatchKey.CompositeExplicitAutograd)
def with_effects_dense(
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]: ...
@with_effects.py_impl(FakeTensorMode)
def with_effects_fake(
    mode,
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]: ...
@with_effects.py_impl(ProxyTorchDispatchMode)
def with_effects_proxy(
    mode,
    token: torch.Tensor,
    op: torch._ops.OpOverload,
    *args: tuple[Any, ...],
    **kwargs: dict[str, Any],
) -> tuple[torch.Tensor, ...]: ...
def handle_effects(
    allow_token_discovery: bool,
    tokens: dict[_EffectType, torch.Tensor],
    op: OpType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any: ...
