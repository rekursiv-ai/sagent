from collections import defaultdict as defaultdict
from collections.abc import (
    Callable as Callable,
    Sequence as Sequence,
)
from functools import (
    lru_cache as lru_cache,
    partial as partial,
    wraps as wraps,
)
from itertools import chain as chain
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
    Optional as Optional,
    TypeVar as TypeVar,
    Union as Union,
)
from typing_extensions import ParamSpec as ParamSpec

from torch._ops import (
    HigherOrderOperator as HigherOrderOperator,
    OperatorBase as OperatorBase,
    OpOverload as OpOverload,
    OpOverloadPacket as OpOverloadPacket,
)
from torch._prims_common import CustomOutParamAnnotation as CustomOutParamAnnotation
from torch._subclasses.functional_tensor import FunctionalTensor as FunctionalTensor
from torch.export.decomp_utils import CustomDecompTable as CustomDecompTable

import torch
import torch._decomp.decompositions
import torch._refs
import torch.library

__all__ = [
    "_should_decompose_because_unsafe_op",
    "core_aten_decompositions",
    "decomposition_table",
    "get_decompositions",
    "meta_table",
    "pre_autograd_decomposition_table",
    "register_decomposition",
]
_T = TypeVar("_T")
_P = ParamSpec("_P")
global_decomposition_table: dict[str, dict[torch._ops.OperatorBase, Callable]] = ...
decomposition_table = ...
pre_autograd_decomposition_table = ...
meta_table = ...

def register_decomposition(
    aten_op, registry=..., *, type=..., unsafe=...
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def get_decompositions(
    aten_ops: Sequence[torch._ops.OperatorBase | OpOverloadPacket], type: str = ...
) -> dict[torch._ops.OperatorBase, Callable]: ...
def remove_decompositions(
    decompositions: dict[torch._ops.OperatorBase, Callable],
    aten_ops: Sequence[OpOverload | OpOverloadPacket],
) -> None: ...
def core_aten_decompositions() -> CustomDecompTable: ...
