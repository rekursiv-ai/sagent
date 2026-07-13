from collections import OrderedDict as OrderedDict
from collections.abc import Callable as Callable
from contextlib import contextmanager as contextmanager
from functools import lru_cache as lru_cache
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
    Any as Any,
    Optional as Optional,
    Union as Union,
)
from unittest.mock import patch as patch

import dataclasses

from torch._C._aoti import AOTIModelContainerRunner as AOTIModelContainerRunner
from torch._dispatch.python import enable_python_dispatcher as enable_python_dispatcher
from torch._guards import compile_context as compile_context
from torch._utils_internal import log_export_usage as log_export_usage
from torch.export._tree_utils import reorder_kwargs as reorder_kwargs
from torch.export.graph_signature import (
    ArgumentSpec as ArgumentSpec,
    ConstantArgument as ConstantArgument,
    ExportGraphSignature as ExportGraphSignature,
    InputKind as InputKind,
    InputSpec as InputSpec,
    OutputKind as OutputKind,
    OutputSpec as OutputSpec,
    SymBoolArgument as SymBoolArgument,
    SymFloatArgument as SymFloatArgument,
    SymIntArgument as SymIntArgument,
    TensorArgument as TensorArgument,
)
from torch.fx._compatibility import compatibility as compatibility
from torch.fx.experimental.proxy_tensor import make_fx as make_fx
from torch.fx.graph import (
    _PyTreeCodeGen as _PyTreeCodeGen,
    _PyTreeInfo as _PyTreeInfo,
)

from .utils import _materialize_cpp_cia_ops as _materialize_cpp_cia_ops
from .wrappers import _wrap_submodules as _wrap_submodules

log = ...

@dataclasses.dataclass
class ExportDynamoConfig:
    allow_rnn: bool = ...

@lru_cache
def aot_compile_warning() -> None: ...
def aot_compile(
    f: Callable,
    args: tuple[Any],
    kwargs: dict[str, Any] | None = ...,
    *,
    dynamic_shapes: dict[str, Any] | None = ...,
    options: dict[str, Any] | None = ...,
    remove_runtime_assertions: bool = ...,
    disable_constraint_solver: bool = ...,
    same_signature: bool = ...,
) -> list[Any] | str: ...
def aot_load(so_path: str, device: str) -> Callable: ...
