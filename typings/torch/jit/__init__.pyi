from collections.abc import Callable
from collections.abc import Iterator as Iterator
from contextlib import contextmanager as contextmanager
from typing import Any as Any

import types

from torch._jit_internal import (
    Final as Final,
    Future as Future,
    _Await as _Await,
    _drop as _drop,
    _IgnoreContextManager as _IgnoreContextManager,
    _isinstance as _isinstance,
    _overload as _overload,
    _overload_method as _overload_method,
    export as export,
    ignore as ignore,
    is_scripting as is_scripting,
    unused as unused,
)
from torch.jit._async import (
    fork as fork,
    wait as wait,
)
from torch.jit._await import (
    _awaitable as _awaitable,
    _awaitable_nowait as _awaitable_nowait,
    _awaitable_wait as _awaitable_wait,
)
from torch.jit._decomposition_utils import (
    _register_decomposition as _register_decomposition,
)
from torch.jit._freeze import (
    freeze as freeze,
    optimize_for_inference as optimize_for_inference,
    run_frozen_optimizations as run_frozen_optimizations,
)
from torch.jit._fuser import (
    fuser as fuser,
    last_executed_optimized_graph as last_executed_optimized_graph,
    optimized_execution as optimized_execution,
    set_fusion_strategy as set_fusion_strategy,
)
from torch.jit._ir_utils import _InsertPoint as _InsertPoint
from torch.jit._script import (
    Attribute as Attribute,
    CompilationUnit as CompilationUnit,
    RecursiveScriptClass as RecursiveScriptClass,
    RecursiveScriptModule as RecursiveScriptModule,
    ScriptFunction as ScriptFunction,
    ScriptModule as ScriptModule,
    ScriptWarning as ScriptWarning,
    _ScriptProfile as _ScriptProfile,
    _unwrap_optional as _unwrap_optional,
    interface as interface,
    script as script,
    script_method as script_method,
)
from torch.jit._serialization import (
    jit_module_from_flatbuffer as jit_module_from_flatbuffer,
    load as load,
    save as save,
    save_jit_module_to_flatbuffer as save_jit_module_to_flatbuffer,
)
from torch.jit._trace import (
    ONNXTracedModule as ONNXTracedModule,
    TopLevelTracedModule as TopLevelTracedModule,
    TracedModule as TracedModule,
    TracerWarning as TracerWarning,
    TracingCheckError as TracingCheckError,
    _flatten as _flatten,
    _get_trace_graph as _get_trace_graph,
    _script_if_tracing as _script_if_tracing,
    _unique_state_dict as _unique_state_dict,
    is_tracing as is_tracing,
    trace as trace,
    trace_module as trace_module,
)
from torch.utils import set_module as set_module

import torch._C

__all__ = [
    "Attribute",
    "CompilationUnit",
    "Error",
    "Future",
    "ScriptFunction",
    "ScriptModule",
    "annotate",
    "enable_onednn_fusion",
    "export",
    "export_opnames",
    "fork",
    "freeze",
    "ignore",
    "interface",
    "isinstance",
    "load",
    "onednn_fusion_enabled",
    "optimize_for_inference",
    "save",
    "script",
    "script_if_tracing",
    "set_fusion_strategy",
    "strict_fusion",
    "trace",
    "trace_module",
    "unused",
    "wait",
]
_fork = ...
_wait = ...
_set_fusion_strategy = ...

def export_opnames(m) -> list[str]: ...

Error = torch._C.JITException

def annotate(the_type, the_value): ...
def script_if_tracing(fn) -> Callable[..., Any]: ...
def isinstance(obj, target_type) -> bool: ...

class strict_fusion:
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None: ...

def enable_onednn_fusion(enabled: bool) -> None: ...
def onednn_fusion_enabled() -> bool: ...
