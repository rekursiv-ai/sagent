from . import (
    aot_compile as aot_compile,
    config as config,
    convert_frame as convert_frame,
    eval_frame as eval_frame,
    functional_export as functional_export,
    resume_execution as resume_execution,
)
from .backends.registry import (
    list_backends as list_backends,
    lookup_backend as lookup_backend,
    register_backend as register_backend,
)
from .callback import (
    callback_handler as callback_handler,
    on_compile_end as on_compile_end,
    on_compile_start as on_compile_start,
)
from .code_context import code_context as code_context
from .convert_frame import replay as replay
from .decorators import (
    allow_in_graph as allow_in_graph,
    assume_constant_result as assume_constant_result,
    disable as disable,
    disallow_in_graph as disallow_in_graph,
    dont_skip_tracing as dont_skip_tracing,
    error_on_graph_break as error_on_graph_break,
    forbid_in_graph as forbid_in_graph,
    graph_break as graph_break,
    mark_dynamic as mark_dynamic,
    mark_static as mark_static,
    mark_static_address as mark_static_address,
    maybe_mark_dynamic as maybe_mark_dynamic,
    nonstrict_trace as nonstrict_trace,
    patch_dynamo_config as patch_dynamo_config,
    run as run,
    set_stance as set_stance,
    skip_frame as skip_frame,
    substitute_in_graph as substitute_in_graph,
)
from .eval_frame import (
    OptimizedModule as OptimizedModule,
    _reset_guarded_backend_cache as _reset_guarded_backend_cache,
    explain as explain,
    export as export,
    is_dynamo_supported as is_dynamo_supported,
    is_inductor_supported as is_inductor_supported,
    optimize as optimize,
    optimize_assert as optimize_assert,
    reset_code as reset_code,
)
from .external_utils import is_compiling as is_compiling
from .mutation_guard import GenerationTracker as GenerationTracker
from .pgo import reset_code_state as reset_code_state
from .symbolic_convert import TensorifyState as TensorifyState
from .utils import (
    graph_break_reasons as graph_break_reasons,
    guard_failures as guard_failures,
    orig_code_map as orig_code_map,
    register_hook_for_recompile_user_context as register_hook_for_recompile_user_context,
    reset_frame_count as reset_frame_count,
)

"""
TorchDynamo is a Python-level JIT compiler designed to make unmodified PyTorch programs faster.
TorchDynamo hooks into the frame evaluation API in CPython (PEP 523) to dynamically modify Python
bytecode right before it is executed. It rewrites Python bytecode in order to extract sequences of
PyTorch operations into an FX Graph which is then just-in-time compiled with a customizable backend.
It creates this FX Graph through bytecode analysis and is designed to mix Python execution with
compiled backends to get the best of both worlds: usability and performance. This allows it to
seamlessly optimize PyTorch programs, including those using modern Python features.
"""
__all__ = [
    "OptimizedModule",
    "allow_in_graph",
    "assume_constant_result",
    "config",
    "disable",
    "disallow_in_graph",
    "dont_skip_tracing",
    "error_on_graph_break",
    "explain",
    "export",
    "forbid_in_graph",
    "graph_break",
    "is_compiling",
    "list_backends",
    "lookup_backend",
    "mark_dynamic",
    "mark_static",
    "mark_static_address",
    "maybe_mark_dynamic",
    "nonstrict_trace",
    "optimize",
    "optimize_assert",
    "patch_dynamo_config",
    "register_backend",
    "replay",
    "reset",
    "run",
    "set_stance",
    "skip_frame",
    "substitute_in_graph",
]

def reset() -> None: ...
def reset_code_caches() -> None: ...
