from collections.abc import Callable, Collection, Generator, Mapping, Sequence
from typing import Any
from typing_extensions import deprecated

import contextlib
import inspect

from torch import _C, Tensor

import torch
import torch._C._onnx as _C_onnx

"""Functions to export models into the ONNX IR format.

These models can be loaded with the ONNX library and then
converted to models which run on other deep learning frameworks.
"""
__all__ = [
    "_add_block",
    "_add_input_to_block",
    "_add_output_to_block",
    "_apply_friendly_debug_names",
    "_check_flatten_did_not_remove",
    "_create_jit_graph",
    "_decide_add_node_names",
    "_decide_constant_folding",
    "_decide_input_format",
    "_decide_keep_init_as_input",
    "_export",
    "_get_aten_op_overload_name",
    "_get_example_outputs",
    "_get_module_attributes",
    "_get_named_param_dict",
    "_get_param_count_list",
    "_is_constant_tensor_list",
    "_model_to_graph",
    "_optimize_graph",
    "_pre_trace_quant_model",
    "_reset_trace_module_map",
    "_resolve_args_by_export_type",
    "_run_symbolic_function",
    "_run_symbolic_method",
    "_set_input_and_output_names",
    "_setup_trace_module_map",
    "_should_aten_fallback",
    "_signature",
    "_split_tensor_list_constants",
    "_trace",
    "_trace_and_get_graph_from_model",
    "_trigger_symbolic_function_registration",
    "_validate_dynamic_axes",
    "_verify_custom_op_name",
    "disable_apex_o2_state_dict_hook",
    "export",
    "exporter_context",
    "model_signature",
    "register_custom_op_symbolic",
    "select_model_mode_for_export",
    "setup_onnx_logging",
    "unconvertible_ops",
    "unpack_quantized_tensor",
    "unregister_custom_op_symbolic",
    "warn_on_static_input_change",
]
_params_dict = ...

@deprecated("Please set training mode before exporting the model", category=None)
@contextlib.contextmanager
def select_model_mode_for_export(
    model, mode: _C_onnx.TrainingMode
) -> Generator[None, Any]: ...
@deprecated(
    "Please remove usage of this function. Copy its logic if it is required in user code",
    category=None,
)
@contextlib.contextmanager
def disable_apex_o2_state_dict_hook(
    model: torch.nn.Module | torch.jit.ScriptFunction,
) -> Generator[None, Any]: ...
@deprecated("The feature will be removed. Please remove usage of this function")
@contextlib.contextmanager
def setup_onnx_logging(verbose: bool) -> Generator[None, Any]: ...
@deprecated(
    "The feature will be removed. Please remove usage of this function and implement equivalent logic if needed",
    category=None,
)
@contextlib.contextmanager
def exporter_context(
    model, mode: _C_onnx.TrainingMode, verbose: bool
) -> Generator[tuple[None, None, None], Any]: ...
def export(
    model: torch.nn.Module | torch.jit.ScriptModule | torch.jit.ScriptFunction,
    args: tuple[Any, ...] | torch.Tensor,
    f: str,
    *,
    kwargs: dict[str, Any] | None = ...,
    export_params: bool = ...,
    verbose: bool = ...,
    training: _C_onnx.TrainingMode = ...,
    input_names: Sequence[str] | None = ...,
    output_names: Sequence[str] | None = ...,
    operator_export_type: _C_onnx.OperatorExportTypes = ...,
    opset_version: int | None = ...,
    do_constant_folding: bool = ...,
    dynamic_axes: Mapping[str, Mapping[int, str]]
    | Mapping[str, Sequence[int]]
    | None = ...,
    keep_initializers_as_inputs: bool | None = ...,
    custom_opsets: Mapping[str, int] | None = ...,
    export_modules_as_functions: bool | Collection[type[torch.nn.Module]] = ...,
    autograd_inlining: bool = ...,
) -> None: ...
def warn_on_static_input_change(input_states) -> None: ...

_qtype_vtype_map = ...

def unpack_quantized_tensor(
    value, cast_onnx_accepted=...
) -> tuple[Tensor, Tensor, Tensor] | tuple[Any | Tensor]: ...
@deprecated(
    "Unconvertible ops are not definitive. Please remove usage of this function"
)
def unconvertible_ops(
    model, args, training: _C_onnx.TrainingMode = ..., opset_version: int | None = ...
) -> tuple[_C.Graph, list[str]]: ...
def register_custom_op_symbolic(
    symbolic_name: str, symbolic_fn: Callable, opset_version: int
) -> None: ...
def unregister_custom_op_symbolic(symbolic_name: str, opset_version: int) -> None: ...
def model_signature(model: torch.nn.Module | Callable) -> inspect.Signature: ...
