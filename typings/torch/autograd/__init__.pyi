from collections.abc import Sequence as Sequence
from typing import (
    Optional as Optional,
    Union as Union,
    cast as cast,
)

from torch import _vmap_internals as _vmap_internals
from torch._C._autograd import (
    DeviceType as DeviceType,
    ProfilerEvent as ProfilerEvent,
    SavedTensor as SavedTensor,
    _add_metadata_json as _add_metadata_json,
    _disable_profiler as _disable_profiler,
    _disable_profiler_legacy as _disable_profiler_legacy,
    _enable_profiler as _enable_profiler,
    _enable_profiler_legacy as _enable_profiler_legacy,
    _enable_record_function as _enable_record_function,
    _get_sequence_nr as _get_sequence_nr,
    _kineto_step as _kineto_step,
    _KinetoEvent as _KinetoEvent,
    _pop_saved_tensors_default_hooks as _pop_saved_tensors_default_hooks,
    _prepare_profiler as _prepare_profiler,
    _profiler_enabled as _profiler_enabled,
    _ProfilerResult as _ProfilerResult,
    _push_saved_tensors_default_hooks as _push_saved_tensors_default_hooks,
    _record_function_with_args_enter as _record_function_with_args_enter,
    _record_function_with_args_exit as _record_function_with_args_exit,
    _set_empty_test_observer as _set_empty_test_observer,
    _supported_activities as _supported_activities,
    _toggle_collection_dynamic as _toggle_collection_dynamic,
    kineto_available as kineto_available,
)
from torch._C._profiler import (
    ProfilerActivity as ProfilerActivity,
    ProfilerConfig as ProfilerConfig,
    ProfilerState as ProfilerState,
)
from torch.overrides import (
    handle_torch_function as handle_torch_function,
    has_torch_function as has_torch_function,
    is_tensor_like as is_tensor_like,
)
from torch.types import (
    _size as _size,
    _TensorOrTensors as _TensorOrTensors,
    _TensorOrTensorsOrGradEdge as _TensorOrTensorsOrGradEdge,
)

import torch

from . import (
    forward_ad as forward_ad,
    functional as functional,
    graph as graph,
    profiler as profiler,
)
from .anomaly_mode import (
    detect_anomaly as detect_anomaly,
    set_detect_anomaly as set_detect_anomaly,
)
from .function import (
    Function as Function,
    NestedIOFunction as NestedIOFunction,
)
from .grad_mode import (
    _force_original_view_tracking as _force_original_view_tracking,
    _unsafe_preserve_version_counter as _unsafe_preserve_version_counter,
    enable_grad as enable_grad,
    inference_mode as inference_mode,
    no_grad as no_grad,
    set_grad_enabled as set_grad_enabled,
    set_multithreading_enabled as set_multithreading_enabled,
)
from .gradcheck import (
    gradcheck as gradcheck,
    gradgradcheck as gradgradcheck,
)
from .graph import _engine_run_backward as _engine_run_backward
from .variable import Variable as Variable

"""
``torch.autograd`` provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.

It requires minimal changes to the existing code - you only need to declare :class:`Tensor` s
for which gradients should be computed with the ``requires_grad=True`` keyword.
As of now, we only support autograd for floating point :class:`Tensor` types (
half, float, double and bfloat16) and complex :class:`Tensor` types (cfloat, cdouble).
"""
__all__ = [
    "Function",
    "NestedIOFunction",
    "Variable",
    "backward",
    "detect_anomaly",
    "enable_grad",
    "grad",
    "grad_mode",
    "gradcheck",
    "gradgradcheck",
    "inference_mode",
    "no_grad",
    "set_detect_anomaly",
    "set_grad_enabled",
    "set_multithreading_enabled",
    "variable",
]
type _OptionalTensor = torch.Tensor | None
type _ShapeorNestedShape = _size | Sequence[_size] | torch.Tensor

def backward(
    tensors: _TensorOrTensorsOrGradEdge,
    grad_tensors: _TensorOrTensors | None = ...,
    retain_graph: bool | None = ...,
    create_graph: bool = ...,
    grad_variables: _TensorOrTensors | None = ...,
    inputs: _TensorOrTensorsOrGradEdge | None = ...,
) -> None: ...
def grad(
    outputs: _TensorOrTensorsOrGradEdge,
    inputs: _TensorOrTensorsOrGradEdge,
    grad_outputs: _TensorOrTensors | None = ...,
    retain_graph: bool | None = ...,
    create_graph: bool = ...,
    only_inputs: bool = ...,
    allow_unused: bool | None = ...,
    is_grads_batched: bool = ...,
    materialize_grads: bool = ...,
) -> tuple[torch.Tensor, ...]: ...
def variable(*args, **kwargs): ...

is_multithreading_enabled = ...
is_view_replay_enabled = ...
