from collections.abc import Callable
from typing import Any

import contextlib

from torch import Tensor, nn
from torch._inductor.cudagraph_utils import BoxedDeviceIndex
from torch._inductor.utils import BoxedBool
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

import torch

from ._aot_autograd.descriptors import AOTInput
from ._aot_autograd.schemas import (
    AOTConfig,
    AOTDispatchCompiler,
    AOTState,
    FakifiedFlatArgs,
    GraphSignature,
    JointWithDescriptors,
)

static_inputs_log = ...
zip = ...
AOT_COUNTER = ...
aot_autograd_decompositions = ...

def create_aot_state(
    stack: contextlib.ExitStack,
    flat_fn,
    fake_flat_args: FakifiedFlatArgs,
    flat_args_descs: list[AOTInput],
    aot_config: AOTConfig,
    fake_mode: FakeTensorMode,
    shape_env: ShapeEnv | None,
) -> AOTState: ...
def aot_function(
    fn: Callable,
    fw_compiler: Callable,
    bw_compiler: Callable | None = ...,
    partition_fn: Callable = ...,
    decompositions: dict | None = ...,
    num_params_buffers: int = ...,
    keep_inference_input_mutations: bool = ...,
    inference_compiler: Callable | None = ...,
    *,
    dynamic=...,
    enable_log=...,
) -> Callable: ...
def aot_module(mod: nn.Module, *args, **kwargs) -> nn.Module: ...
def prepare_aot_module_simplified(
    mod: nn.Module,
    args,
    kwargs,
    fw_compiler: AOTDispatchCompiler | None,
    bw_compiler: AOTDispatchCompiler | None,
    partition_fn: Callable,
    decompositions: dict,
    keep_inference_input_mutations,
    inference_compiler: AOTDispatchCompiler | None,
    boxed_forward_device_index: BoxedDeviceIndex,
    ignore_shape_env: bool,
    flatten: bool,
    *,
    force_non_lazy_backward_lowering: bool = ...,
) -> tuple[
    Callable[..., Any] | Callable[..., Any | tuple[Any, ...] | list[Any]],
    list[Tensor | Parameter],
    list[str],
    list[str],
    FakifiedFlatArgs,
    list[Any],
    AOTConfig,
    FakeTensorMode,
    ShapeEnv | None,
    TreeSpec | None,
    PytreeThunk | None,
]: ...
def aot_module_simplified(
    mod: nn.Module,
    args,
    fw_compiler: AOTDispatchCompiler,
    bw_compiler: AOTDispatchCompiler | None = ...,
    partition_fn: Callable = ...,
    decompositions: dict | None = ...,
    keep_inference_input_mutations=...,
    inference_compiler: AOTDispatchCompiler | None = ...,
    cudagraphs: BoxedBool | None = ...,
    boxed_forward_device_index: BoxedDeviceIndex | None = ...,
    ignore_shape_env: bool = ...,
) -> nn.Module: ...
def boxed_nop_preserve_node_meta(fx_g, example_inputs) -> Callable[..., Any]: ...
def aot_export_joint_with_descriptors(
    stack: contextlib.ExitStack,
    mod: nn.Module,
    args,
    kwargs=...,
    *,
    decompositions: dict | None = ...,
    keep_inference_input_mutations=...,
    ignore_shape_env=...,
    fw_compiler: AOTDispatchCompiler | None = ...,
    bw_compiler: AOTDispatchCompiler | None = ...,
) -> JointWithDescriptors: ...
def aot_compile_joint_with_descriptors(jd: JointWithDescriptors) -> callable: ...
def aot_export_module(
    mod: nn.Module,
    args,
    *,
    decompositions: dict | None = ...,
    trace_joint: bool,
    output_loss_index: int | None = ...,
    pre_dispatch: bool = ...,
    dynamic_shapes: bool | None = ...,
    kwargs=...,
) -> tuple[torch.fx.GraphModule, GraphSignature]: ...
def aot_export_joint_simple(
    func: Callable,
    args,
    *,
    trace_joint: bool,
    num_params_buffers: int = ...,
    decompositions: dict | None = ...,
) -> torch.fx.GraphModule: ...

compiled_function = ...
compiled_module = ...
