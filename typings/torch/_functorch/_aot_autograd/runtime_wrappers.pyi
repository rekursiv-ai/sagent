from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import Tensor, fx
from torch._guards import CompileContext, TracingContext

import torch

from .descriptors import AOTInput
from .schemas import (
    AOTConfig,
    CompilerWrapper,
    FxValue,
    InductorWrapper,
    InputAliasInfo,
    MemoryFormatMeta,
    PlainTensorMeta,
    SubclassCreationMeta,
    SubclassMeta,
    TraceFn,
    ViewAndMutationMeta,
)

"""
This module defines runtime wrappers, which, based on previous analysis attempts to:
1. process the inputs and outputs
2. apply mutations
3. handle functionalized randomness
4. deduplicate inputs and consolidate views into their bases (see input_output_analysis)
"""
zip = ...

@dataclass
class RuntimeWrapper(CompilerWrapper):
    indices_of_inps_to_detach: list[int]
    trace_joint: bool
    disable_amp: bool
    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> Callable[..., list[Any]]: ...

class NoopAliasHandler:
    def __init__(self, info, runtime_metadata, trace_joint) -> None: ...
    def __call__(self, orig_inputs, fw_outs, out): ...

class AliasOfInputHandler:
    def __init__(self, info, runtime_metadata, trace_joint) -> None: ...
    def __call__(self, orig_inputs, fw_outs, out): ...

class IsInputHandler:
    def __init__(self, info, runtime_metadata, trace_joint) -> None: ...
    def __call__(self, orig_inputs, fw_outs, out): ...

class AliasOfIntermediateHandler:
    def __init__(self, info, runtime_metadata, trace_joint) -> None: ...
    def __call__(self, orig_inputs, fw_outs, out): ...

_HANDLER_MAP = ...

def make_output_handler(
    info, runtime_metadata, trace_joint
) -> (
    AliasOfInputHandler | AliasOfIntermediateHandler | IsInputHandler | NoopAliasHandler
): ...
def maybe_mark_dynamic_helper(t: torch.Tensor, dims: set[int]) -> None: ...

@dataclass
class FunctionalizedRngRuntimeWrapper(InductorWrapper):
    return_new_outs: bool = ...
    def pre_compile(
        self, flat_fn: torch.fx.GraphModule, flat_args, aot_config, *, fw_metadata
    ) -> None: ...
    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> _Wrapped[..., Any, ..., Any]: ...

@dataclass
class FakifiedOutWrapper(InductorWrapper):
    out_metas: list[torch.Tensor] = ...
    fwd_output_strides: list[list[int] | None] | None = ...
    needs_post_compile: bool = ...
    def pre_compile(
        self, fw_module: fx.GraphModule, flat_args, aot_config, *, fw_metadata
    ) -> None: ...
    def set_fwd_output_strides(self, fwd_output_strides) -> None: ...
    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> _Wrapped[..., Any, ..., list[Tensor] | Any]: ...

@dataclass
class AOTDispatchSubclassWrapper(CompilerWrapper):
    trace_joint: bool
    fw_only: Callable | None
    maybe_subclass_meta: SubclassMeta | None
    num_fw_outs_saved_for_bw: int | None
    def pre_compile(
        self,
        flat_fn: TraceFn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> tuple[Any, Any, Any, ViewAndMutationMeta]: ...
    def post_compile(
        self,
        compiled_fn,
        _aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> _Wrapped[..., Any, ..., tuple[Any, ...]]: ...

@dataclass
class EffectTokensWrapper(CompilerWrapper):
    def post_compile(
        self, compiled_fn, _aot_config, *, runtime_metadata: ViewAndMutationMeta
    ) -> _Wrapped[..., Any, ..., Any | None]: ...

@dataclass
class AOTDedupeWrapper(CompilerWrapper):
    keep_arg_mask: list[bool] = ...
    add_dupe_map: list[int] = ...
    old_input_metadata: list[InputAliasInfo] = ...
    needs_post_compile: bool = ...
    def remove_dupe_args(self, args) -> list[Any]: ...
    def add_dupe_args(self, args) -> list[Any]: ...
    def pre_compile(
        self,
        flat_fn: TraceFn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> tuple[TraceFn, list[FxValue], list[AOTInput], ViewAndMutationMeta]: ...
    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> _Wrapped[..., Any, ..., Any]: ...

@dataclass
class AOTSyntheticBaseWrapper(CompilerWrapper):
    trace_joint: bool
    needs_post_compile: bool = ...
    aliased_arg_idx_with_metadata_mutations: list[int] = ...
    def pre_compile(
        self,
        flat_fn: TraceFn,
        flat_args: list[FxValue],
        flat_args_descs: list[AOTInput],
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
    ) -> tuple[Callable, list[FxValue], list[AOTInput], ViewAndMutationMeta]: ...
    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> _Wrapped[..., Any, ..., Any]: ...

def merge_view_inputs(
    aot_config: AOTConfig,
    fwd_inputs: list[Any],
    fwd_inputs_descs: list[AOTInput] | None,
    mutated_input_info: list[InputAliasInfo],
    *,
    is_inference: bool,
) -> tuple[list[Any], list[AOTInput], list[int | tuple[int, torch.Tensor]] | None]: ...

@dataclass
class AutogradLazyBackwardCompileInfo:
    bw_module: Callable
    placeholder_list: list[Any]
    saved_context: TracingContext | None
    saved_compile_context: CompileContext | None

@dataclass
class CachedAutogradLazyBackwardCompileInfo:
    bw_module_fn: Callable

def initialize_rng_states(
    num_rng: int,
    graphsafe_idx: int,
    fwd_rng_states: list[torch.Generator],
    bwd_rng_states: list[torch.Generator],
) -> None: ...
def coerce_to_expected_memory_format(
    x: torch.Tensor, memory_format: MemoryFormatMeta
) -> Tensor: ...

class AOTDispatchAutograd:
    @staticmethod
    def process_runtime_tangent(
        x, meta: PlainTensorMeta | SubclassCreationMeta
    ) -> tuple[Any, list[Any]] | tuple[Tensor, list[Tensor]]: ...
    @staticmethod
    def post_compile(
        compiled_fw_func,
        compiled_bw_func,
        maybe_subclass_meta: SubclassMeta | None,
        num_symints_saved_for_bw_: int,
        backward_state_indices: list[int],
        disable_amp: bool,
        indices_of_inps_to_detach: list[int],
        lazy_backward_info: AutogradLazyBackwardCompileInfo
        | CachedAutogradLazyBackwardCompileInfo
        | None,
        aot_config: AOTConfig,
        *,
        fw_metadata: ViewAndMutationMeta,
        try_save_cache_entry: Callable | None,
    ) -> Callable[..., list[Any]]: ...

@dataclass
class DebugAssertWrapper(CompilerWrapper):
    flat_requires_grad: list[bool | None] = ...
    def post_compile(
        self,
        compiled_fn,
        aot_config: AOTConfig,
        *,
        runtime_metadata: ViewAndMutationMeta,
    ) -> _Wrapped[..., Any, ..., Any]: ...

def pre_compile(
    wrappers: list[CompilerWrapper],
    flat_fn: TraceFn,
    flat_args: list[FxValue],
    flat_args_descs: list[AOTInput],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> tuple[TraceFn, list[FxValue], list[AOTInput], ViewAndMutationMeta]: ...
def post_compile(
    wrappers: list[CompilerWrapper],
    compiled_fn: Callable,
    aot_config: AOTConfig,
    *,
    runtime_metadata: ViewAndMutationMeta,
) -> tuple[Callable, ViewAndMutationMeta]: ...
def make_runtime_safe(
    fw_metadata: ViewAndMutationMeta, maybe_subclass_meta: SubclassMeta | None
) -> None: ...
