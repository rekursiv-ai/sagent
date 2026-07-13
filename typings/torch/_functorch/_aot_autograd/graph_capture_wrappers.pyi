from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from torch import Tensor

from .descriptors import AOTInput
from .schemas import (
    AOTConfig,
    FxValue,
    JointTraceFn,
    PreppedForAutogradTraceFn,
    SubclassTracingInfo,
    TraceFn,
    ViewAndMutationMeta,
)

"""
This module is responsible for transforming functions to be traced into a form
that is easier for the downstream infra (e.g. Autograd, FX, AOTAutograd analysis)
to handle.

It does so by:
1. functionalization (including RNG functionalzation)
2. creating a joint graph when required
3. transforming mutations into extra outputs
4. dispatching subclasses
"""

def fn_input_mutations_to_outputs(
    fn: Callable,
    args_descs: list[AOTInput],
    meta: ViewAndMutationMeta,
    keep_data_input_mutations: bool,
) -> Any: ...
@contextmanager
def disable_autocast() -> Generator[None, Any, None]: ...
def fn_prepped_for_autograd(
    fn: TraceFn, args_descs: list[AOTInput], meta: ViewAndMutationMeta
) -> PreppedForAutogradTraceFn: ...

@dataclass
class JointFnHandle:
    post_forward: Callable | None = ...

def create_joint(
    fn: Any, primals_descs: list[AOTInput] | None = ..., *, aot_config: AOTConfig
) -> Any: ...
def create_functionalized_rng_ops_wrapper(
    func, args, args_descs, trace_joint=...
) -> Any: ...
@contextmanager
def set_partitioner_tag(tag: str) -> Generator[None, Any, None]: ...
def set_partitioner_tag_is_backward() -> _GeneratorContextManager[None, None, None]: ...
def set_partitioner_tag_must_be_in_backward() -> _GeneratorContextManager[
    None, None, None
]: ...
def set_partitioner_tag_must_be_in_forward() -> _GeneratorContextManager[
    None, None, None
]: ...

@dataclass
class MutationCounters:
    mc_data: int
    mc_storage: int
    mc_inductor_storage_resized: int

T = TypeVar("T")

def sc_visit(
    t, fn: Callable[[Tensor], T], reduce_fn: Callable[[T, T], T], accum_init: T
) -> T: ...
def apply_in_graph_mutations(
    input_info,
    inpt_old,
    inpt_new,
    f_inpt,
    input_idx,
    mcs: MutationCounters | None = ...,
    applied_mcs: MutationCounters | None = ...,
) -> None: ...
def create_functionalized_fn(
    fn,
    args,
    args_descs,
    *,
    meta: ViewAndMutationMeta,
    aot_config: AOTConfig,
    trace_joint: bool,
    joint_fn_handle: JointFnHandle | None = ...,
) -> Any: ...
def handle_effect_tokens_fn(
    fn,
    args,
    args_descs: list[AOTInput],
    *,
    meta: ViewAndMutationMeta,
    trace_joint: bool,
) -> Any: ...
def aot_dispatch_subclass(
    flat_fn_maybe_joint: JointTraceFn | TraceFn,
    args: list[FxValue] | tuple[list[FxValue], list[FxValue]],
    args_descs: list[AOTInput] | tuple[list[AOTInput], list[AOTInput]],
    *,
    is_joint_structure: bool,
    meta: ViewAndMutationMeta,
    fw_only: Callable,
) -> SubclassTracingInfo: ...
def create_functional_call(
    mod, params_spec, params_len, store_orig_mod=..., strict_out_tuple=...
) -> Callable[..., Any | tuple[Any, ...] | list[Any]]: ...
