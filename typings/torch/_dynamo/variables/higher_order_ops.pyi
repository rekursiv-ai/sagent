from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import contextlib

from torch._dynamo.symbolic_convert import InstructionTranslator
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._guards import Source
from torch._ops import HigherOrderOperator
from torch.utils import _pytree as pytree

from .base import VariableTracker

"""
This module contains classes and utilities for handling higher-order operators in Dynamo.
It provides functionality for tracing and transforming control flow constructs like
conditions (torch.cond), loops (torch.while_loop), maps (torch.ops.higher_order.map),
and other higher-order operations.

The module includes specialized VariableTracker classes for different types of
higher-order operations, along with utilities for:
- Speculating and capturing subgraphs
- Managing control flow
- Handling autograd function applications
- Supporting function transformations
- Processing activation checkpoints

These classes work together to enable Dynamo to correctly trace and compile code
containing complex control flow patterns and higher-order functions while preserving
their semantic behavior.
"""
log = ...
hc_log = ...

@dataclass
class OutputSpec:
    treespec: pytree.TreeSpec
    masks_to_filter_const_values: list[bool] | None = ...
    const_values: list[Any] | None = ...
    def __post_init__(self) -> None: ...

def raise_hard_error_if_graph_break(
    reason,
) -> Callable[..., _Wrapped[..., Any, ..., Any]]: ...
@contextlib.contextmanager
def discard_graph_changes(tx) -> Generator[None, Any, None]: ...
def check_meta_consistency_vt(
    vars1: list[VariableTracker],
    vars2: list[VariableTracker],
    lhs_name: str,
    rhs_name: str,
    include_contiguity: bool = ...,
) -> None: ...
@contextlib.contextmanager
def dynamo_enable_grad(
    tx: InstructionTranslator, enable=...
) -> Generator[None, Any, None]: ...
@contextlib.contextmanager
def dynamo_under_activation_checkpoint(
    tx: InstructionTranslator,
) -> Generator[None, Any, None]: ...
def find_mismatched_vars(var, types, allow_none=...) -> set[Any]: ...
def only_consist_of(var, types, allow_none=...) -> bool: ...
def are_same_graph_modules(fn_name, a_mod, b_mod, fake_mode) -> bool: ...
def validate_args_and_maybe_create_graph_inputs(
    sub_args, tracer, tx, set_subgraph_inputs, description, sub_args_names=...
) -> list[VariableTracker] | list[Any]: ...
def speculate_subgraph(
    tx,
    f,
    sub_args,
    sub_kwargs,
    description,
    *,
    source_target=...,
    always_restore=...,
    enable_grad=...,
    set_subgraph_inputs=...,
    restore_side_effects=...,
    should_flatten_outputs=...,
    remove_consts_from_outputs=...,
    under_activation_checkpoint=...,
    supports_input_mutation=...,
    supports_aliasing=...,
    tracer=...,
) -> tuple[tuple[VariableTracker | Any, OutputSpec], Any, Any]: ...
def make_attr(tx: InstructionTranslator, name) -> Proxy: ...

class TorchHigherOrderOperatorVariable(VariableTracker):
    def __init__(
        self, value: HigherOrderOperator, source: Source | None = ..., **kwargs
    ) -> None: ...
    @staticmethod
    def make(
        value, source=..., **kwargs
    ) -> (
        AssociativeScanHigherOrderVariable
        | AutoFunctionalizeHigherOrderVariable
        | CheckpointHigherOrderVariable
        | CondHigherOrderVariable
        | CustomFunctionHigherOrderOperatorVariable
        | ExecutorchCallDelegateHigherOrderVariable
        | ExportTracepointHigherOrderVariable
        | FlexAttentionBackwardHighOrderVariable
        | FlexAttentionHigherOrderVariable
        | HintsWrapperHigherOrderVariable
        | InvokeSubgraphHigherOrderVariable
        | MapHigherOrderVariable
        | OutDtypeHigherOrderVariable
        | RunWithRNGStateHigherOrderVariable
        | ScanHigherOrderVariable
        | StrictModeHigherOrderVariable
        | TraceWrappedHigherOrderOperatorVariable
        | WhileLoopHigherOrderVariable
        | WhileLoopStackOutputHigherOrderVariable
        | WrapHigherOrderVariable
        | WrapWithAutocastHigherOrderVariable
        | WrapWithSetGradEnabledHigherOrderVariable
        | CallTorchbindHigherOrderVariable
        | DynamoBypassingWrapperHigherOrderVariable
        | BaseHOPVariable
    ): ...
    def call_function(
        self,
        tx: InstructionTranslator,
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker: ...
    def as_python_constant(self) -> HigherOrderOperator: ...

class CustomFunctionHigherOrderOperatorVariable(TorchHigherOrderOperatorVariable): ...

class CondHigherOrderVariable(TorchHigherOrderOperatorVariable):
    supports_input_mutation = ...
    supports_aliasing = ...

class CallTorchbindHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def __init__(self, hop, source, script_obj_var, method_name) -> None: ...

def validate_subgraph_output_types(output: VariableTracker) -> None: ...

class WhileLoopHigherOrderVariable(TorchHigherOrderOperatorVariable):
    supports_input_mutation = ...
    supports_aliasing = ...

class WhileLoopStackOutputHigherOrderVariable(TorchHigherOrderOperatorVariable):
    supports_input_mutation = ...
    supports_aliasing = ...

class AssociativeScanHigherOrderVariable(TorchHigherOrderOperatorVariable):
    supports_input_mutation = ...
    supports_aliasing = ...

class ScanHigherOrderVariable(TorchHigherOrderOperatorVariable):
    supports_input_mutation = ...
    supports_aliasing = ...

def non_single_tensor_return_unsupported(api, ret) -> None: ...

class MapHigherOrderVariable(TorchHigherOrderOperatorVariable):
    supports_input_mutation = ...
    supports_aliasing = ...

class ExecutorchCallDelegateHigherOrderVariable(TorchHigherOrderOperatorVariable): ...

class FunctorchHigherOrderVariable(UserFunctionVariable):
    def call_function(
        self,
        tx: InstructionTranslator,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker: ...

class FunctionalCallVariable(FunctorchHigherOrderVariable):
    def call_function(
        self, tx, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> VariableTracker: ...

class ReparametrizeModuleCallVariable(FunctorchHigherOrderVariable):
    def __init__(self, *args, **kwargs) -> None: ...
    def call_function(
        self, tx, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> VariableTracker: ...

class WrapHigherOrderVariable(TorchHigherOrderOperatorVariable):
    supports_input_mutation = ...
    supports_aliasing = ...
    def install_subgraph_in_output_graph(
        self, tx, fn_vt, fn_args_vt, kwargs, body_gmod, attr_name=...
    ): ...
    def create_wrapped_node(
        self,
        tx: InstructionTranslator,
        fn_vt,
        fn_args_vt,
        kwargs,
        description,
        under_activation_checkpoint=...,
        *,
        subgraph_name=...,
    ) -> tuple[
        tuple[Proxy, *tuple[Any, ...]],
        dict[Any, Any],
        PyTree,
        VariableTracker | Any,
        OutputSpec,
        GraphModule,
        Any,
    ]: ...

class WrapWithSetGradEnabledHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self,
        tx: InstructionTranslator,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker: ...

class WrapWithAutocastHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self,
        tx: InstructionTranslator,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker: ...

class HintsWrapperHigherOrderVariable(TorchHigherOrderOperatorVariable): ...
class OutDtypeHigherOrderVariable(TorchHigherOrderOperatorVariable): ...
class StrictModeHigherOrderVariable(TorchHigherOrderOperatorVariable): ...
class CheckpointHigherOrderVariable(WrapHigherOrderVariable): ...

class DynamoBypassingWrapperHigherOrderVariable(WrapHigherOrderVariable):
    def __init__(self, hop, source) -> None: ...

class ExportTracepointHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self,
        tx: InstructionTranslator,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker: ...

class RunWithRNGStateHigherOrderVariable(TorchHigherOrderOperatorVariable): ...
class AutoFunctionalizeHigherOrderVariable(TorchHigherOrderOperatorVariable): ...

class FlexAttentionBackwardHighOrderVariable(TorchHigherOrderOperatorVariable):
    def proxy_submod(self, tx, arg) -> Proxy: ...
    def to_proxy(self, tx, arg) -> Proxy | list[Any] | tuple[Any, ...]: ...

class TraceWrappedHigherOrderOperatorVariable(TorchHigherOrderOperatorVariable): ...

class FlexAttentionHigherOrderVariable(TorchHigherOrderOperatorVariable):
    @staticmethod
    def normalize_to_args(args, kwargs): ...
    def create_wrapped_node(
        self,
        tx: InstructionTranslator,
        query: VariableTracker,
        fn: VariableTracker,
        fn_name: str,
    ) -> tuple[Proxy, tuple[Any, ...]]: ...

class AutogradFunctionApplyVariable(VariableTracker):
    def __init__(self, fwd_graph, bwd_graph, parent_source, **kwargs) -> None: ...
    def call_function(
        self,
        tx: InstructionTranslator,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker: ...

def maybe_positional_arg_names(func) -> list[Any] | None: ...

class BaseHOPVariable(WrapHigherOrderVariable):
    supports_input_mutation = ...
    supports_aliasing = ...
    def python_type(self) -> type[HigherOrderOperator]: ...

class InvokeSubgraphHigherOrderVariable(WrapHigherOrderVariable):
    supports_input_mutation = ...
    supports_aliasing = ...
    def install_subgraph_in_output_graph(
        self, tx, fn_vt, fn_args_vt, kwargs, body_gmod, attr_name
    ): ...

_hop_name_to_variable_class = ...
