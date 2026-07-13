from contextlib import contextmanager

from torch._dynamo.symbolic_convert import InstructionTranslator

import torch.nn

from .base import VariableTracker
from .user_defined import UserDefinedObjectVariable

"""
This module implements variable tracking for PyTorch nn.Module instances during Dynamo tracing.

It provides specialized handling for different types of nn.Module instances through several key classes:

- NNModuleVariable: Handles instance-specific module tracing, specializing on module id() and placing
  parameters directly on the torch.fx.GraphModule. This creates one graph per module instance.

- UnspecializedNNModuleVariable: Provides class-level module tracing, treating nn.Modules like other
  user-defined objects and passing parameters as inputs to the FX graph. This creates one graph per
  module class.

- UnspecializedBuiltinNNModuleVariable: Specifically handles built-in PyTorch modules (e.g. nn.Linear)
  with appropriate optimizations.

- FSDPManagedNNModuleVariable: Special handling for FSDP-wrapped modules with modified guarding behavior
  and parameter handling.

The module integrates with Dynamo's broader tracing functionality to handle module method calls,
parameter access, hooks, and other nn.Module behaviors while maintaining proper scoping and guarding
of module state.
"""

def initialize_lazy_module(tx: InstructionTranslator, mod, args, kwargs) -> None: ...
@contextmanager
def record_nn_module_stack(
    module_key: str, source, tx, mod: torch.nn.Module
) -> Generator[None, Any, None]: ...
def guard_to_detect_forward_monkeypatching(source, mod) -> None: ...

class NNModuleVariable(VariableTracker):
    _nonvar_fields = ...
    def __init__(
        self, module_type: type, module_key: str, value: torch.nn.Module, **kwargs
    ) -> None: ...
    def get_nn_module_stack_source(self) -> Source: ...
    def set_nn_module_stack_source(self, source) -> None: ...
    def python_type(self) -> type | Tensor | Module | Any: ...
    def unpack_var_sequence(self, tx) -> list[Any]: ...
    def call_obj_hasattr(
        self, tx: InstructionTranslator, name: str
    ) -> VariableTracker: ...
    def is_training(self, tx) -> Any | bool: ...
    def convert_to_unspecialized(self, tx): ...
    def has_key_in_generic_dict(self, tx: InstructionTranslator, key) -> bool: ...
    def var_getattr(
        self, tx: InstructionTranslator, name
    ) -> (
        GetAttrVariable
        | VariableTracker
        | UserDefinedClassVariable
        | NNModuleVariable
        | UnspecializedNNModuleVariable
        | Any
        | UserMethodVariable
        | UserFunctionVariable
    ): ...
    def call_function(
        self, tx, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> VariableTracker: ...
    def call_method(
        self,
        tx,
        name,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
        constant=...,
    ) -> VariableTracker: ...

class UnspecializedNNModuleVariable(UserDefinedObjectVariable):
    _nonvar_fields = ...
    def __init__(self, value, **kwargs) -> None: ...
    def get_nn_module_stack_source(self) -> Source: ...
    def set_nn_module_stack_source(self, source) -> None: ...
    def unpack_var_sequence(
        self, tx
    ) -> list[LazyVariableTracker] | list[VariableTracker]: ...
    def call_function(
        self,
        tx: InstructionTranslator,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker: ...
    def call_method(
        self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> VariableTracker: ...
    def getattr_helper(
        self, tx: InstructionTranslator, field, name_vt
    ) -> VariableTracker | None: ...
    def var_getattr(
        self, tx: InstructionTranslator, name
    ) -> (
        ConstDictVariable
        | NNModuleHooksDictVariable
        | VariableTracker
        | GetAttrVariable
        | UserDefinedClassVariable
        | UnspecializedNNModuleVariable
        | NNModuleVariable
        | Any
        | UserMethodVariable
        | WrapperUserMethodVariable
        | LazyVariableTracker
    ): ...
    def manually_trace_nn_module_getattr(
        self, tx: InstructionTranslator, name
    ) -> VariableTracker: ...

class UnspecializedBuiltinNNModuleVariable(UnspecializedNNModuleVariable): ...

class FSDPManagedNNModuleVariable(UnspecializedNNModuleVariable):
    def __init__(self, value, **kwargs) -> None: ...
