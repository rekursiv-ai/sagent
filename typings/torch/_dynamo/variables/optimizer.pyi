from typing import Any

from torch._dynamo.symbolic_convert import InstructionTranslator

from .base import VariableTracker
from .user_defined import UserDefinedObjectVariable

"""
This module implements variable tracking for PyTorch optimizers during Dynamo tracing.

The OptimizerVariable class provides specialized handling for optimizer instances by:
- Optimizing the tracing of expensive optimizer initialization
- Managing optimizer state and parameter group tracking
- Handling tensor sources and guards for optimizer state tensors
- Supporting CUDA graph execution through static tensor address management
- Providing special handling for parameter gradients and optimizer state tensors

Key features include:
- Efficient initialization tracing via _init_group optimization
- Automatic marking of optimizer state tensors as static for CUDA graphs
- Proper source tracking for parameter groups, gradients, and state tensors
- Guard installation for optimizer state structure
- Support for both CPU and GPU tensor handling
- Cleanup of static tensor references via finalizers

The module integrates with Dynamo's broader tracing system while providing
optimizer-specific optimizations and safety guarantees.
"""

class ArgMappingException(Exception): ...
class GuardInstallException(Exception): ...

perf_hint_log = ...

class OptimizerVariable(UserDefinedObjectVariable):
    _nonvar_fields = ...
    def __init__(
        self,
        value,
        grad_to_source=...,
        static_tensor_names=...,
        tensor_to_source=...,
        **kwargs,
    ) -> None: ...
    def call_method(
        self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> VariableTracker: ...
    def var_getattr(
        self, tx: InstructionTranslator, name
    ) -> (
        GetAttrVariable
        | VariableTracker
        | UserDefinedClassVariable
        | UnspecializedNNModuleVariable
        | NNModuleVariable
        | Any
        | UserMethodVariable
        | WrapperUserMethodVariable
        | LazyVariableTracker
    ): ...
    def graph_break_if_pending_mutation(self, tx) -> None: ...
    def get_python_args(
        self, *args, **kwargs
    ) -> tuple[
        list[complex | Any | list[Any]], dict[str, complex | Any | list[Any]]
    ]: ...
    def move_step_if_cpu(self) -> None: ...
    def map_sources_and_install_guards(self, tx) -> None: ...
    def wrap_tensor(self, tx: InstructionTranslator, tensor_value) -> Any: ...
    def update_list_args(
        self, tx: InstructionTranslator, args, kwargs, py_args, py_kwargs
    ) -> None: ...
    def create_finalizer(self, tx) -> None: ...
