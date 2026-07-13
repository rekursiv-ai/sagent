from torch._dynamo.symbolic_convert import InstructionTranslator

import torch

from .base import VariableTracker

"""
Distributed computing variable tracking classes for PyTorch Dynamo.

This module implements variable tracking for distributed computing components:
- Process Groups (for collective communication)
- Device Meshes (for distributed tensor sharding)
- Placement Types (for specifying distribution strategies)
- Distributed Tensors and their operations
- Backward hooks for distributed module operations

These classes are responsible for tracking distributed operations during graph
compilation while maintaining proper guards and handling distributed-specific
behaviors. They ensure correct handling of distributed components like process
groups, device meshes, and placement strategies while preserving proper semantics
for distributed tensor operations in the compiled code.

The implementation provides special handling for distributed package availability
checks and proper tracking of distributed state and operations across processes.
"""

class DistributedVariable(VariableTracker):
    def __init__(self, value, **kwargs) -> None: ...
    def python_type(self) -> Any: ...
    @staticmethod
    def is_available() -> bool: ...

def is_from_local(value) -> TypeIs[FunctionType] | bool: ...
def is_constant_pg_functions(value) -> TypeIs[FunctionType] | bool: ...

class WorldMetaClassVariable(DistributedVariable):
    @classmethod
    def is_group_member_type(cls, value) -> bool: ...
    def var_getattr(self, tx: InstructionTranslator, name: str) -> VariableTracker: ...

class PlacementClassVariable(DistributedVariable):
    @staticmethod
    def is_placement_type(value) -> bool: ...
    def as_python_constant(self) -> Any: ...
    def call_function(
        self,
        tx: InstructionTranslator,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker: ...

class PlacementVariable(DistributedVariable):
    @staticmethod
    def is_placement(value) -> bool: ...
    def as_python_constant(self) -> Any: ...
    def var_getattr(self, tx: InstructionTranslator, name: str) -> VariableTracker: ...
    def call_method(
        self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> VariableTracker: ...

class DeviceMeshVariable(DistributedVariable):
    @staticmethod
    def is_device_mesh(value) -> TypeIs[DeviceMesh] | Literal[False]: ...
    def as_python_constant(self) -> Any: ...
    def var_getattr(self, tx: InstructionTranslator, name: str) -> VariableTracker: ...
    def call_method(
        self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> VariableTracker: ...

class ProcessGroupVariable(DistributedVariable):
    def as_python_constant(self) -> Any: ...
    def call_method(
        self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> VariableTracker: ...
    def var_getattr(
        self, tx: InstructionTranslator, name
    ) -> VariableTracker | LambdaVariable: ...
    @staticmethod
    def is_process_group(value) -> bool: ...

class BackwardHookVariable(VariableTracker):
    @staticmethod
    def create(
        tx,
        module: VariableTracker,
        user_hooks: VariableTracker,
        user_pre_hooks: VariableTracker,
    ) -> BackwardHookVariable: ...
    def __init__(
        self,
        proxy: torch.fx.Proxy,
        module: VariableTracker,
        user_hooks: VariableTracker,
        user_pre_hooks: VariableTracker,
        **options,
    ) -> None: ...
    def as_proxy(self) -> Proxy: ...
    def call_method(
        self, tx, name, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> VariableTracker: ...
