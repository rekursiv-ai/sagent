from collections.abc import Generator
from types import CellType
from typing import Any

import contextlib

from torch._dynamo.output_graph import OutputGraph
from torch._dynamo.symbolic_convert import InstructionTranslatorBase
from torch._dynamo.variables.misc import AutogradFunctionContextVariable

from . import variables
from .codegen import PyCodegen
from .source import Source
from .variables.base import VariableTracker

"""
Side effect tracking and management for TorchDynamo's compilation system.

This module provides infrastructure for tracking and managing side effects that occur
during symbolic execution, including:

- Tracking mutations to objects, attributes, and variables
- Managing context changes (cell variables, global namespace modifications)
- Handling aliasing and object identity preservation
- Managing stack frame state and local variable changes
- Tracking function calls with side effects

Key classes:
- SideEffects: Main container for tracking all side effects during execution
- MutableSideEffects: Specialization for mutable object tracking
- AttributeMutation/ValueMutation: Track specific types of mutations
- Various specialized side effect classes for different scenarios

The side effect system ensures that mutations performed during symbolic execution
are properly replayed during runtime, maintaining the correctness of compiled code
while enabling optimizations where safe.
"""

class SideEffects:
    id_to_variable: dict[int, VariableTracker]
    store_attr_mutations: dict[VariableTracker, dict[str, VariableTracker]]
    keepalive: list[Any]
    def __init__(
        self,
        output_graph: OutputGraph,
        id_to_variable: dict[int, VariableTracker] | None = ...,
        store_attr_mutations: dict[VariableTracker, dict[str, VariableTracker]]
        | None = ...,
        keepalive: list[Any] | None = ...,
        save_for_backward: list[
            tuple[AutogradFunctionContextVariable, list[VariableTracker]]
        ]
        | None = ...,
        tensor_hooks: dict[
            int,
            tuple[
                variables.TensorVariable,
                VariableTracker,
                variables.RemovableHandleVariable,
                str,
            ],
        ]
        | None = ...,
    ) -> None: ...
    def ignore_mutations_on(self, var: VariableTracker) -> None: ...
    def stop_ignoring_mutations_on(self, var: VariableTracker) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def diff(self, other: SideEffects) -> str | None: ...
    def clone(self) -> SideEffects: ...
    def __contains__(self, item: Any) -> bool: ...
    def __getitem__(self, item: Any) -> VariableTracker: ...
    def should_allow_side_effects_under_checkpoint(self) -> bool: ...
    def should_allow_externally_visible_side_effects_in_subtracer(self) -> bool: ...
    def is_reconstructing_generator(self) -> bool: ...
    def check_allowed_side_effect(self, item: VariableTracker) -> bool: ...
    def store_attr(
        self, item: VariableTracker, name: str, value: VariableTracker
    ) -> None: ...
    def load_attr(
        self,
        item: VariableTracker,
        name: str,
        deleted_ok: bool = ...,
        check: bool = ...,
    ) -> VariableTracker: ...
    def store_cell(self, cellvar: VariableTracker, value: VariableTracker) -> None: ...
    def load_cell(self, cellvar: VariableTracker) -> VariableTracker: ...
    def load_global(self, gvar: VariableTracker, name: str) -> VariableTracker: ...
    def store_global(
        self, gvar: VariableTracker, name: str, value: VariableTracker
    ) -> None: ...
    @staticmethod
    def cls_supports_mutation_side_effects(cls: type) -> bool: ...
    def is_attribute_mutation(self, item: VariableTracker) -> bool: ...
    def has_pending_mutation(self, item: VariableTracker) -> bool: ...
    def has_pending_mutation_of_attr(
        self, item: VariableTracker, name: str
    ) -> bool: ...
    def is_modified(self, item: VariableTracker) -> bool: ...

    track_mutable = ...
    def track_object_existing(
        self, item: Any, variable: VariableTracker
    ) -> VariableTracker: ...
    def track_object_new(
        self,
        cls_source: Source,
        user_cls: Any,
        variable_cls: Any,
        options: dict[str, Any],
    ) -> VariableTracker: ...
    def get_variable_cls(self, user_cls: type) -> type: ...
    def get_example_value(
        self,
        base_cls_vt: VariableTracker,
        cls_vt: VariableTracker,
        init_args: list[VariableTracker],
    ) -> Any: ...
    def track_new_user_defined_object(
        self,
        base_cls_vt: VariableTracker,
        cls_vt: VariableTracker,
        init_args: list[VariableTracker],
    ) -> VariableTracker: ...
    def track_cell_new(self) -> VariableTracker: ...
    def track_cell_existing(
        self, source: Source | None, cell: CellType, contents: VariableTracker
    ) -> VariableTracker: ...
    def track_global_existing(self, source: Source, item: Any) -> VariableTracker: ...
    def track_save_for_backward(
        self, ctx: VariableTracker, args: list[VariableTracker]
    ) -> None: ...
    def track_runahead_tensor_and_symvar_side_effects(
        self, other: SideEffects
    ) -> None: ...
    def prune_dead_object_new(self, tx: InstructionTranslatorBase) -> None: ...
    def mutation(self, var: VariableTracker) -> None: ...
    def has_existing_dict_mutation(self) -> bool: ...
    def codegen_save_tempvars(self, cg: PyCodegen) -> None: ...
    def register_hook(
        self,
        tensor: variables.TensorVariable,
        hook: VariableTracker,
        handle: variables.RemovableHandleVariable,
        name: str,
    ) -> None: ...
    def remove_hook(self, idx: int) -> None: ...
    def codegen_hooks(self, cg: PyCodegen) -> None: ...
    def get_ca_final_callbacks_var(self) -> variables.ListVariable: ...
    def codegen_update_mutated(self, cg: PyCodegen) -> None: ...
    def is_empty(self) -> bool: ...
    def clear(self) -> None: ...

@contextlib.contextmanager
def allow_side_effects_under_checkpoint(
    tx: InstructionTranslatorBase,
) -> Generator[None]: ...
@contextlib.contextmanager
def allow_externally_visible_side_effects_in_subtracer(
    tx: InstructionTranslatorBase,
) -> Generator[None]: ...
@contextlib.contextmanager
def disallow_side_effects_in_generator(
    tx: InstructionTranslatorBase,
) -> Generator[None]: ...
