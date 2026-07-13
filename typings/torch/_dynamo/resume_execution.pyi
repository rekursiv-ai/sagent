from typing import Any

import dataclasses
import types

from .bytecode_transformation import Instruction

"""
This module provides functionality for resuming Python execution at specific points in code,
primarily used by PyTorch Dynamo for control flow handling and optimization. It implements
bytecode transformation and execution state management to enable:

- Resuming execution at arbitrary points in Python bytecode
- Managing context managers and their state across execution boundaries
- Transforming and generating new code objects with preserved execution state
- Supporting Python 3.11+ exception handling and block management
- Restoring torch function mode stacks and other execution context

The module is critical for PyTorch Dynamo's ability to optimize code while preserving
Python semantics and execution state.
"""
CO_OPTIMIZED = ...
CO_NEWLOCALS = ...
CO_VARARGS = ...
CO_VARKEYWORDS = ...
CO_NESTED = ...
CO_GENERATOR = ...
CO_NOFREE = ...
CO_COROUTINE = ...
CO_ITERABLE_COROUTINE = ...
CO_ASYNC_GENERATOR = ...
TORCH_DYNAMO_RESUME_IN_PREFIX = ...
IS_TRACING_RESUME_PROLOGUE_VARNAME = ...

@dataclasses.dataclass(frozen=True)
class ReenterWith:
    stack_index: int
    target_values: tuple[Any, ...] | None = ...
    def try_except_torch_function_mode(
        self, code_options: dict[str, Any], cleanup: list[Instruction]
    ) -> list[Instruction]: ...
    def try_finally(
        self, code_options: dict[str, Any], cleanup: list[Instruction]
    ) -> list[Instruction]: ...
    def __call__(
        self, code_options: dict[str, Any], cleanup: list[Instruction]
    ) -> tuple[list[Instruction], Instruction | None]: ...

@dataclasses.dataclass
class ResumeFunctionMetadata:
    code: types.CodeType
    instructions: list[Instruction] = ...
    prefix_block_target_offset_remap: list[int] = ...
    block_target_offset_remap: dict[int, dict[int, int]] = ...

class ContinueExecutionCache:
    cache = ...
    generated_code_metadata = ...
    @classmethod
    def lookup(cls, code: types.CodeType, lineno: int, *key: Any) -> types.CodeType: ...
    @classmethod
    def generate(
        cls,
        code: types.CodeType,
        lineno: int,
        offset: int,
        setup_fn_target_offsets: tuple[int, ...],
        nstack: int,
        argnames: tuple[str, ...],
        argnames_null: tuple[str, ...],
        setup_fns: tuple[ReenterWith, ...],
        stack_ctx_vars: tuple[tuple[int, tuple[Any, ...]], ...],
        argnames_ctx_vars: tuple[tuple[str, tuple[Any, ...]], ...],
        null_idxes: tuple[int, ...],
        nested_code_objs: tuple[types.CodeType],
    ) -> types.CodeType: ...
    @staticmethod
    def unreachable_codes(code_options: dict[str, Any]) -> list[Instruction]: ...
    @classmethod
    def generate_based_on_original_code_object(
        cls,
        code: types.CodeType,
        lineno: int,
        offset: int,
        setup_fn_target_offsets: tuple[int, ...],
        *args: Any,
    ) -> types.CodeType: ...
