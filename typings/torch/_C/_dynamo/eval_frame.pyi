from typing import overload

import enum
import types

from torch._dynamo.guards import GuardManagerWrapper
from torch._dynamo.types import DynamoCallback, DynamoGuardCompleteHook, DynamoGuardHook
from torch._guards import CompileId

def set_eval_frame(callback: DynamoCallback) -> DynamoCallback: ...
def set_skip_guard_eval_unsafe(value: bool) -> bool: ...
def get_eval_frame_callback() -> DynamoCallback: ...
def reset_code(code: types.CodeType) -> None: ...
def unsupported(obj1: object, obj2: object) -> object: ...
def set_code_exec_strategy(
    code: types.CodeType, strategy: _FrameExecStrategy
) -> None: ...
def set_guard_error_hook(hook: DynamoGuardHook) -> None: ...
def set_guard_complete_hook(
    hook: DynamoGuardCompleteHook | None,
) -> DynamoGuardCompleteHook | None: ...
def raise_sigtrap() -> None: ...

class _CacheEntry:
    def check_fn(self, *args: object, **kwargs: object) -> bool: ...
    def update_diff_guard_root_manager(self) -> None: ...

    code: types.CodeType
    compile_id: CompileId
    guard_manager: GuardManagerWrapper
    next: _CacheEntry | None

class _PrecompileEntry:
    guard_manager: GuardManagerWrapper

class _ExtraState:
    def invalidate(
        self, cache_entry: _CacheEntry, guard_manager: GuardManagerWrapper
    ) -> None: ...

class _FrameAction(enum.IntEnum):
    DEFAULT = ...
    SKIP = ...
    RUN_ONLY = ...

class _FrameExecStrategy:
    cur_action: _FrameAction
    recursive_action: _FrameAction
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self, cur_action: _FrameAction, recursive_action: _FrameAction
    ) -> None: ...

class _PyInterpreterFrame:
    f_code: types.CodeType
    f_locals: dict[str, object]
    f_globals: dict[str, object]
    f_builtins: dict[str, object]
    f_lasti: int
    f_lineno: int
    f_back: types.FrameType
    closure: tuple[types.CellType]

py_opcode_caches: list[int]

def code_framelocals_names(code: types.CodeType) -> tuple[str]: ...
