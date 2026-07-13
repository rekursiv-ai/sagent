from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import enum
import threading

"""
This module provides callback management functionality for TorchDynamo's compilation process.

It implements a thread-safe system for registering, managing and executing callbacks that run
at the start and end of TorchDynamo compilations. Key features include:

- Registration and deregistration of compilation callbacks
- Thread-safe callback handling with proper locking mechanisms
- Prevention of duplicate callback execution when configured
- Decorator utilities for easy callback registration
- Context manager for controlled callback lifecycle

The module centers around the CompilationCallbackHandler class which maintains separate
lists for start and end callbacks, manages their execution order, and ensures thread-safety.
Utility decorators @on_compile_start and @on_compile_end provide a convenient way to
register compilation hooks.

Example usage:
    @on_compile_start
    def my_start_callback():
        print("Starting compilation")

    @on_compile_end
    def my_end_callback():
        print("Compilation complete")
"""

class CallbackTrigger(enum.Enum):
    DYNAMO = ...
    LAZY_BACKWARD = ...
    TRITON_AUTOTUNING = ...
    CUDAGRAPH_RECORDING = ...

@dataclass
class CallbackArgs:
    callback_trigger: CallbackTrigger
    compile_id: str

@dataclass
class CompilationCallbackHandler:
    start_callbacks: list[Callable[[CallbackArgs], None]] = ...
    end_callbacks: list[Callable[[CallbackArgs], None]] = ...
    __pending_callbacks_counter: int = ...
    __pending_callbacks_counter_lock: threading.Lock = ...
    def register_start_callback(
        self, callback: Callable[[CallbackArgs], None]
    ) -> Callable[[CallbackArgs], None]: ...
    def register_end_callback(
        self, callback: Callable[[CallbackArgs], None]
    ) -> Callable[[CallbackArgs], None]: ...
    def remove_start_callback(
        self, callback: Callable[[CallbackArgs], None]
    ) -> None: ...
    def remove_end_callback(self, callback: Callable[[CallbackArgs], None]) -> None: ...
    def run_start_callbacks(self, args: CallbackArgs) -> None: ...
    def run_end_callbacks(self, args: CallbackArgs) -> None: ...
    @contextmanager
    def install_callbacks(
        self, trigger: CallbackTrigger, compile_id: str
    ) -> Generator[None, Any, Any]: ...
    def clear(self) -> None: ...

callback_handler = ...

def on_compile_start(
    callback: Callable[[CallbackArgs], None],
) -> Callable[[CallbackArgs], None]: ...
def on_compile_end(
    callback: Callable[[CallbackArgs], None],
) -> Callable[[CallbackArgs], None]: ...
