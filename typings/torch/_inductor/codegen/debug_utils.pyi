from collections.abc import Callable
from enum import Enum

log = ...

class IntermediateValueDebuggingLevel(Enum):
    OFF = ...
    SAVE_ONLY = ...
    PRINT_ONLY = ...
    PRINT_KERNEL_NAMES_ONLY = ...

class DebugPrinterManager:
    def __init__(
        self,
        debug_printer_level,
        use_array_ref: bool,
        writeline: Callable[..., None] | None = ...,
        args_to_print_or_save: list[str] | None = ...,
        kernel_name: str = ...,
        kernel=...,
        arg_signatures: list[type] | None = ...,
        kernel_type=...,
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, args_to_print_or_save, kernel_name, arg_signatures) -> None: ...
    def set_printer_args(
        self,
        args_to_print_or_save: list[str],
        kernel_name: str,
        arg_signatures: list[type] | None,
        kernel,
        kernel_type=...,
    ) -> None: ...
    def codegen_model_inputs_value_print(
        self, input_args_to_print: list[str]
    ) -> None: ...
    def codegen_intermediate_tensor_value_save(
        self,
        args_to_save,
        kernel_name,
        before_launch=...,
        arg_signatures: list[type] | None = ...,
    ) -> None: ...
    def codegen_intermediate_tensor_value_print(
        self,
        args_to_print,
        kernel_name,
        before_launch=...,
        arg_signatures: list[type] | None = ...,
    ) -> None: ...
