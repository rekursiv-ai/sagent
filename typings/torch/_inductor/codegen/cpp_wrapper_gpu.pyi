from typing import Any, Self

import dataclasses

from torch import dtype as torch_dtype

from .cpp_wrapper_cpu import CppWrapperCpu
from .wrapper import PythonWrapperCodegen
from ..ir import GraphPartitionSignature
from ..utils import IndentedBuffer, cache_on_self

_cpp_string_literal_escapes = ...
_cpp_string_literal_pattern = ...

def cpp_string_literal(s: str) -> str: ...

@dataclasses.dataclass
class DeferredTritonCallWrapper:
    wrapper_name: str
    kernel_name: str
    kernel_name_to_body: dict[str, str]
    arg_types: list[Any]
    def generate(self, wrapper: CppWrapperGpu) -> None: ...
    def generate_grid(
        self,
        prefix: IndentedBuffer,
        inductor_meta: dict[str, Any],
        params: dict[str, Any],
    ) -> None: ...
    def generate_load_kernel(self, prefix, kernel_var_name, params) -> None: ...
    def generate_launch_kernel(
        self, prefix, wrapper, kernel_var_name, params
    ) -> None: ...

class CppWrapperGpu(CppWrapperCpu):
    def __init__(self) -> None: ...
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: GraphPartitionSignature | None = ...,
    ) -> CppWrapperGpu: ...
    def write_header(self) -> None: ...
    @cache_on_self
    def write_tma_descriptor_helpers_once(self) -> None: ...
    def write_get_raw_stream(self, device_idx: int, graph_name: str) -> str: ...
    def get_autotuning_input_name(self, idx) -> str: ...
    def codegen_inputs(self) -> None: ...
    def generate(self, is_inference) -> tuple[ValueWithLineMap, ValueWithLineMap]: ...
    def finalize_prefix(self) -> None: ...
    def generate_tma_descriptor(self, desc) -> None: ...
    def generate_args_decl(
        self,
        code: IndentedBuffer | Self,
        call_args,
        arg_types,
        arg_signatures,
        is_triton_kernel=...,
        scratch_spaces: dict[str, int] | None = ...,
    ) -> str: ...
    @staticmethod
    def prepare_triton_wrapper_args(
        call_args: list[Any], arg_types: list[Any]
    ) -> tuple[list[Any], list[Any]]: ...
    def make_zero_buffer(self, name) -> str: ...

@dataclasses.dataclass
class UnwrapUnspecArg:
    dtype: torch_dtype
