from collections.abc import Callable
from typing import Any, Literal

from .cpp_gemm_template import CppGemmTemplate
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import GemmBlocking
from .. import ir

GEMM_SINGLE_THREAD_MM_STUB = ...
GEMM_THREADED_MM_STUB = ...
BMM_TEMPLATE = ...

class CppBmmTemplate(CppGemmTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta=...,
        alpha=...,
        has_bias=...,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = ...,
        should_block_weights: bool = ...,
        name=...,
    ) -> None: ...
    @staticmethod
    def get_padded_size(
        n, block_n, k, should_block_weight
    ) -> tuple[list[Any], Any] | tuple[list[int | Any], Any]: ...
    @staticmethod
    def check_if_block_weight(W, micro_gemm) -> Literal[True]: ...
    def get_gemm_function_call(
        self,
        kernel: CppTemplateKernel,
        function_name: str,
        placeholder: str,
        b_index: str,
    ) -> str: ...
    def get_default_reindexers(self, epilogue_nodes) -> list[Callable[..., Any]]: ...
    def get_options(
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: ir.CppTemplateBuffer | None = ...,
        flag_template_buffer_has_other_users: bool | None = ...,
        epilogue_nodes: list[ir.IRNode] | None = ...,
        **kwargs,
    ) -> dict[str, Any]: ...
    def render(
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: ir.CppTemplateBuffer | None = ...,
        flag_template_buffer_has_other_users: bool | None = ...,
        epilogue_nodes: list[ir.IRNode] | None = ...,
        **kwargs,
    ) -> str: ...
    def codegen_single_thread_gemm(self) -> Any: ...
    def codegen_multi_thread_gemm(self) -> Any: ...
    def codegen_gemm_stub_def(self) -> Literal[""]: ...
