from abc import ABC, abstractmethod
from typing import Any

import functools

from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import clear_on_fresh_cache

from .cuda_kernel import CUDATemplateKernel
from .cuda_template import CUTLASSTemplate
from ..common import IndentedBuffer
from ... import ir
from ...ir import (
    Buffer,
    ChoiceCaller,
    CUDATemplateBuffer,
    IRNode,
    Layout,
    ReinterpretView,
)

type GemmOperation = Any
type EVTArgRenames = Any
log = ...
GEMM_TEMPLATE_CUTLASS_3X = ...
GEMM_ARGS_CUTLASS_3X = ...
GEMM_ARGS_CUTLASS_3X_EPILOGUE = ...
GEMM_TEMPLATE_CUTLASS_2X = ...
GEMM_ARGS_CUTLASS_2X = ...
GEMM_ARGS_SPARSE_CUTLASS_2X = ...
GEMM_STANDALONE_RUNNER_ADDITIONAL_INCLUDES = ...
GEMM_STANDALONE_RUNNER_TEMPLATE = ...

@clear_on_fresh_cache
class CUTLASSGemmTemplate(CUTLASSTemplate, ABC):
    filtered_ops_cache: dict[str, list[Any]] = ...
    cache_clear = ...
    def __init__(
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: list[int] | None = ...,
        use_fast_accum: bool | None = ...,
    ) -> None: ...
    @staticmethod
    @abstractmethod
    def add_cutlass_gemm_choices(
        choices: list[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: list[Buffer],
        alpha: float = ...,
        beta: float = ...,
        input_reorder: list[int] | None = ...,
        use_fast_accum: bool | None = ...,
        **extra_kwargs,
    ) -> None: ...
    def header(self) -> IndentedBuffer: ...
    @staticmethod
    def cutlass_layout(torch_layout: ir.Layout) -> cutlass_lib.LayoutType | None: ...
    @staticmethod
    def flip_cutlass_layout(
        cutlass_layout: cutlass_lib.LayoutType,
    ) -> cutlass_lib.LayoutType: ...
    @staticmethod
    @functools.lru_cache(32)
    def layout_match(
        torch_layout: ir.Layout, cutlass_layout: cutlass_lib.LayoutType
    ) -> bool: ...
    @staticmethod
    def set_layout(tensor_desc: TensorDescription, torch_layout: ir.Layout) -> None: ...
    @staticmethod
    def set_alignment(torch_layout, op_element) -> bool: ...
    @staticmethod
    def should_swap_XW(bias: IRNode) -> bool: ...
    @staticmethod
    def swap_XW(
        op: cutlass_library.gemm_op.GemmOperation,
    ) -> cutlass_library.gemm_op.GemmOperation: ...
    def fix_op_layout(
        self,
        op: cutlass_library.gemm_op.GemmOperation,
        X: Buffer,
        W: Buffer,
        Bias: Buffer | None,
        Y: Buffer | ReinterpretView,
    ) -> cutlass_library.gemm_op.GemmOperation: ...
    @classmethod
    def global_filter_ops(
        cls, ops: list[cutlass_library.gemm_op.GemmOperation]
    ) -> list[cutlass_library.gemm_op.GemmOperation]: ...
    def filter_op(
        self, op: cutlass_library.gemm_op.GemmOperation
    ) -> cutlass_library.gemm_op.GemmOperation: ...
    def gen_ops(self) -> list[tuple[str, cutlass_gemm_op.GemmOperation]]: ...
    def gemm_mode(self) -> str: ...
    def render(
        self,
        kernel: CUDATemplateKernel,
        op: cutlass_gemm_op.GemmOperation = ...,
        template_buffer_node: CUDATemplateBuffer | None = ...,
        epilogue_nodes: list[BaseSchedulerNode] | None = ...,
        **kwargs,
    ) -> str: ...
    def test_call_statement(self, kernel, input_nodes, names_str: str = ...) -> str: ...

class CUTLASS3xGemmTemplate(CUTLASSGemmTemplate):
    def __init__(
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: list[int] | None = ...,
        use_fast_accum: bool | None = ...,
    ) -> None: ...
    @staticmethod
    def add_cutlass_gemm_choices(
        choices: list[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: list[Buffer],
        alpha: float = ...,
        beta: float = ...,
        input_reorder: list[int] | None = ...,
        use_fast_accum: bool | None = ...,
        **extra_kwargs,
    ) -> None: ...
    @staticmethod
    def supports_epilogue_fusion(op: GemmOperation) -> bool: ...
    def render_gemm_arguments(
        self,
        argument_template: str,
        epilogue_template: str,
        should_swap_xw: bool,
        X: IRNode,
        W: IRNode,
        Bias: IRNode,
        Y: IRNode,
        alpha: float,
        beta: float,
        kernel: CUDATemplateKernel,
        epilogue_args,
    ) -> str: ...

class CUTLASS2xGemmTemplate(CUTLASSGemmTemplate):
    def __init__(
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        alpha: float,
        beta: float,
        input_reorder: list[int] | None = ...,
    ) -> None: ...
    @staticmethod
    def add_cutlass_gemm_choices(
        choices: list[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: list[Buffer],
        alpha: float = ...,
        beta: float = ...,
        input_reorder: list[int] | None = ...,
        use_fast_accum: bool | None = ...,
        **extra_kwargs,
    ) -> None: ...
    def render_gemm_arguments(
        self,
        instance_type: str,
        argument_template: str,
        epilogue_template: str,
        should_swap_xw: bool,
        X: IRNode,
        W: IRNode,
        Bias: IRNode,
        Meta: IRNode,
        Y: IRNode,
        alpha: float,
        beta: float,
        kernel: CUDATemplateKernel,
        epilogue_args,
    ) -> str: ...
