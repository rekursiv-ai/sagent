from collections.abc import Callable
from typing import Any, TypeVar

import torch

from .cpp_micro_gemm import CppMicroGemm
from .cpp_template import CppTemplate
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import GemmBlocking
from .. import ir

log = ...
GEMM_TEMPLATE_INIT_BLOCKING_BASIC_BLOCK = ...
GEMM_TEMPLATE_INIT_BLOCKING_EXTENDED = ...
GEMM_TEMPLATE_MULTI_THREADS_PARAMS = ...
GEMM_TEMPLATE_SINGLE_THREAD_PARAMS = ...
GEMM_TEMPLATE_M_LOOP_PARAMS = ...
GEMM_TEMPLATE_N_LOOP_PARAMS = ...
GEMM_TEMPLATE_MICROKERNEL_DEF = ...
GEMM_TEMPLATE_STUB_DEF = ...
GEMM_TEMPLATE = ...
SMALL_M_GEMM_TEMPLATE = ...

def get_padded_n(n, block_n): ...

_T = TypeVar("_T", ir.IRNode, torch.Tensor)

def transpose_w(W: _T, trans_w: bool) -> _T: ...
def expand_bias(B: _T | None, X: _T) -> _T | None: ...
def prune_tensors(
    input_nodes: list[ir.IRNode], new_input_nodes: list[ir.IRNode]
) -> None: ...
def gen_2d_view_of_epilogue_buf(
    Y: ir.Buffer,
    template_buffer: ir.Buffer,
    epilogue_nodes: list[ir.IRNode],
    reindexers: list[Callable[[list[Any]], list[Any]] | None],
    default_reindexers: list[Callable[[list[Any]], list[Any]] | None],
) -> tuple[
    ir.Buffer | ir.ReinterpretView, list[Callable[[list[Any]], list[Any]] | None]
]: ...

class CppGemmTemplate(CppTemplate):
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
    def make_thread_blocking_cache(self) -> Callable[..., GemmBlocking]: ...
    def make_cache_blocking_cache(self) -> Callable[..., GemmBlocking]: ...
    def log_blockings(self) -> None: ...
    def maybe_k_slicing(self) -> bool: ...
    @classmethod
    def add_choices(
        cls,
        choices,
        layout,
        input_nodes,
        beta=...,
        alpha=...,
        has_bias=...,
        trans_w=...,
        input_indices=...,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = ...,
        act_mapping: dict[int, ir.IRNode] | None = ...,
    ) -> DataProcessorTemplateWrapper: ...
    @staticmethod
    def get_padded_size(
        n, block_n, k, should_block_weight
    ) -> tuple[list[Any], Any]: ...
    @classmethod
    def prep_weight(
        cls,
        inputs,
        layout: ir.Layout,
        micro_gemm: CppMicroGemm,
        should_block_weight: bool,
        use_int8_fast_compensation_path: bool = ...,
        skip_int8_compensation: bool = ...,
    ) -> tuple[list[Any], Layout]: ...
    @staticmethod
    def check_if_block_weight(W, micro_gemm) -> Literal[True]: ...
    @classmethod
    def block_weight(cls, W, new_size, padding) -> Any | TensorBox | Tensor: ...
    @classmethod
    def pack_vnni_weight(cls, W, micro_gemm, new_size) -> IRNode | Any: ...
    def get_default_reindexers(self, epilogue_nodes) -> list[None]: ...
    def get_options(
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: ir.CppTemplateBuffer | None = ...,
        flag_template_buffer_has_other_users: bool | None = ...,
        epilogue_nodes: list[ir.IRNode] | None = ...,
    ) -> dict[str, Any]: ...
    def is_int8_woq_gemm_small_m_dim(
        self, X: ir.ReinterpretView, W: ir.ReinterpretView, N, K, micro_gemm
    ) -> bool: ...
    def render(
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: ir.CppTemplateBuffer | None = ...,
        flag_template_buffer_has_other_users: bool | None = ...,
        epilogue_nodes: list[ir.IRNode] | None = ...,
        **kwargs,
    ) -> str: ...
    def codegen_blocks(
        self,
        num_threads,
        N,
        K,
        micro_gemm,
        is_dynamic_M,
        kernel,
        GemmOut,
        config,
        L1_cache_size,
        L2_cache_size,
        X,
        W,
    ) -> Any: ...
    def codegen_microkernel_def(self) -> Any: ...
    def codegen_gemm_stub_def(self) -> Any: ...
    def codegen_multi_threads_params(self) -> Any: ...
    def codegen_single_thread_params(self, is_dynamic_M) -> Any: ...
    def codegen_m_loop_params(self) -> Any: ...
    def codegen_n_loop_params(self) -> Any: ...
    @classmethod
    def is_woq_int4(cls) -> Literal[False]: ...
    @classmethod
    def q_group_size(cls) -> None: ...

class CppWoqInt4GemmTemplateMeta(type):
    def __getitem__(cls, q_group_size) -> type[CppWoqInt4GemmTemplateInstance]: ...

class CppWoqInt4GemmTemplate(metaclass=CppWoqInt4GemmTemplateMeta): ...
