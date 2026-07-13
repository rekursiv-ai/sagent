from collections.abc import Callable

from .cpp_gemm_template import CppGemmTemplate
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import GemmBlocking
from .. import ir
from ..select_algorithm import ChoiceCaller, DataProcessorTemplateWrapper

log = ...
GEMM_TEMPLATE = ...

def get_deduplicated_act(act_mapping: dict[int, ir.IRNode]) -> list[ir.IRNode]: ...

class CppGroupedGemmTemplate(CppGemmTemplate):
    def __init__(
        self,
        input_nodes: list[ir.IRNode],
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta: int = ...,
        alpha: int = ...,
        has_bias: bool = ...,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = ...,
        act_mapping: dict[int, ir.IRNode] | None = ...,
        gemm_grouped_num: int = ...,
    ) -> None: ...
    @classmethod
    def add_choices(
        cls,
        choices: list[ChoiceCaller],
        layout: ir.Layout,
        input_nodes: list[ir.IRNode],
        beta: int = ...,
        alpha: int = ...,
        has_bias: tuple[bool, ...] = ...,
        trans_w: bool = ...,
        input_indices: list[int] | None = ...,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = ...,
        act_mapping: dict[int, ir.IRNode] | None = ...,
    ) -> DataProcessorTemplateWrapper: ...
    def render(
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: ir.CppTemplateBuffer | None = ...,
        flag_template_buffer_has_other_users: bool | None = ...,
        epilogue_nodes: list[ir.IRNode] | None = ...,
        **kwargs,
    ) -> str: ...
