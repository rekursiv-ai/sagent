from .cpp_template import CppTemplate
from .. import ir

log = ...
SOFTMAX_FUSIONS = ...
BRGEMM_PACK_FUNCTIONS = ...
MICRO_GEMM_TEMPLATE = ...
ALLOCATE_BUFFER = ...
FLEX_ATTENTION_TEMPLATE = ...

class CppFlexAttentionTemplate(CppTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        scale,
        score_mod,
        mask_mod,
        kv_block_size,
        q_block_size,
        has_other_buffer,
        no_full_kv_block,
        fake_buffers,
        len_score_other,
        len_mask_other,
        kernel_input_name_to_buffer,
        block_vars,
    ) -> None: ...
    def update_kernel_args(self, kernel_args): ...
    def generate_other_buffer(
        self, buf_list, start_offset, len_attr, kernel_args
    ) -> str: ...
    def modification(self, subgraph_buffer, output_name, output_idx) -> str: ...
    @staticmethod
    def add_choices(
        choices,
        input_nodes,
        layout,
        scale,
        score_mod,
        mask_mod,
        kv_block_size,
        q_block_size,
        has_other_buffer,
        no_full_kv_block,
        fake_buffers,
        len_score_other,
        len_mask_other,
        kernel_input_name_to_buffer,
        block_vars,
    ) -> DataProcessorTemplateWrapper: ...
    def apply_score_mod(self, score, b, h, q_idx, kv_idx): ...
    def render(
        self,
        kernel,
        template_buffer_node: ir.CppTemplateBuffer | None = ...,
        epilogue_nodes: list[ir.IRNode] | None = ...,
        **kwargs,
    ) -> str: ...
    def codegen_softmax_fusion(self, kernel_name: str) -> Any: ...
    def codegen_brgemm_pack_function(self, kernel_name: str) -> Any: ...
    def codegen_allocate_buffer(
        self, buffer_name: str, buffer_dtype, buffer_size
    ) -> Any: ...
    def micro_gemm_define(self, kernel_name: str) -> str: ...
    def codegen_micro_gemm(self, kernel_name: str) -> Any: ...
