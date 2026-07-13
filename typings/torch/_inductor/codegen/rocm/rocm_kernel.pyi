from collections.abc import Callable, Sequence
from typing import Any

from torch._inductor.codegen.rocm.rocm_template import ArgInfo, ROCmTemplate

from .rocm_benchmark_request import ROCmBenchmarkRequest
from .rocm_template_buffer import ROCmTemplateBuffer
from ..common import Kernel, OpOverrides
from ...ir import (
    Buffer,
    ChoiceCaller,
    IRNode,
    Layout,
    PrimitiveInfoType,
    ShapeAsConstantBuffer,
    TensorBox,
)

log = ...
cexpr = ...

class ROCmKernel(Kernel):
    overrides = OpOverrides

class ROCmTemplateKernel(ROCmKernel):
    _EXTRA_CPP_ARGS = ...
    def __init__(
        self,
        kernel_name: str,
        runtime_arg_info: list[ArgInfo],
        runtime_arg_values: list[Any],
    ) -> None: ...
    def get_signature(self) -> str: ...
    def def_kernel(
        self,
        inputs: list[IRNode],
        outputs: list[IRNode],
        size_args: list[str],
        names_str: str = ...,
        input_reorder: list[int] | None = ...,
    ) -> str: ...
    def call_kernel(self, name: str, node: ROCmTemplateBuffer) -> None: ...

class ROCmTemplateCaller(ChoiceCaller):
    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: list[Buffer],
        layout: Layout,
        make_kernel_render: Callable[
            [ROCmTemplateBuffer, Sequence[IRNode] | None], str
        ],
        bmreq: ROCmBenchmarkRequest,
        template: ROCmTemplate,
        info_kwargs: dict[str, PrimitiveInfoType | list[PrimitiveInfoType]] | None,
    ) -> None: ...
    def precompile(self) -> None: ...
    def benchmark(self, *args, out) -> float: ...
    def call_name(self) -> str: ...
    def hash_key(self) -> str: ...
    def info_dict(self) -> dict[str, PrimitiveInfoType | list[PrimitiveInfoType]]: ...
    def output_node(self) -> TensorBox | ShapeAsConstantBuffer: ...
