from collections.abc import Callable

from .common import KernelTemplate
from .. import ir
from ..utils import IndentedBuffer

log = ...

class CppTemplate(KernelTemplate):
    index_counter = ...
    def __init__(
        self,
        name: str,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        epilogue_creator: Callable[[ir.Buffer], ir.Pointwise] | None = ...,
    ) -> None: ...
    def generate(self, **kwargs) -> CppTemplateCaller: ...
    def header(self) -> IndentedBuffer: ...
    def render(self, **kwargs) -> str: ...
