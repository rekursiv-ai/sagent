from typing import override

from torch._inductor import ir

from .wrapper import PythonWrapperCodegen

class PythonWrapperMtia(PythonWrapperCodegen):
    @override
    def write_header(self) -> None: ...
    @override
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = ...,
    ) -> PythonWrapperCodegen: ...
