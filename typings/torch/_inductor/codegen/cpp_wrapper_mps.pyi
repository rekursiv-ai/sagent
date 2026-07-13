from .cpp_wrapper_gpu import CppWrapperGpu
from .wrapper import PythonWrapperCodegen
from ..ir import GraphPartitionSignature

class CppWrapperMps(CppWrapperGpu):
    def __init__(self) -> None: ...
    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: GraphPartitionSignature | None = ...,
    ) -> CppWrapperMps: ...
    def write_mps_kernel_call(self, name: str, call_args: list[str]) -> None: ...
    @staticmethod
    def get_device_include_path(device: str) -> str: ...
    def codegen_additional_funcs(self) -> None: ...
