from torch._export.serde.schema import ExternKernelNode
from torch._inductor.ir import ExternKernelNode as inductor_ExternKernelNode

def serialize_extern_kernel_node(
    extern_kernel_node: inductor_ExternKernelNode,
) -> ExternKernelNode: ...
def extern_node_json_serializer(
    extern_kernel_nodes: list[inductor_ExternKernelNode],
) -> str: ...
