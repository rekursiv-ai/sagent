from typing import Any
from collections.abc import Sequence

from ...scheduler import BaseSchedulerNode, BaseScheduling

log = ...

class ROCmCPPScheduling(BaseScheduling):
    def group_fn(self, sizes) -> tuple[Any, ...]: ...
    @staticmethod
    def is_rocm_cpp_template(node: BaseSchedulerNode) -> bool: ...
    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool: ...
    def define_kernel(self, src_code: str, node_schedule) -> str: ...
    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ) -> None: ...
