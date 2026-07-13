from collections.abc import Sequence

from torch.utils._ordered_set import OrderedSet

from ..common import BackendFeature
from ...scheduler import BaseSchedulerNode, BaseScheduling

log = ...

class CuteDSLScheduling(BaseScheduling):
    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]: ...
    @staticmethod
    def is_cutedsl_template(node: BaseSchedulerNode) -> bool: ...
    def is_cutedsl_fused_template(self, node: BaseSchedulerNode) -> bool: ...
    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool: ...
    def define_kernel(self, src_code_str: str, node_schedule) -> str: ...
    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ) -> None: ...
