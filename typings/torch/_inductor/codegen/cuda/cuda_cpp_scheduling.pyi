from typing import Any
from collections.abc import Sequence

from torch.utils._ordered_set import OrderedSet

from ..common import BackendFeature
from ...scheduler import BaseSchedulerNode, BaseScheduling, WhyNoFuse

log = ...

class WhyNoFuseNames(WhyNoFuse):
    def __init__(self, name1: str, name2: str) -> None: ...

class CUDACPPScheduling(BaseScheduling):
    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]: ...
    def group_fn(self, sizes) -> tuple[Any, ...]: ...
    @staticmethod
    def is_cuda_cpp_template(node: BaseSchedulerNode) -> bool: ...
    def is_cuda_cpp_fused_template(self, node: BaseSchedulerNode) -> bool: ...
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
