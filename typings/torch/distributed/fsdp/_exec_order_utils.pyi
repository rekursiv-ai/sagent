from enum import Enum

from torch import nn
from torch.distributed.fsdp._common_utils import _FSDPState
from torch.distributed.fsdp._flat_param import FlatParamHandle

import torch.distributed as dist

class _ExecOrderWarnStatus(Enum):
    NONE = ...
    WARNING = ...
    WARNED = ...

class _ExecOrderData:
    def __init__(
        self,
        debug_level: dist.DebugLevel,
        backward_prefetch_limit: int,
        forward_prefetch_limit: int,
    ) -> None: ...
    def init(
        self,
        state: _FSDPState,
        root_module: nn.Module,
        process_group: dist.ProcessGroup,
    ) -> None: ...
    @property
    def is_first_iter(self) -> bool: ...
    def get_handle_to_backward_prefetch(
        self, current_handle: FlatParamHandle
    ) -> FlatParamHandle | None: ...
    def get_handle_to_forward_prefetch(
        self, current_handle: FlatParamHandle
    ) -> FlatParamHandle | None: ...
    def record_post_forward(self, handle: FlatParamHandle | None) -> None: ...
    def record_pre_forward(
        self, handle: FlatParamHandle | None, is_training: bool
    ) -> None: ...
    def next_iter(self) -> None: ...
