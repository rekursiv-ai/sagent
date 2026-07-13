from typing import Any

from torch.ao.quantization.backend_config import BackendConfig
from torch.fx import GraphModule

from .custom_config import FuseCustomConfig

__all__ = ["FuseHandler", "fuse"]

def fuse(
    model: GraphModule,
    is_qat: bool,
    fuse_custom_config: FuseCustomConfig | dict[str, Any] | None = ...,
    backend_config: BackendConfig | dict[str, Any] | None = ...,
) -> GraphModule: ...
