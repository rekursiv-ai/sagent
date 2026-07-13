from collections.abc import Callable, Sequence

from onnxscript import ir
from torch.onnx._internal.exporter import _registration

import torch
import torch.fx

logger = ...
_TORCH_DTYPE_TO_ONNX_COMPATIBLE: dict[torch.dtype, ir.DataType] = ...

def get_matching_overload(
    node: torch.fx.Node, overloads: Sequence[_registration.OnnxDecompMeta]
) -> tuple[Callable | None, str]: ...
def dispatch(
    node: torch.fx.Node, registry: _registration.ONNXRegistry
) -> tuple[Callable | None, str]: ...
