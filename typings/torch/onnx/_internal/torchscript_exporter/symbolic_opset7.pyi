from torch.onnx._internal.torchscript_exporter import jit_utils

"""
Note [ONNX operators that are added/updated from opset 7 to opset 8]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
  Expand

Updated operators:
  Min, Max, Sum, Mean: supports multidirectional broadcasting.
  MaxPool: added optional indices output.
  Scan
"""
_onnx_symbolic = ...
block_listed_operators = ...

@_onnx_symbolic("aten::max")
def max(
    g: jit_utils.GraphContext, self, dim_or_y=..., keepdim=...
) -> tuple[Any, Any]: ...
@_onnx_symbolic("aten::min")
def min(
    g: jit_utils.GraphContext, self, dim_or_y=..., keepdim=...
) -> tuple[Any, Any]: ...
