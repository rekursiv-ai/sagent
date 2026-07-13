from collections.abc import Mapping, Sequence
from typing import Any

from onnxscript import evaluator, ir
from torch.onnx._internal.exporter import _tensors

import onnx
import onnxscript

"""NOTES:

We need a typing module that will handling Python to ONNX type promotion for use.
For example, if we have torch.ops.aten.add(Tensor, 1.0), we need to promote 1.0
to the same type as Tensor. The same thing needs to work for
torch.ops.aten.add(1.0, Tensor) as well, which means we need a mechanism to`
"""
logger = ...
type ValidAttributeType = (
    ir.TensorProtocol
    | int
    | float
    | bool
    | str
    | Sequence[int]
    | Sequence[float]
    | None
)
type AllowedArgType = (
    ir.Value | Sequence[ir.Value | ValidAttributeType] | ValidAttributeType
)

class OpRecorder(evaluator.Evaluator):
    def __init__(
        self, opset: onnxscript.values.Opset, constant_farm: dict[Any, ir.Value]
    ) -> None: ...
    def eval(
        self,
        schema: onnx.defs.OpSchema,
        args: Sequence[AllowedArgType],
        kwargs: Mapping[str, AllowedArgType],
    ) -> _tensors.SymbolicTensor | Sequence[_tensors.SymbolicTensor]: ...
    def eval_function(
        self,
        function: onnxscript.OnnxFunction,
        args: Sequence[AllowedArgType],
        kwargs: Mapping[str, AllowedArgType],
    ) -> _tensors.SymbolicTensor | Sequence[_tensors.SymbolicTensor] | bool | int: ...
