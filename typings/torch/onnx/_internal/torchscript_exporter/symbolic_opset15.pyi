from torch.onnx._internal.torchscript_exporter import (
    jit_utils,
    symbolic_opset9 as opset9,
)

"""This file exports ONNX ops for opset 15.

Note [ONNX operators that are added/updated in opset 15]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/master/docs/Changelog.md#version-15-of-the-default-onnx-operator-set
New operators:
    Bernoulli
    CastLike
    Optional
    OptionalGetElement
    OptionalHasElement

Updated operators:
    BatchNormalization https://github.com/onnx/onnx/pull/3545
                        Backwards compatible
                        TODO: test coverage for mixed types inputs.
    Pow                https://github.com/onnx/onnx/pull/3412
                        Backwards compatible
                        TODO: bfloat16 support.
    Shape              https://github.com/onnx/onnx/pull/3580
                        Backwards compatible
                        TODO: optional start/end attribute.
"""
_onnx_symbolic = ...

@_onnx_symbolic("aten::__is_")
def aten__is_(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::__isnot_")
@opset9.wrap_logical_op_with_negation
def aten__isnot_(g: jit_utils.GraphContext, self, other): ...
@_onnx_symbolic("aten::bernoulli")
def bernoulli(
    g: jit_utils.GraphContext, input, p=..., generator=..., out=...
) -> None: ...
@_onnx_symbolic("prim::unchecked_cast")
def prim_unchecked_cast(g: jit_utils.GraphContext, self): ...
