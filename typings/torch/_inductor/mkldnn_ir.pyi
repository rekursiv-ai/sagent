from collections.abc import Sequence
from typing import Any

from torch.utils._ordered_set import OrderedSet

import sympy

from .ir import ExternKernelAlloc, ShapeAsConstantBuffer, TensorBox

class ConvolutionUnary(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=...) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(
        cls,
        x: TensorBox,
        weight: TensorBox,
        bias: TensorBox,
        padding_: list[int],
        stride_: list[int],
        dilation_: list[int],
        groups: int,
        attr,
        scalars: list[Any] | None,
        algorithm,
    ) -> MultiOutput: ...

class ConvolutionBinary(ExternKernelAlloc):
    def __init__(
        self, layout, inputs, constant_args=..., cpp_constant_args=...
    ) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(
        cls,
        x: TensorBox,
        other: TensorBox,
        weight: TensorBox,
        bias: TensorBox,
        padding_: list[int],
        stride_: list[int],
        dilation_: list[int],
        groups: int,
        binary_attr: str,
        binary_alpha: float | None,
        unary_attr: str | None,
        unary_scalars: list[Any] | None,
        unary_algorithm: str | None,
    ) -> MultiOutput: ...

class ConvolutionBinaryInplace(ExternKernelAlloc):
    def __init__(self, kernel_layout, inputs, constant_args=...) -> None: ...
    def codegen(self, wrapper) -> None: ...
    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]: ...
    @classmethod
    def create(
        cls,
        x: TensorBox,
        other: TensorBox,
        weight: TensorBox,
        bias: TensorBox,
        padding_: list[int],
        stride_: list[int],
        dilation_: list[int],
        groups: int,
        binary_attr: str,
        binary_alpha: float | None,
        unary_attr: str | None,
        unary_scalars: list[Any] | None,
        unary_algorithm: str | None,
    ) -> Any: ...

class ConvolutionTransposeUnary(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=...) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(
        cls,
        x: TensorBox,
        weight: TensorBox,
        bias: TensorBox,
        padding_: list[int],
        output_padding_: list[int],
        stride_: list[int],
        dilation_: list[int],
        groups_: int,
        attr,
        scalars: list[Any] | None,
        algorithm,
    ) -> MultiOutput: ...

class QConvPointWisePT2E(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=...) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(
        cls,
        qx: TensorBox,
        x_scale: ShapeAsConstantBuffer | TensorBox,
        x_zero_point: ShapeAsConstantBuffer | TensorBox,
        qw: TensorBox,
        w_scale: TensorBox,
        w_zero_point: TensorBox,
        bias: TensorBox,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
        output_scale: float,
        output_zero_point: int,
        output_dtype,
        attr,
        scalars,
        algorithm,
    ) -> QConvPointWisePT2E: ...

class QConvPointWiseBinaryPT2E(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=...) -> None: ...
    def codegen(self, wrapper) -> None: ...
    def get_mutation_names(self) -> Sequence[str]: ...
    def get_unbacked_symbol_defs(self) -> OrderedSet[sympy.Symbol]: ...
    @classmethod
    def create(
        cls,
        qx: TensorBox,
        x_scale: TensorBox,
        x_zero_point: TensorBox,
        qw: TensorBox,
        w_scale,
        w_zero_point,
        qaccum: TensorBox,
        bias: TensorBox,
        stride: list[int],
        padding: list[int],
        dilation: list[int],
        groups: int,
        output_scale: TensorBox,
        output_zero_point: TensorBox,
        output_dtype,
        accum_scale,
        accum_zero_point,
        binary_attr,
        alpha,
        unary_attr,
        unary_scalars,
        unary_algorithm,
    ) -> Any: ...

class MKLPackedLinear(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=...) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(cls, x, packed_w, orig_w, B, batch_size) -> MKLPackedLinear: ...

class LinearUnary(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=...) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(cls, x, w, B, attr, scalars, algorithm) -> MultiOutput: ...
    def apply_constraint(self) -> None: ...

class LinearBinary(ExternKernelAlloc):
    kernel = ...
    def __init__(self, layout, inputs, constant_args=...) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(cls, x, y, w, B, attr) -> MultiOutput: ...
    def apply_constraint(self) -> None: ...

class QLinearPointwisePT2E(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=..., has_bias=...) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(
        cls,
        qx: TensorBox,
        x_scale: TensorBox,
        x_zero_point: TensorBox,
        qw: TensorBox,
        w_scale: TensorBox,
        w_zero_point: TensorBox,
        bias: TensorBox,
        output_scale: float,
        output_zero_point: int,
        output_dtype,
        post_op_name,
        post_op_args,
        post_op_algorithm,
    ) -> QLinearPointwisePT2E: ...

class QLinearPointwiseBinaryPT2E(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=..., has_bias=...) -> None: ...
    def codegen(self, wrapper) -> None: ...
    def get_mutation_names(self) -> Sequence[str]: ...
    @classmethod
    def create(
        cls,
        qx: TensorBox,
        x_scale: TensorBox,
        x_zero_point: TensorBox,
        qw: TensorBox,
        w_scale: TensorBox,
        w_zero_point: TensorBox,
        other: TensorBox,
        bias: TensorBox,
        output_scale: float,
        output_zero_point: int,
        output_dtype,
        other_scale,
        other_zp,
        binary_post_op,
        binary_alpha,
        unary_post_op,
        unary_post_op_args,
        unary_post_op_algorithm,
    ) -> QLinearPointwiseBinaryPT2E: ...

class MkldnnRnnLayer(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=...) -> None: ...
    @classmethod
    def create(
        cls,
        x: TensorBox,
        w0: TensorBox,
        w1: TensorBox,
        w2: TensorBox,
        w3: TensorBox,
        hx: TensorBox,
        cx: TensorBox,
        reverse: bool,
        batch_sizes: list[int],
        mode: int,
        hidden_size: int,
        num_layers: int,
        has_biases: bool,
        bidirectional: bool,
        batch_first: bool,
        train: bool,
    ) -> list[MultiOutput]: ...
    def codegen(self, wrapper) -> None: ...

class WeightInt4PackMatmul(ExternKernelAlloc):
    def __init__(self, layout, inputs, constant_args=...) -> None: ...
    def codegen(self, wrapper) -> None: ...
    @classmethod
    def create(
        cls,
        x: TensorBox,
        w: TensorBox,
        qGroupSize: TensorBox,
        qScalesAndZeros: TensorBox,
    ) -> WeightInt4PackMatmul: ...
