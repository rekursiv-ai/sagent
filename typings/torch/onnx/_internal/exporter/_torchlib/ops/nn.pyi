from torch.onnx._internal.exporter._torchlib._tensor_typing import TFloat, TReal
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl

"""torch.ops.aten operators under the `core` module."""
aten = ...
_INT64_MAX = ...
_INT64_MIN = ...

@onnx_impl(aten.gelu.default, trace_only=True, opset_introduced=20)
def aten_gelu_opset20(self: TReal, approximate: str = ...) -> TReal: ...
@onnx_impl(aten.group_norm.default, trace_only=True, opset_introduced=21)
def aten_group_norm(
    input: TFloat,
    num_groups: int,
    weight: TFloat | None = ...,
    bias: TFloat | None = ...,
    eps: float = ...,
    cudnn_enabled: bool = ...,
) -> TFloat: ...
@onnx_impl(aten.rms_norm.default, trace_only=True, opset_introduced=23)
def aten_rms_norm(
    input: TFloat,
    normalized_shape: list[int],
    weight: TFloat | None = ...,
    eps: float | None = ...,
) -> TFloat: ...
@onnx_impl(
    aten.scaled_dot_product_attention.default, trace_only=True, opset_introduced=23
)
def aten_scaled_dot_product_attention_23(
    query: TFloat,
    key: TFloat,
    value: TFloat,
    attn_mask: TFloat | None = ...,
    dropout_p: float = ...,
    is_causal: bool = ...,
    scale: float | None = ...,
    enable_gqa: bool = ...,
) -> TFloat: ...
