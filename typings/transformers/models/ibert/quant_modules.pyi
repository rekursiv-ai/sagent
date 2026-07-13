from torch import nn
from torch.autograd import Function

logger = ...

class QuantEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=...,
        max_norm=...,
        norm_type=...,
        scale_grad_by_freq=...,
        sparse=...,
        _weight=...,
        weight_bit=...,
        momentum=...,
        quant_mode=...,
    ) -> None: ...
    def forward(
        self, x, positions=..., incremental_state=...
    ):  # -> tuple[Tensor, None] | tuple[Any, Any]:
        ...

class QuantAct(nn.Module):
    def __init__(
        self,
        activation_bit,
        act_range_momentum=...,
        per_channel=...,
        channel_len=...,
        quant_mode=...,
    ) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def forward(
        self,
        x,
        pre_act_scaling_factor=...,
        identity=...,
        identity_scaling_factor=...,
        specified_min=...,
        specified_max=...,
    ):  # -> tuple[Any, None] | tuple[Any, Any]:
        ...

class QuantLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=...,
        weight_bit=...,
        bias_bit=...,
        per_channel=...,
        quant_mode=...,
    ) -> None: ...
    def __repr__(self):  # -> str:
        ...
    def forward(
        self, x, prev_act_scaling_factor=...
    ):  # -> tuple[Tensor, None] | tuple[Any, Any]:
        ...

class IntGELU(nn.Module):
    def __init__(self, quant_mode=..., force_dequant=...) -> None: ...
    def int_erf(self, x_int, scaling_factor):  # -> tuple[Any | None, Any]:
        ...
    def forward(self, x, scaling_factor=...):  # -> tuple[Any, None] | tuple[Any, Any]:
        ...

class IntSoftmax(nn.Module):
    def __init__(self, output_bit, quant_mode=..., force_dequant=...) -> None: ...
    def int_polynomial(self, x_int, scaling_factor):  # -> tuple[Any, Any]:
        ...
    def int_exp(self, x_int, scaling_factor):  # -> tuple[Any, Any]:
        ...
    def forward(self, x, scaling_factor):  # -> tuple[Tensor, None] | tuple[Any, float]:
        ...

class IntLayerNorm(nn.Module):
    def __init__(
        self, normalized_shape, eps, output_bit=..., quant_mode=..., force_dequant=...
    ) -> None: ...
    def set_shift(self, y_int):  # -> None:
        ...
    def overflow_fallback(self, y_int): ...
    def forward(
        self, x, scaling_factor=...
    ):  # -> tuple[Any, None] | tuple[Any, Tensor]:
        ...

def get_percentile_min_max(
    input, lower_percentile, upper_percentile, output_tensor=...
):  # -> tuple[Number | Tensor, Number | Tensor]:
    ...
def linear_quantize(input, scale, zero_point, inplace=...):  # -> Tensor:
    ...
def symmetric_linear_quantization_params(
    num_bits, saturation_min, saturation_max, per_channel=...
): ...

class SymmetricQuantFunction(Function):
    @staticmethod
    def forward(ctx, x, k, percentile_mode, scale):  # -> Tensor:
        ...
    @staticmethod
    def backward(ctx, grad_output):  # -> tuple[Any, None, None, None, None]:
        ...

class floor_ste(Function):
    @staticmethod
    def forward(ctx, x):  # -> Tensor:
        ...
    @staticmethod
    def backward(ctx, grad_output): ...

class round_ste(Function):
    @staticmethod
    def forward(ctx, x):  # -> Tensor:
        ...
    @staticmethod
    def backward(ctx, grad_output): ...

def batch_frexp(inputs, max_bit=...):  # -> tuple[Tensor, Tensor]:
    ...

class FixedPointMul(Function):
    @staticmethod
    def forward(
        ctx,
        pre_act,
        pre_act_scaling_factor,
        bit_num,
        z_scaling_factor,
        identity=...,
        identity_scaling_factor=...,
    ):  # -> Tensor:
        ...
    @staticmethod
    def backward(
        ctx, grad_output
    ):  # -> tuple[Any, None, None, None, None, Any | None, None]:
        ...
