from torch import nn

import torch

logger = ...
VALUES_PER_ITEM = ...

def pack_weights(quantized_weights: torch.Tensor) -> torch.Tensor: ...
@torch.compile
def unpack_weights(packed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor: ...

class BitLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device=...,
        dtype=...,
        use_rms_norm: bool = ...,
        rms_norm_eps: float = ...,
    ) -> None: ...
    @torch.compile
    def activation_quant(self, input, num_bits=...):  # -> tuple[Any, Any]:
        ...
    @torch.compile
    def post_quant_process(self, input, input_scale, weight_scale): ...
    def forward(self, input): ...

class WeightQuant(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, weight): ...
    @staticmethod
    def backward(ctx, grad_output): ...

class ActQuant(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(ctx, activation): ...
    @staticmethod
    def backward(ctx, grad_output): ...

class AutoBitLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = ...,
        device=...,
        dtype=...,
        online_quant: bool = ...,
        use_rms_norm: bool = ...,
        rms_norm_eps: float = ...,
    ) -> None: ...
    def load_hook(self, state_dict, prefix, *args, **kwargs): ...
    def forward(self, input):  # -> Tensor:
        ...

def replace_with_bitnet_linear(
    model,
    modules_to_not_convert=...,
    current_key_name=...,
    quantization_config=...,
    pre_quantized=...,
): ...
