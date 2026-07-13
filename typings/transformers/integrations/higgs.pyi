import torch

"HIGGS through FLUTE (Flexible Lookup Table Engine for LUT-quantized LLMs) integration file"

def pad_to_block(tensor, dims, had_block_size, value=...):  # -> Tensor:
    ...
def get_higgs_grid(p: int, n: int) -> torch.Tensor: ...
def quantize_with_higgs(
    weight,
    bits: int = ...,
    p: int = ...,
    group_size: int = ...,
    hadamard_size: int = ...,
):  # -> dict[str, Any]:
    ...

class HiggsLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_bits: int,
        bias=...,
        dtype: torch.dtype | None = ...,
        device: torch.device | None = ...,
        group_size: int = ...,
        hadamard_size: int = ...,
    ) -> None: ...
    def forward(self, x): ...

def replace_with_higgs_linear(
    model,
    quantization_config=...,
    current_key_name=...,
    has_been_replaced=...,
    modules_to_not_convert=...,
):  # -> tuple[Any, bool | Any]:
    ...
def dequantize_higgs(model, current_key_name=...): ...
