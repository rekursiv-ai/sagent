from typing import Any

from torch import nn

import torch

from .configuration_recurrent_gemma import RecurrentGemmaConfig
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithNoAttention, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch RecurrentGemma model."""
logger = ...
_MAX_SQRT_GRADIENT = ...

class RecurrentGemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = ...) -> None: ...
    def forward(self, x): ...
    def extra_repr(self):  # -> str:
        ...

class RecurrentGemmaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, dim, base=..., device=...) -> None: ...
    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=...):  # -> tuple[Tensor, Tensor]:
        ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: ...

class RecurrentGemmaSdpaAttention(nn.Module):
    def __init__(self, config: RecurrentGemmaConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        use_cache: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class SqrtBoundDerivative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor: ...
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor: ...

class RecurrentGemmaRglru(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self, activations: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class RecurrentGemmaRecurrentBlock(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(
        self,
        input_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        use_cache: bool = ...,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...

TEMPORAL_BLOCK_CLASSES = ...

class RecurrentGemmaMlp(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class RecurrentGemmaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx) -> None: ...
    def forward(
        self,
        activations: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...

@auto_docstring
class RecurrentGemmaPreTrainedModel(PreTrainedModel):
    config: RecurrentGemmaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    def reset_cache(self, batch, device, dtype):  # -> None:
        ...

@auto_docstring
class RecurrentGemmaModel(RecurrentGemmaPreTrainedModel):
    def __init__(self, config: RecurrentGemmaConfig) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithNoAttention: ...

@auto_docstring
class RecurrentGemmaForCausalLM(RecurrentGemmaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        use_cache: bool | None = ...,
        **kwargs,
    ) -> tuple | CausalLMOutput: ...

__all__ = [
    "RecurrentGemmaForCausalLM",
    "RecurrentGemmaModel",
    "RecurrentGemmaPreTrainedModel",
]
