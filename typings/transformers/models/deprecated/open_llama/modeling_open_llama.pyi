from torch import nn

import torch

from .configuration_open_llama import OpenLlamaConfig
from ....cache_utils import Cache
from ....modeling_layers import GradientCheckpointingLayer
from ....modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ....modeling_utils import PreTrainedModel
from ....utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ....utils.deprecation import deprecate_kwarg

"""PyTorch Open-Llama model."""
logger = ...
_CONFIG_FOR_DOC = ...

class OpenLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class OpenLlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor
    def __init__(
        self, dim, max_position_embeddings=..., base=..., device=...
    ) -> None: ...
    def forward(self, x, seq_len=...):  # -> tuple[Tensor, Tensor]:
        ...

class OpenLlamaLinearScalingRotaryEmbedding(OpenLlamaRotaryEmbedding):
    def __init__(
        self, dim, max_position_embeddings=..., base=..., device=..., scaling_factor=...
    ) -> None: ...

class OpenLlamaDynamicNTKScalingRotaryEmbedding(OpenLlamaRotaryEmbedding):
    def __init__(
        self, dim, max_position_embeddings=..., base=..., device=..., scaling_factor=...
    ) -> None: ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids, unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...

class OpenLlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        dropout_prob: float,
    ) -> None: ...
    def forward(self, x):  # -> Any:
        ...

class OpenLlamaAttention(nn.Module):
    def __init__(self, config: OpenLlamaConfig) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool = ...,
        use_cache: bool = ...,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]: ...

class OpenLlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: OpenLlamaConfig) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        output_attentions: bool | None = ...,
        use_cache: bool | None = ...,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None
    ]: ...

OPEN_LLAMA_START_DOCSTRING = ...

@add_start_docstrings(
    ...,
    OPEN_LLAMA_START_DOCSTRING,
)
class OpenLlamaPreTrainedModel(PreTrainedModel):
    config: OpenLlamaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...

OPEN_LLAMA_INPUTS_DOCSTRING = ...

@add_start_docstrings(
    ...,
    OPEN_LLAMA_START_DOCSTRING,
)
class OpenLlamaModel(OpenLlamaPreTrainedModel):
    def __init__(self, config: OpenLlamaConfig) -> None: ...
    @add_start_docstrings_to_model_forward(OPEN_LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutputWithPast: ...

class OpenLlamaForCausalLM(OpenLlamaPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(OPEN_LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | CausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        attention_mask=...,
        inputs_embeds=...,
        **kwargs,
    ):  # -> dict[str, Any]:
        ...

@add_start_docstrings(
    ...,
    OPEN_LLAMA_START_DOCSTRING,
)
class OpenLlamaForSequenceClassification(OpenLlamaPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings_to_model_forward(OPEN_LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | SequenceClassifierOutputWithPast: ...

__all__ = [
    "OpenLlamaForCausalLM",
    "OpenLlamaForSequenceClassification",
    "OpenLlamaModel",
    "OpenLlamaPreTrainedModel",
]
