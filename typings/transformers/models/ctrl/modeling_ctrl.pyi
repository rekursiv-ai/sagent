from torch import nn

import torch

from .configuration_ctrl import CTRLConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch CTRL model."""
logger = ...

def angle_defn(pos, i, d_model_size): ...
def positional_encoding(position, d_model_size, dtype):  # -> Tensor:
    ...
def scaled_dot_product_attention(
    q, k, v, mask, attention_mask=..., head_mask=...
):  # -> tuple[Tensor, Any | Tensor]:
    ...

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model_size, num_heads, layer_idx=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    def split_into_heads(self, x, batch_size): ...
    def forward(
        self,
        v,
        k,
        q,
        mask,
        layer_past=...,
        attention_mask=...,
        head_mask=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Any, Any | Tensor]:
        ...

def point_wise_feed_forward_network(d_model_size, dff):  # -> Sequential:
    ...

class EncoderLayer(nn.Module):
    def __init__(
        self, d_model_size, num_heads, dff, rate=..., layer_idx=...
    ) -> None: ...
    def forward(
        self,
        x,
        mask,
        layer_past=...,
        attention_mask=...,
        head_mask=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> Any:
        ...

@auto_docstring
class CTRLPreTrainedModel(PreTrainedModel):
    config: CTRLConfig
    base_model_prefix = ...

@auto_docstring
class CTRLModel(CTRLPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPast: ...

@auto_docstring(custom_intro=...)
class CTRLLMHeadModel(CTRLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | CausalLMOutputWithPast: ...
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=..., use_cache=..., **kwargs
    ):  # -> dict[str, Any | None]:
        ...

@auto_docstring(custom_intro=...)
class CTRLForSequenceClassification(CTRLPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutput: ...

__all__ = [
    "CTRLForSequenceClassification",
    "CTRLLMHeadModel",
    "CTRLModel",
    "CTRLPreTrainedModel",
]
