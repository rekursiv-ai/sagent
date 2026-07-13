from torch import Tensor, nn

import torch

from .configuration_gpt_neox_japanese import GPTNeoXJapaneseConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import dynamic_rope_update
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch GPTNeoX model."""
logger = ...

@auto_docstring
class GPTNeoXJapanesePreTrainedModel(PreTrainedModel):
    config: GPTNeoXJapaneseConfig
    base_model_prefix = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _can_compile_fullgraph = ...

class GPTNeoXJapaneseAttention(nn.Module):
    def __init__(self, config, use_bias=..., layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: torch.FloatTensor | None = ...,
        layer_past: Cache | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
    ):  # -> tuple[Any, Any, Parameter | None]:
        ...

class GPTNeoXJapaneseRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: GPTNeoXJapaneseConfig, device=...) -> None: ...
    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):  # -> tuple[Tensor, Tensor]:
        ...

def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=..., unsqueeze_dim=...
):  # -> tuple[Any, Any]:
    ...
def bias_dropout_add(
    x: Tensor, bias: Tensor, residual: Tensor | None, prob: float, training: bool
) -> Tensor: ...

class GPTNeoXJapaneseMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class GPTNeoXJapaneseLayer(nn.Module):
    def __init__(self, config, layer_number) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor | None,
        attention_mask: torch.FloatTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        layer_past: Cache | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = ...,
    ):  # -> tuple[Tensor, Any]:
        ...

@auto_docstring
class GPTNeoXJapaneseModel(GPTNeoXJapanesePreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        past_key_values: Cache | tuple[tuple[torch.FloatTensor]] | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple | BaseModelOutputWithPast: ...

@auto_docstring(custom_intro=...)
class GPTNeoXJapaneseForCausalLM(GPTNeoXJapanesePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> Linear:
        ...
    def set_output_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | tuple[tuple[torch.FloatTensor]] | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast: ...

__all__ = [
    "GPTNeoXJapaneseForCausalLM",
    "GPTNeoXJapaneseLayer",
    "GPTNeoXJapaneseModel",
    "GPTNeoXJapanesePreTrainedModel",
]
