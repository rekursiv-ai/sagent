from torch import nn

import torch

from .configuration_codegen import CodeGenConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch CodeGen model."""
logger = ...

def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor: ...
def rotate_every_two(x: torch.Tensor) -> torch.Tensor: ...
def apply_rotary_pos_emb(
    tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor: ...

class CodeGenAttention(nn.Module):
    def __init__(self, config, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor | None,
        layer_past: Cache | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> (
        tuple[torch.Tensor, tuple[torch.Tensor]]
        | tuple[torch.Tensor, tuple[torch.Tensor], tuple[torch.Tensor, ...]]
        | None
    ): ...

class CodeGenMLP(nn.Module):
    def __init__(self, intermediate_size, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor | None) -> torch.FloatTensor: ...

class CodeGenBlock(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor | None,
        layer_past: Cache | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> (
        tuple[torch.Tensor] | tuple[torch.Tensor, tuple[torch.FloatTensor, ...]] | None
    ): ...

@auto_docstring
class CodeGenPreTrainedModel(PreTrainedModel):
    config: CodeGenConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _can_compile_fullgraph = ...
    def __init__(self, *inputs, **kwargs) -> None: ...

@auto_docstring
class CodeGenModel(CodeGenPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | tuple[tuple[torch.Tensor]] | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast: ...

@auto_docstring(custom_intro=...)
class CodeGenForCausalLM(CodeGenPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | tuple[tuple[torch.Tensor]] | None = ...,
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
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast: ...

__all__ = ["CodeGenForCausalLM", "CodeGenModel", "CodeGenPreTrainedModel"]
