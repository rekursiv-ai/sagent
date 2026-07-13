from torch import nn

import torch
import torch.fx

from .configuration_gptj import GPTJConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, auto_docstring

"""PyTorch GPT-J model."""
logger = ...

def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor: ...
@torch.fx.wrap
def get_embed_positions(embed_positions, position_ids): ...
def rotate_every_two(x: torch.Tensor) -> torch.Tensor: ...
def apply_rotary_pos_emb(
    tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor: ...

class GPTJAttention(nn.Module):
    def __init__(self, config, layer_idx=...) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
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

class GPTJFlashAttention2(GPTJAttention):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
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

GPTJ_ATTENTION_CLASSES = ...

class GPTJMLP(nn.Module):
    def __init__(self, intermediate_size, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor | None) -> torch.FloatTensor: ...

class GPTJBlock(GradientCheckpointingLayer):
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
class GPTJPreTrainedModel(PreTrainedModel):
    config: GPTJConfig
    base_model_prefix = ...
    is_parallelizable = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _can_compile_fullgraph = ...
    _supports_param_buffer_assignment = ...
    def __init__(self, *inputs, **kwargs) -> None: ...

PARALLELIZE_DOCSTRING = ...
DEPARALLELIZE_DOCSTRING = ...

@auto_docstring
class GPTJModel(GPTJPreTrainedModel):
    def __init__(self, config) -> None: ...
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=...):  # -> None:
        ...
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):  # -> None:
        ...
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
    ) -> tuple | BaseModelOutputWithPast: ...

@auto_docstring(custom_intro=...)
class GPTJForCausalLM(GPTJPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=...):  # -> None:
        ...
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):  # -> None:
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
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPast: ...

@auto_docstring(custom_intro=...)
class GPTJForSequenceClassification(GPTJPreTrainedModel):
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
    ) -> tuple | SequenceClassifierOutputWithPast: ...

@auto_docstring
class GPTJForQuestionAnswering(GPTJPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        start_positions: torch.LongTensor | None = ...,
        end_positions: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | QuestionAnsweringModelOutput: ...

__all__ = [
    "GPTJForCausalLM",
    "GPTJForQuestionAnswering",
    "GPTJForSequenceClassification",
    "GPTJModel",
    "GPTJPreTrainedModel",
]
