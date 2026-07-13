from torch import nn

import torch

from .configuration_gpt_neo import GPTNeoConfig
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring

"""PyTorch GPT Neo model."""
_prepare_4d_causal_attention_mask = ...
logger = ...

def load_tf_weights_in_gpt_neo(model, config, gpt_neo_checkpoint_path): ...

class GPTNeoSelfAttention(nn.Module):
    def __init__(self, config, attention_type, layer_id=...) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        layer_past=...,
        head_mask=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Any, Any]:
        ...

class GPTNeoFlashAttention2(GPTNeoSelfAttention):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(
        self,
        hidden_states,
        attention_mask=...,
        layer_past=...,
        head_mask=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Any, Tensor | Any]:
        ...

GPT_NEO_ATTENTION_CLASSES = ...

class GPTNeoAttention(nn.Module):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(
        self,
        hidden_states,
        layer_past=...,
        attention_mask=...,
        head_mask=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> Any:
        ...

class GPTNeoMLP(nn.Module):
    def __init__(self, intermediate_size, config) -> None: ...
    def forward(self, hidden_states):  # -> Any:
        ...

class GPTNeoBlock(GradientCheckpointingLayer):
    def __init__(self, config, layer_id=...) -> None: ...
    def forward(
        self,
        hidden_states,
        layer_past=...,
        attention_mask=...,
        head_mask=...,
        use_cache=...,
        output_attentions=...,
        cache_position=...,
    ):  # -> tuple[Any, Any]:
        ...

@auto_docstring
class GPTNeoPreTrainedModel(PreTrainedModel):
    config: GPTNeoConfig
    load_tf_weights = ...
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _can_compile_fullgraph = ...
    def __init__(self, *inputs, **kwargs) -> None: ...

@auto_docstring
class GPTNeoModel(GPTNeoPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        past_key_values: Cache | tuple[torch.FloatTensor] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
    ) -> tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring(custom_intro=...)
class GPTNeoForCausalLM(GPTNeoPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ...
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        past_key_values: Cache | tuple[torch.FloatTensor] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.LongTensor | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor] | CausalLMOutputWithCrossAttentions: ...

@auto_docstring(custom_intro=...)
class GPTNeoForSequenceClassification(GPTNeoPreTrainedModel):
    def __init__(self, config) -> None: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = ...,
        past_key_values: Cache | tuple[torch.FloatTensor] | None = ...,
        attention_mask: torch.Tensor | None = ...,
        token_type_ids: torch.Tensor | None = ...,
        position_ids: torch.Tensor | None = ...,
        head_mask: torch.Tensor | None = ...,
        inputs_embeds: torch.Tensor | None = ...,
        labels: torch.Tensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple[torch.Tensor] | SequenceClassifierOutputWithPast: ...

@auto_docstring
class GPTNeoForTokenClassification(GPTNeoPreTrainedModel):
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
    ) -> tuple | TokenClassifierOutput: ...

@auto_docstring
class GPTNeoForQuestionAnswering(GPTNeoPreTrainedModel):
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
    "GPTNeoForCausalLM",
    "GPTNeoForQuestionAnswering",
    "GPTNeoForSequenceClassification",
    "GPTNeoForTokenClassification",
    "GPTNeoModel",
    "GPTNeoPreTrainedModel",
    "load_tf_weights_in_gpt_neo",
]
