from typing import Any
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_gpt2 import GPT2Config
from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch OpenAI GPT-2 model."""
logger = ...

def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path): ...
def eager_attention_forward(
    module, query, key, value, attention_mask, head_mask=..., **kwargs
):  # -> tuple[Tensor, Any]:
    ...

class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=..., layer_idx=...) -> None: ...
    def prune_heads(self, heads):  # -> None:
        ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]: ...

class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config) -> None: ...
    def forward(
        self, hidden_states: tuple[torch.FloatTensor] | None
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class GPT2Block(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        **kwargs,
    ) -> (
        tuple[torch.Tensor] | tuple[torch.Tensor, tuple[torch.FloatTensor, ...]] | None
    ): ...

class GPT2SequenceSummary(nn.Module):
    def __init__(self, config: GPT2Config) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        cls_index: torch.LongTensor | None = ...,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

@auto_docstring
class GPT2PreTrainedModel(PreTrainedModel):
    config: GPT2Config
    load_tf_weights = ...
    base_model_prefix = ...
    is_parallelizable = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    _skip_keys_device_placement = ...
    _supports_flash_attn = ...
    _supports_sdpa = ...
    _supports_attention_backend = ...
    _can_compile_fullgraph = ...
    def __init__(self, *inputs, **kwargs) -> None: ...

@dataclass
@auto_docstring(custom_intro=...)
class GPT2DoubleHeadsModelOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    mc_loss: torch.FloatTensor | None = ...
    logits: torch.FloatTensor | None = ...
    mc_logits: torch.FloatTensor | None = ...
    past_key_values: Cache | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

PARALLELIZE_DOCSTRING = ...
DEPARALLELIZE_DOCSTRING = ...

@auto_docstring
class GPT2Model(GPT2PreTrainedModel):
    _supports_param_buffer_assignment = ...
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
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring(custom_intro=...)
class GPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin):
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
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        encoder_hidden_states: torch.Tensor | None = ...,
        encoder_attention_mask: torch.FloatTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        logits_to_keep: int | torch.Tensor = ...,
        **kwargs,
    ) -> tuple | CausalLMOutputWithCrossAttentions: ...

@auto_docstring(custom_intro=...)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel, GenerationMixin):
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
        past_key_values: Cache | None = ...,
        cache_position: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        mc_token_ids: torch.LongTensor | None = ...,
        labels: torch.LongTensor | None = ...,
        mc_labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        **kwargs,
    ) -> tuple | GPT2DoubleHeadsModelOutput: ...

@auto_docstring(custom_intro=...)
class GPT2ForSequenceClassification(GPT2PreTrainedModel):
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
class GPT2ForTokenClassification(GPT2PreTrainedModel):
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
    ) -> tuple | TokenClassifierOutput: ...

@auto_docstring
class GPT2ForQuestionAnswering(GPT2PreTrainedModel):
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
    "GPT2DoubleHeadsModel",
    "GPT2ForQuestionAnswering",
    "GPT2ForSequenceClassification",
    "GPT2ForTokenClassification",
    "GPT2LMHeadModel",
    "GPT2Model",
    "GPT2PreTrainedModel",
    "load_tf_weights_in_gpt2",
]
