from typing import Any
from dataclasses import dataclass

from torch import nn

import torch

from .configuration_clvp import ClvpConfig, ClvpDecoderConfig
from ...cache_utils import Cache
from ...generation import GenerationConfig, GenerationMixin
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    CausalLMOutputWithCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring
from ...utils.deprecation import deprecate_kwarg

"""PyTorch CLVP model."""
logger = ...

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor: ...
def clvp_loss(similarity: torch.Tensor) -> torch.Tensor: ...
def rotate_half(x):  # -> Tensor:
    ...
def apply_rotary_pos_emb(
    q, k, v, cos, sin, position_ids, unsqueeze_dim=...
):  # -> tuple[Any, Any, Any]:
    ...

@dataclass
@auto_docstring(custom_intro=...)
class ClvpEncoderOutput(ModelOutput):
    embeds: torch.FloatTensor | None = ...
    last_hidden_state: torch.FloatTensor | None = ...
    pooler_output: torch.FloatTensor | None = ...
    hidden_states: tuple[torch.FloatTensor] | None = ...
    attentions: tuple[torch.FloatTensor] | None = ...

@dataclass
@auto_docstring
class ClvpOutput(ModelOutput):
    loss: torch.FloatTensor | None = ...
    speech_ids: torch.LongTensor | None = ...
    logits_per_speech: torch.FloatTensor | None = ...
    logits_per_text: torch.FloatTensor | None = ...
    text_embeds: torch.FloatTensor | None = ...
    speech_embeds: torch.FloatTensor | None = ...
    text_model_output: BaseModelOutputWithPooling = ...
    speech_model_output: BaseModelOutputWithPooling = ...
    decoder_hidden_states: torch.FloatTensor | None = ...
    text_encoder_hidden_states: torch.FloatTensor | None = ...
    speech_encoder_hidden_states: torch.FloatTensor | None = ...

class ClvpRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None: ...
    def forward(self, hidden_states): ...
    def extra_repr(self):  # -> str:
        ...

class ClvpRotaryPositionalEmbedding(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class ClvpSelfAttention(nn.Module):
    def __init__(self, config, layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        rotary_pos_emb: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        past_key_values: Cache | None = ...,
        use_cache: bool | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor | None,
        tuple[torch.FloatTensor] | None,
    ]: ...

class ClvpGatedLinearUnit(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class ClvpEncoderMLP(nn.Module):
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class ClvpEncoderLayer(nn.Module):
    def __init__(self, config: ClvpConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        rotary_pos_emb: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        output_attentions: bool | None = ...,
    ) -> tuple[torch.FloatTensor]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.FloatTensor]: ...

class ClvpSequenceSummary(nn.Module):
    def __init__(self, config: ClvpConfig) -> None: ...
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        cls_index: torch.LongTensor | None = ...,
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class ClvpDecoderMLP(nn.Module):
    def __init__(self, intermediate_size, config) -> None: ...
    def forward(
        self, hidden_states: tuple[torch.FloatTensor] | None
    ) -> torch.FloatTensor: ...
    def __call__(self, *args: Any, **kwargs: Any) -> torch.FloatTensor: ...

class ClvpDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx=...) -> None: ...
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> (
        tuple[torch.Tensor] | tuple[torch.Tensor, tuple[torch.FloatTensor, ...]] | None
    ): ...

class ClvpConditioningEncoder(nn.Module):
    def __init__(self, config: ClvpConfig) -> None: ...
    def compute_groupnorm_groups(self, channels: int, groups: int = ...):  # -> int:
        ...
    def forward(
        self,
        input_features: torch.FloatTensor,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ):  # -> Tensor:
        ...

@auto_docstring
class ClvpPreTrainedModel(PreTrainedModel):
    config: ClvpConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _skip_keys_device_placement = ...

class ClvpEncoder(ClvpPreTrainedModel):
    def __init__(self, config: ClvpConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding | Module:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        inputs_embeds: torch.LongTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
    ) -> tuple | BaseModelOutput: ...
    def __call__(self, *args: Any, **kwargs: Any) -> tuple | BaseModelOutput: ...

class ClvpDecoder(ClvpPreTrainedModel):
    def __init__(self, config) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring
class ClvpModel(ClvpPreTrainedModel):
    def __init__(self, config: ClvpDecoderConfig) -> None: ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, value):  # -> None:
        ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        attention_mask: torch.FloatTensor | None = ...,
        token_type_ids: torch.LongTensor | None = ...,
        position_ids: torch.LongTensor | None = ...,
        head_mask: torch.FloatTensor | None = ...,
        past_key_values: Cache | None = ...,
        inputs_embeds: torch.FloatTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions: ...

@auto_docstring(custom_intro=...)
class ClvpForCausalLM(ClvpPreTrainedModel, GenerationMixin):
    def __init__(self, config) -> None: ...
    def get_output_embeddings(self):  # -> None:
        ...
    def get_input_embeddings(self):  # -> Embedding:
        ...
    def set_input_embeddings(self, new_embeddings):  # -> None:
        ...
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=...,
        inputs_embeds=...,
        conditioning_embeds=...,
        cache_position=...,
        **kwargs,
    ):  # -> dict[Any, Any]:
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
        labels: torch.LongTensor | None = ...,
        use_cache: bool | None = ...,
        output_attentions: bool | None = ...,
        output_hidden_states: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | CausalLMOutputWithCrossAttentions: ...

@auto_docstring(custom_intro=...)
class ClvpModelForConditionalGeneration(ClvpPreTrainedModel, GenerationMixin):
    config: ClvpConfig
    def __init__(self, config: ClvpConfig) -> None: ...
    def fix_speech_decoder_output(
        self, speech_ids: torch.LongTensor
    ) -> torch.LongTensor: ...
    def get_text_features(
        self,
        input_ids: torch.LongTensor | None = ...,
        text_encoder_inputs_embeds: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
    ) -> torch.FloatTensor: ...
    def get_speech_features(
        self,
        speech_ids: torch.LongTensor | None = ...,
        input_ids: torch.LongTensor | None = ...,
        input_features: torch.FloatTensor | None = ...,
        conditioning_encoder_inputs_embeds: torch.FloatTensor | None = ...,
        attention_mask: torch.Tensor | None = ...,
        generation_config: GenerationConfig | None = ...,
        **kwargs,
    ) -> torch.FloatTensor: ...
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = ...,
        input_features: torch.FloatTensor | None = ...,
        conditioning_encoder_inputs_embeds: torch.FloatTensor | None = ...,
        text_encoder_inputs_embeds: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        return_loss: bool | None = ...,
        output_hidden_states: bool | None = ...,
        output_attentions: bool | None = ...,
        return_dict: bool | None = ...,
        cache_position: torch.Tensor | None = ...,
    ) -> tuple | ClvpOutput: ...
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor | None = ...,
        input_features: torch.FloatTensor | None = ...,
        attention_mask: torch.LongTensor | None = ...,
        generation_config: GenerationConfig | None = ...,
        pad_to_max_mel_tokens: int | None = ...,
        output_hidden_states: bool | None = ...,
        **kwargs,
    ):  # -> tuple[LongTensor, Tensor, Tensor, Any, Any, Any, Any, Any | Tensor, Any, Any] | tuple[LongTensor, Tensor, Tensor, Any, Any, Any, Any] | ClvpOutput:
        ...

__all__ = [
    "ClvpDecoder",
    "ClvpEncoder",
    "ClvpForCausalLM",
    "ClvpModel",
    "ClvpModelForConditionalGeneration",
    "ClvpPreTrainedModel",
]
